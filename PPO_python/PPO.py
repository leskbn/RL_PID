import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import numpy as np


class RunningMeanStd:
    def __init__(self, shape, eps=1e-5):
        # CPU에 통계 저장 (안전하고 가벼움)
        if isinstance(shape, int):
            shape = (shape,)
        self.mean = torch.zeros(shape, dtype=torch.float32)
        self.var = torch.ones(shape, dtype=torch.float32)
        self.count = eps

    @torch.no_grad()
    def update(self, x):
        """
        x: torch.Tensor on any device, shape (..., D)
        한 스텝(state) 또는 미니배치 모두 지원
        """
        x = x.detach().to("cpu")
        if x.dim() == 1:
            batch_mean = x
            batch_var = torch.zeros_like(x)
            batch_count = 1.0
        else:
            # 마지막 차원을 feature로 간주
            batch_mean = x.mean(dim=0)
            batch_var = x.var(dim=0, unbiased=False)
            batch_count = x.shape[0]

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * (batch_count / tot_count)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta.pow(2) * self.count * batch_count / tot_count

        self.mean = new_mean
        self.var = M2 / tot_count
        self.count = tot_count

    def normalize(self, x):
        # x: torch.Tensor on device
        m = self.mean.to(x.device)
        s = (self.var.to(x.device).clamp_min(1e-8)).sqrt()
        z = (x - m) / s
        # 과도한 값 방지 (선택)
        return torch.clamp(z, -5.0, 5.0)


################################## set device ##################################
print(
    "============================================================================================"
)
# set device to cpu or cuda
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print(
    "============================================================================================"
)


################################## PPO Policy ##################################
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []  # true 종료(terminated)
        self.is_timeouts = []  # 시간초과(truncated)
        self.timeout_bootstrap = []
        self.last_value = None  # 롤아웃 마지막 V(s_T) 저장용

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]
        del self.is_timeouts[:]
        del self.timeout_bootstrap[:]
        self.last_value = None


class ActorCritic(nn.Module):
    def __init__(
        self, state_dim, action_dim, has_continuous_action_space, action_std_init
    ):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_dim = action_dim
            self.log_std = nn.Parameter(
                torch.full((action_dim,), np.log(action_std_init), dtype=torch.float32)
            )
        # actor
        if has_continuous_action_space:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 32),
                nn.Tanh(),
                nn.Linear(32, 32),
                nn.Tanh(),
                nn.Linear(32, action_dim),
            )
        else:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, action_dim),
                nn.Softmax(dim=-1),
            )
        # critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            with torch.no_grad():
                self.log_std.data[:] = np.log(new_action_std)
        else:
            print("WARNING : set_action_std() on discrete policy")

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        if self.has_continuous_action_space:
            mu = self.actor(state)  # unbounded
            log_std = torch.clamp(self.log_std, -5.0, 2.0)
            std = log_std.exp().expand_as(mu)
            # 독립정규(각 차원)
            dist = torch.distributions.Normal(mu, std)

            u = dist.rsample()  # reparameterized
            a = torch.tanh(u)
            # log_prob 보정: sum over dims
            log_prob_u = dist.log_prob(u).sum(dim=-1)
            log_prob = log_prob_u - torch.log(1 - a.pow(2) + 1e-6).sum(dim=-1)

            state_val = self.critic(state)
            return a.detach(), log_prob.detach(), state_val.detach()
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)
        return action.detach(), action_logprob.detach(), state_val.detach()

    def evaluate(self, state, action):
        if self.has_continuous_action_space:
            mu = self.actor(state)  # unbounded
            log_std = torch.clamp(self.log_std, -5.0, 2.0)
            std = log_std.exp().expand_as(mu)
            dist = torch.distributions.Normal(mu, std)

            a = torch.clamp(action, -1 + 1e-6, 1 - 1e-6)
            u = 0.5 * torch.log((1 + a) / (1 - a))  # atanh(a)

            log_prob_u = dist.log_prob(u).sum(dim=-1)
            log_prob = log_prob_u - torch.log(1 - a.pow(2) + 1e-6).sum(dim=-1)

            state_values = self.critic(state)
            dist_entropy = dist.entropy().sum(dim=-1)  # 각 차원 합
            return log_prob, state_values, dist_entropy
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)
            action = action.long().view(-1)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy


class PPO:
    def __init__(
        self,
        state_dim,
        action_dim,
        lr_actor,
        lr_critic,
        gamma,
        K_epochs,
        eps_clip,
        has_continuous_action_space,
        action_std_init=0.6,
    ):

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(
            state_dim, action_dim, has_continuous_action_space, action_std_init
        ).to(device)
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.policy.actor.parameters(), "lr": lr_actor},
                {"params": [self.policy.log_std], "lr": lr_actor},
                {"params": self.policy.critic.parameters(), "lr": lr_critic},
            ]
        )

        self.policy_old = ActorCritic(
            state_dim, action_dim, has_continuous_action_space, action_std_init
        ).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

        self.obs_rms = RunningMeanStd(shape=state_dim)

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
        else:
            print(
                "WARNING : Calling PPO::set_action_std() on discrete action space policy"
            )

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print(
            "--------------------------------------------------------------------------------------------"
        )
        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if self.action_std <= min_action_std:
                self.action_std = min_action_std
                print(
                    "setting actor output action_std to min_action_std : ",
                    self.action_std,
                )
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            print(
                "WARNING : Calling PPO::decay_action_std() on discrete action space policy"
            )
        print(
            "--------------------------------------------------------------------------------------------"
        )

    def select_action(self, state):
        with torch.no_grad():
            state_t = torch.as_tensor(state, dtype=torch.float32, device=device)
            self.obs_rms.update(state_t)
            s_norm = self.obs_rms.normalize(state_t)

            action, action_logprob, state_val = self.policy_old.act(s_norm)

        self.buffer.states.append(s_norm)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)

        if self.has_continuous_action_space:
            self.buffer.actions.append(action)
            return action.detach().cpu().numpy().flatten()
        else:
            act_int = int(action.item())
            self.buffer.actions.append(torch.as_tensor(act_int))
            return act_int

    def update(
        self,
        gae_lambda: float = 0.95,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        minibatch_size: int = 256,
        target_kl: float = None,
    ):
        # 텐서화
        old_states = torch.stack(self.buffer.states, dim=0).detach().to(device).float()
        old_actions = torch.stack(self.buffer.actions, dim=0).detach().to(device)
        old_logprobs = torch.stack(self.buffer.logprobs, dim=0).detach().to(device)
        old_values = (
            torch.stack(self.buffer.state_values, dim=0).detach().to(device).squeeze(-1)
        )
        rewards = torch.tensor(self.buffer.rewards, dtype=torch.float32, device=device)
        dones_term = torch.tensor(
            self.buffer.is_terminals, dtype=torch.float32, device=device
        )
        timeouts = torch.tensor(
            self.buffer.is_timeouts, dtype=torch.float32, device=device
        )
        tb = torch.tensor(
            self.buffer.timeout_bootstrap, dtype=torch.float32, device=device
        )

        if not self.has_continuous_action_space:
            old_actions = old_actions.long().view(-1)

        # --- GAE(λ)
        T = rewards.size(0)
        advantages = torch.zeros(T, dtype=torch.float32, device=device)

        # ★ 롤아웃 끝의 V(s_T)로 부트스트랩할 값
        last_v = (
            self.buffer.last_value
            if self.buffer.last_value is not None
            else torch.zeros((), device=device)
        )  # (안 세팅됐으면 0으로라도)

        last_gae = 0.0
        for t in reversed(range(T)):
            # 다음 상태의 V: 마지막 스텝은 last_v, 그 외엔 old_values[t+1]
            if t == T - 1:
                next_value = last_v
            elif timeouts[t] > 0.5:
                next_value = tb[t]
            else:
                next_value = old_values[t + 1]

            nonterminal = 1.0 - dones_term[t]  # terminated일 때만 0으로 컷
            delta = rewards[t] + self.gamma * next_value * nonterminal - old_values[t]
            last_gae = delta + self.gamma * gae_lambda * nonterminal * last_gae
            advantages[t] = last_gae

        returns = advantages + old_values

        # --- advantage만 정규화
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # --- 미니배치 설정
        N = T
        if minibatch_size is None or minibatch_size > N:
            minibatch_size = N

        idx = torch.randperm(N, device=device)

        for _ in range(self.K_epochs):
            # 에폭마다 셔플
            idx = idx[torch.randperm(N, device=device)]
            for start in range(0, N, minibatch_size):
                mb = idx[start : start + minibatch_size]
                mb_states = old_states[mb]
                mb_actions = old_actions[mb]
                mb_old_logprobs = old_logprobs[mb]
                mb_adv = advantages[mb]
                mb_returns = returns[mb]

                logprobs, state_values, dist_entropy = self.policy.evaluate(
                    mb_states, mb_actions
                )
                state_values = state_values.squeeze(-1)

                ratios = torch.exp(logprobs - mb_old_logprobs)
                surr1 = ratios * mb_adv
                surr2 = (
                    torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * mb_adv
                )

                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = self.MseLoss(state_values, mb_returns)
                entropy_loss = -dist_entropy.mean()

                loss = policy_loss + vf_coef * value_loss + ent_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), max_grad_norm
                )  # ★
                self.optimizer.step()

            # (옵션) KL 모니터링/조기종료
            if target_kl is not None:
                with torch.no_grad():
                    new_logprobs, _, _ = self.policy.evaluate(old_states, old_actions)
                    approx_kl = (old_logprobs - new_logprobs).mean().abs().item()
                if approx_kl > target_kl:
                    # print(f"Early stop at epoch due to KL: {approx_kl:.4f} > {target_kl:.4f}")
                    break

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(
            torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        )
        self.policy.load_state_dict(
            torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        )
