import os
import glob
import time
from datetime import datetime
from collections import deque

import matplotlib.pyplot as plt  # ← 추가

import torch
import numpy as np

import gymnasium as gym

from PPO import PPO
from gymnasium.wrappers import RescaleAction, ClipAction

from envs.pid_dc_motor_env import PIDTuningDCMotorEnv


def _mod_device(ppo_agent):
    # policy_old 파라미터가 올라가 있는 디바이스를 추출
    return next(ppo_agent.policy_old.parameters()).device


def _value_of(ppo_agent, state):
    dev = _mod_device(ppo_agent)
    with torch.no_grad():
        s = torch.as_tensor(state, dtype=torch.float32, device=dev)
        s_n = ppo_agent.obs_rms.normalize(s)
        v = ppo_agent.policy_old.critic(s_n).squeeze(-1)
    return v


# --- persistent plot (no flicker) ---
_PLOT = {"fig": None}


def _ensure_plot():
    if _PLOT["fig"] is not None:
        return
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 4), sharex=True)
    (lw,) = ax1.plot([], [], label="ω(t)")
    lref = ax1.axhline(0.0, ls="--", lw=1, color="k", label="ω_ref")
    (lu,) = ax2.plot([], [], label="u(t)=V")
    ax1.set_ylabel("ω [rad/s]")
    ax1.grid(True)
    ax1.legend(loc="best")
    ax2.set_ylabel("V [V]")
    ax2.set_xlabel("time [s]")
    ax2.grid(True)
    fig.tight_layout()
    _PLOT.update(fig=fig, ax1=ax1, ax2=ax2, lw=lw, lu=lu, lref=lref)


def _update_plot_episode(info, i_episode=None):
    if "t" not in info:  # PID 환경 아닐 때 자동 skip
        return
    _ensure_plot()
    t, w, u = info["t"], info["w"], info["u"]
    wref = info.get("w_ref")

    ax1, ax2 = _PLOT["ax1"], _PLOT["ax2"]
    _PLOT["lw"].set_data(t, w)
    _PLOT["lu"].set_data(t, u)

    if wref is not None and len(t) > 0:
        _PLOT["lref"].set_ydata([wref, wref])
        _PLOT["lref"].set_xdata([t[0], t[-1]])

    if len(t) > 0:
        ax1.set_xlim(t[0], t[-1])
    ax1.relim()
    ax1.autoscale_view(scalex=False, scaley=True)
    ax2.relim()
    ax2.autoscale_view(scalex=False, scaley=True)

    if i_episode is not None:
        kp, ki, kd = info.get("Kp"), info.get("Ki"), info.get("Kd")
        if kp is not None and ki is not None and kd is not None:
            _PLOT["fig"].suptitle(
                f"Episode {i_episode} | Kp={kp:.3f}, Ki={ki:.3f}, Kd={kd:.3f}"
            )

    _PLOT["fig"].canvas.draw_idle()
    _PLOT["fig"].canvas.flush_events()
    plt.pause(0.001)


################################### Training ###################################
def train():
    print(
        "============================================================================================"
    )

    ####### initialize environment hyperparameters ######
    env_name = "BipedalWalker-v3"

    # has_continuous_action_space = True  # continuous action space; else discrete

    max_ep_len = 10  # max timesteps in one episode
    max_training_timesteps = int(
        3e6
    )  # break training loop if timeteps > max_training_timesteps

    print_freq = max_ep_len * 10  # print avg reward in the interval (in num timesteps)
    log_freq = max_ep_len * 2  # log avg reward in the interval (in num timesteps)
    save_model_freq = int(1e5)  # save model frequency (in num timesteps)

    action_std = 0.6  # starting std for action distribution (Multivariate Normal)
    # action_std_decay_rate = 0.05  # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    # min_action_std = (
    #     0.1  # minimum action_std (stop decay after action_std <= min_action_std)
    # )
    # action_std_decay_freq = int(2.5e5)  # action_std decay frequency (in num timesteps)
    #####################################################

    ## Note : print/log frequencies should be > than max_ep_len

    ################ PPO hyperparameters ################
    update_timestep = max_ep_len * 2  # update policy every n timesteps
    K_epochs = 5  # update policy for K epochs in one PPO update

    eps_clip = 0.2  # clip parameter for PPO
    gamma = 0.99  # discount factor

    lr_actor = 3e-4  # learning rate for actor network
    lr_critic = 3e-4  # learning rate for critic network

    random_seed = 1234  # set random seed if required (0 = no random seed)
    #####################################################

    print("training environment name : " + env_name)

    # env = gym.make(env_name, hardcore=True)

    env = PIDTuningDCMotorEnv(
        dt=0.01,
        T_eval=10.0,
        w_ref=40.0,
        Vmax=12.0,
        domain_rand=True,  # 도메인 랜덤화 원하면 True
        return_history=False,  # 에피소드 끝에 시간응답 히스토리 info로 받기
        history_stride=2,  # 길면 2~5로 다운샘플
    )

    is_box = isinstance(env.action_space, gym.spaces.Box)
    has_continuous_action_space = is_box

    # 연속이면: 에이전트 [-1,1] → env.low/high 로 자동 매핑
    if is_box:
        env = RescaleAction(env, -1.0, 1.0)
        env = ClipAction(env)

    # dim
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] if is_box else env.action_space.n

    ###################### logging ######################
    # region
    #### log files for multiple runs are NOT overwritten
    log_dir = "PPO_Python/PPO_logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_dir = log_dir + "/" + env_name + "/"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    #### get number of log files in log directory
    run_num = 0
    current_num_files = next(os.walk(log_dir))[2]
    run_num = len(current_num_files)

    #### create new log file for each run
    log_f_name = log_dir + "/PPO_" + env_name + "_log_" + str(run_num) + ".csv"

    print("current logging run number for " + env_name + " : ", run_num)
    print("logging at : " + log_f_name)
    #####################################################
    # endregion

    ################### checkpointing ###################
    # region
    run_num_pretrained = (
        0  #### change this to prevent overwriting weights in same env_name folder
    )

    directory = "PPO_Python/PPO_preTrained"
    if not os.path.exists(directory):
        os.makedirs(directory)

    directory = directory + "/" + env_name + "/"
    if not os.path.exists(directory):
        os.makedirs(directory)

    checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(
        env_name, random_seed, run_num_pretrained
    )
    print("save checkpoint path : " + checkpoint_path)
    #####################################################
    # endregion

    ############# print all hyperparameters #############
    # region
    print(
        "--------------------------------------------------------------------------------------------"
    )
    print("max training timesteps : ", max_training_timesteps)
    print("max timesteps per episode : ", max_ep_len)
    print("model saving frequency : " + str(save_model_freq) + " timesteps")
    print("log frequency : " + str(log_freq) + " timesteps")
    print(
        "printing average reward over episodes in last : "
        + str(print_freq)
        + " timesteps"
    )
    print(
        "--------------------------------------------------------------------------------------------"
    )
    print("state space dimension : ", state_dim)
    print("action space dimension : ", action_dim)
    print(
        "--------------------------------------------------------------------------------------------"
    )
    if has_continuous_action_space:
        print("Initializing a continuous action space policy")
        print(
            "--------------------------------------------------------------------------------------------"
        )
        print("starting std of action distribution : ", action_std)
    else:
        print("Initializing a discrete action space policy")
    print(
        "--------------------------------------------------------------------------------------------"
    )
    print("PPO update frequency : " + str(update_timestep) + " timesteps")
    print("PPO K epochs : ", K_epochs)
    print("PPO epsilon clip : ", eps_clip)
    print("discount factor (gamma) : ", gamma)
    print(
        "--------------------------------------------------------------------------------------------"
    )
    print("optimizer learning rate actor : ", lr_actor)
    print("optimizer learning rate critic : ", lr_critic)
    if random_seed:
        print("setting random seed to ", random_seed)
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        try:
            # Gymnasium 방식
            env.reset(seed=random_seed)
            env.action_space.seed(random_seed)
            env.observation_space.seed(random_seed)
        except Exception:
            pass  # 구버전 호환용
    #####################################################

    print(
        "============================================================================================"
    )
    # endregion

    ################# training procedure ################

    # initialize a PPO agent
    ppo_agent = PPO(
        state_dim,
        action_dim,
        lr_actor,
        lr_critic,
        gamma,
        K_epochs,
        eps_clip,
        has_continuous_action_space,
        action_std,
    )

    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    print(
        "============================================================================================"
    )

    # logging file
    log_f = open(log_f_name, "w+")
    log_f.write("episode,timestep,reward\n")

    # printing and logging variables
    print_running_reward = 0
    print_running_episodes = 0

    log_running_reward = 0
    log_running_episodes = 0

    time_step = 0
    i_episode = 0

    last100 = deque(maxlen=100)

    # training loop
    while time_step <= max_training_timesteps:

        state, _ = env.reset()
        current_ep_reward = 0

        for t in range(1, max_ep_len + 1):

            # select action with policy
            raw = ppo_agent.select_action(state)
            action = np.clip(raw, -1.0, 1.0) if is_box else raw  # 연속만 보호용 클립

            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # saving reward and is_terminals
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(float(terminated))  # '진짜 종료'만 컷
            ppo_agent.buffer.is_timeouts.append(
                float(truncated)
            )  # 시간초과는 부트스트랩

            v_next = _value_of(ppo_agent, state)
            ppo_agent.buffer.timeout_bootstrap.append(
                float(v_next) if truncated else 0.0
            )

            time_step += 1
            current_ep_reward += reward

            # update PPO agent
            if time_step % update_timestep == 0:
                # 현재 상태(state)는 step 이후의 s_{t+1} 이므로, 이걸로 V(s_{t+1})를 계산해 last_value로 사용
                ppo_agent.buffer.last_value = _value_of(ppo_agent, state)
                ppo_agent.update(target_kl=0.02, minibatch_size=256)

            # log in logging file
            if time_step % log_freq == 0:

                # log average reward till last episode
                log_avg_reward = log_running_reward / log_running_episodes
                log_avg_reward = round(log_avg_reward, 4)

                log_f.write("{},{},{}\n".format(i_episode, time_step, log_avg_reward))
                log_f.flush()

                log_running_reward = 0
                log_running_episodes = 0

            # printing average reward
            # region
            if time_step % print_freq == 0:

                # print average reward till last episode
                print_avg_reward = print_running_reward / print_running_episodes
                print_avg_reward = round(print_avg_reward, 2)

                avg100 = float(np.mean(last100)) if len(last100) > 0 else float("nan")
                print(
                    "Episode : {} \t\t Timestep : {} \t\t Average Reward : {} \t\t Avg100 : {:.2f}".format(
                        i_episode, time_step, print_avg_reward, avg100
                    )
                )

                print_running_reward = 0
                print_running_episodes = 0

            # save model weights
            if time_step % save_model_freq == 0:
                print(
                    "--------------------------------------------------------------------------------------------"
                )
                print("saving model at : " + checkpoint_path)
                ppo_agent.save(checkpoint_path)
                print("model saved")
                print(
                    "Elapsed Time  : ",
                    datetime.now().replace(microsecond=0) - start_time,
                )
                print(
                    "--------------------------------------------------------------------------------------------"
                )
            # endregion

            # break; if the episode is over
            if done:

                if terminated:
                    # 진짜 종료면 부트스트랩 X → 0
                    ppo_agent.buffer.last_value = torch.zeros(
                        (), device=_mod_device(ppo_agent)
                    )
                else:
                    # 시간초과면 V(next)로 부트스트랩 O
                    ppo_agent.buffer.last_value = _value_of(ppo_agent, state)

                # (선택) 여기서도 업데이트를 한 번 더 강제하고 버퍼를 비워 안정화할 수 있음
                # if len(ppo_agent.buffer.rewards) > 0:
                #     ppo_agent.update(target_kl=0.02, minibatch_size=256)
                _update_plot_episode(info, i_episode)  # ★ 에피소드마다 갱신

                break

        print_running_reward += current_ep_reward
        last100.append(current_ep_reward)
        print_running_episodes += 1

        log_running_reward += current_ep_reward
        log_running_episodes += 1

        i_episode += 1

    log_f.close()
    env.close()

    # print total training time
    print(
        "============================================================================================"
    )
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print(
        "============================================================================================"
    )


if __name__ == "__main__":

    train()
