# cma_ctx_tune.py
# pip install cma
import numpy as np
import cma
import matplotlib.pyplot as plt
from envs.pid_dc_motor_env import PIDTuningDCMotorEnv

# 컨텍스트 관측 사용, 도메인 랜덤화 ON
ENV_KW = dict(
    dt=0.01,
    T_eval=10.0,
    w_ref=40.0,
    Vmax=12.0,
    domain_rand=True,
    motor_params=None,
    return_history=False,
    history_stride=1,
    use_context_obs=True,
    ctx_log=True,
)

env = PIDTuningDCMotorEnv(**ENV_KW)
ctx_dim = env.observation_space.shape[0]  # 보통 8
act_dim = env.action_space.shape[0]  # 3


# 정책 a = tanh(W c + b) ; θ = [W(3xD), b(3)] 펼쳐서 최적화
def unpack_theta(theta):
    W = theta[: act_dim * ctx_dim].reshape(act_dim, ctx_dim)
    b = theta[act_dim * ctx_dim :].reshape(act_dim)
    return W, b


def policy(theta, ctx):
    W, b = unpack_theta(theta)
    a = np.tanh(W @ ctx + b)
    return np.clip(a, -1.0, 1.0)


def eval_theta(theta, seeds=(101, 102, 103, 104, 105)):
    # 여러 컨텍스트(시드) 평균 리워드로 견고성 평가
    rewards = []
    for sd in seeds:
        ctx, _ = env.reset(seed=sd)  # 컨텍스트 관측
        a = policy(theta, ctx)
        _, r, _, _, _ = env.step(a)
        rewards.append(float(r))
    return np.mean(rewards)


# 초기 θ (작은 난수)
theta_dim = act_dim * ctx_dim + act_dim
x0 = np.zeros(theta_dim, dtype=np.float64)
sigma0 = 0.2
options = {
    "popsize": 24,
    "maxiter": 200,
    "verb_disp": 1,
    # θ에는 명시적 bounds가 없지만 tanh로 a는 [-1,1] 보장됨
}

es = cma.CMAEvolutionStrategy(x0, sigma0, options)
while not es.stop():
    Thetas = es.ask()
    fitness = [-eval_theta(th) for th in Thetas]  # -평균리워드 (minimize)
    es.tell(Thetas, fitness)
    es.disp()

theta_best = es.result.xbest
print("Done. Testing best θ on a fixed plant...")

# 고정 플랜트에서 응답 확인 (재현용)
test_env = PIDTuningDCMotorEnv(
    dt=0.01,
    T_eval=10.0,
    w_ref=40.0,
    Vmax=12.0,
    domain_rand=False,
    motor_params=dict(J=0.01, b=0.01, Kt=0.1, Ke=0.1, R=1.0, L=0.5, Vmax=12.0, dt=0.01),
    return_history=True,
    history_stride=1,
    use_context_obs=True,
    ctx_log=True,
)

ctx, _ = test_env.reset(seed=42)
a = policy(theta_best, ctx)
_, r, _, _, info = test_env.step(a)
print(f"Reward={r:.4f}, K=({info['Kp']:.4g},{info['Ki']:.4g},{info['Kd']:.4g})")

t, w, u, wref = info["t"], info["w"], info["u"], info["w_ref"]
plt.figure(figsize=(9, 5))
plt.subplot(2, 1, 1)
plt.plot(t, w)
plt.axhline(wref, ls=":", c="k")
plt.grid(True)
plt.ylabel("ω [rad/s]")
plt.subplot(2, 1, 2)
plt.plot(t, u)
plt.grid(True)
plt.ylabel("V [V]")
plt.xlabel("time [s]")
plt.tight_layout()
plt.show()
