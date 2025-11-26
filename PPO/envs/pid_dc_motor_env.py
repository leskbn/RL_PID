# pid_tuning_dc_motor_env_single.py  (플롯 삭제 + 히스토리 info 반환)
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from PID.pid import PID


# ── DC Motor ────────────────────────────────────────────────────────────────
class _DCMotor:
    def __init__(
        self,
        J=0.01,
        b=0.01,
        Kt=0.1,
        Ke=0.1,
        R=1.0,
        L=0.5,
        Vmax=12.0,
        dt=0.01,
        w0=0.0,
        i0=0.0,
    ):
        self.J, self.b, self.Kt, self.Ke, self.R, self.L = J, b, Kt, Ke, R, L
        self.Vmax = float(Vmax)
        self.dt = float(dt)
        self.reset(w0, i0)

    def reset(self, w0=0.0, i0=0.0):
        self.w = float(w0)
        self.i = float(i0)
        return self.state()

    def state(self):
        return np.array([self.w, self.i], dtype=np.float32)

    def _f(self, x, v):
        w, i = x
        v = float(np.clip(v, -self.Vmax, self.Vmax))
        dw = (self.Kt * i - self.b * w) / self.J
        di = (v - self.R * i - self.Ke * w) / self.L
        return np.array([dw, di], dtype=np.float32)

    def step(self, v):
        x = self.state()
        h = self.dt
        k1 = self._f(x, v)
        k2 = self._f(x + 0.5 * h * k1, v)
        k3 = self._f(x + 0.5 * h * k2, v)
        k4 = self._f(x + h * k3, v)
        x_next = x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        self.w, self.i = float(x_next[0]), float(x_next[1])
        return self.state()


# ── util ───────────────────────────────────────────────────────────────────
def _softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


# ── Env ────────────────────────────────────────────────────────────────────
class PIDTuningDCMotorEnv(gym.Env):
    """
    action a∈[-1,1]^3 → PID 게인, 내부에서 T_eval 동안 스텝응답 시뮬.
    관측(state): [J,b,Kt,Ke,R,L,Vmax,w_ref] (log10 선택), 에피소드=1 스텝.
    보상: -(w_itae*ITAE_n + w_settle*(Ts/T) + w_over*over_n + w_energy*Energy_n + w_du*Δu_n)
    """

    metadata = {}

    def __init__(
        self,
        dt=0.01,
        T_eval=10.0,
        w_ref=50.0,
        Vmax=12.0,
        domain_rand=False,
        motor_params=None,
        band_ratio=0.02,
        hold_time=0.5,
        w_itae=1.0,
        w_settle=0.3,
        w_over=0.8,
        w_energy=0.05,
        return_history=True,
        history_stride=1,
        normalize_reward=True,
        w_du=0.01,  # 제어 변화 벌점 가중치
        use_context_obs=True,
        ctx_log=True,
    ):
        super().__init__()
        self.dt = float(dt)
        self.T_eval = float(T_eval)
        self.H = int(round(self.T_eval / self.dt))
        self.w_ref = float(w_ref)
        self.Vmax = float(Vmax)
        self.domain_rand = bool(domain_rand)
        self.band_ratio = float(band_ratio)
        self.hold_time = float(hold_time)
        self.w_itae, self.w_settle, self.w_over, self.w_energy = (
            float(w_itae),
            float(w_settle),
            float(w_over),
            float(w_energy),
        )
        self._base_params = motor_params
        self.rng = np.random.default_rng()

        self.return_history = bool(return_history)
        self.history_stride = int(history_stride)

        self.use_context_obs = bool(use_context_obs)
        self.ctx_log = bool(ctx_log)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        if self.use_context_obs:
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32
            )
        else:
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
            )
        self.plant = None
        self.normalize_reward = bool(normalize_reward)

        self.w_du = float(w_du)

    # ── env internals ──────────────────────────────────────────────────────
    def _default_params(self):
        return dict(
            J=0.01, b=0.01, Kt=0.1, Ke=0.1, R=1.0, L=0.5, Vmax=self.Vmax, dt=self.dt
        )

    def _build_context_vector(self, params):
        v = np.array(
            [
                params["J"],
                params["b"],
                params["Kt"],
                params["Ke"],
                params["R"],
                params["L"],
                params["Vmax"],
                self.w_ref,
            ],
            dtype=np.float32,
        )
        if self.ctx_log:
            v = np.log10(np.maximum(v, 1e-8))  # 스케일 안정화
        return v

    def _rand_params(self, base):
        def jit(v, s=0.2):
            return v * self.rng.uniform(1.0 - s, 1.0 + s)

        return dict(
            J=jit(base["J"]),
            b=jit(base["b"]),
            Kt=jit(base["Kt"]),
            Ke=jit(base["Ke"]),
            R=jit(base["R"]),
            L=jit(base["L"]),
            Vmax=base["Vmax"],
            dt=base["dt"],
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)
        base = self._base_params or self._default_params()
        params = self._rand_params(base) if self.domain_rand else base
        self.plant = _DCMotor(**params)
        self.plant.reset(0.0, 0.0)
        self._ctx = self._build_context_vector(params)  # ← 저장
        obs0 = self._ctx if self.use_context_obs else np.zeros(1, dtype=np.float32)
        return obs0, {}

    def _decode_gains(self, a, use_loglin=True):
        a = np.asarray(a, dtype=np.float32)
        if use_loglin:
            Kmins = np.array([1e-3, 1e-3, 1e-5], dtype=np.float32)
            Kmaxs = np.array([1e2, 1e2, 1e0], dtype=np.float32)
            t = (a + 1.0) * 0.5  # [-1,1] → [0,1]
            logK = np.log10(Kmins) + t * (np.log10(Kmaxs) - np.log10(Kmins))
            K = np.power(10.0, logK)
            return float(K[0]), float(K[1]), float(K[2])
        else:
            s = np.array([3.0, 3.0, 2.0], dtype=np.float32)
            b = np.array([0.0, -2.0, -1.0], dtype=np.float32)
            x = s * a + b
            K = _softplus(x)
            return float(K[0]), float(K[1]), float(K[2])

    def step(self, action):
        Kp, Ki, Kd = self._decode_gains(action)
        pid = PID(
            kp=Kp,
            ki=Ki,
            kd=Kd,
            dt=self.dt,
            i_clamp=10.0,
            out_clamp=self.Vmax,
            deriv_tau=0.02,
        )
        pid.reset()

        itae = 0.0
        energy = 0.0
        overshoot = 0
        in_band_steps = 0
        settled_time = None
        band = self.band_ratio * self.w_ref
        du_energy = 0.0
        v_prev = None

        # 히스토리 버퍼 (플롯 없음, info로만 반환)
        t_hist, w_hist, u_hist = [], [], []

        t = 0.0
        for _ in range(self.H):
            w, i = self.plant.state()
            e = self.w_ref - w
            v = pid.step(e)
            v = float(np.clip(v, -self.Vmax, self.Vmax))
            self.plant.step(v)

            if v_prev is not None:
                dv = v - v_prev
                du_energy += dv * dv
            v_prev = v

            t += self.dt
            itae += t * abs(e) * self.dt
            energy += (v * v) * self.dt
            if w > self.w_ref + band:
                overshoot = 1
            if abs(e) < band:
                in_band_steps += 1
                if settled_time is None and in_band_steps >= int(
                    self.hold_time / self.dt
                ):
                    settled_time = t
            else:
                in_band_steps = 0

            # 히스토리 기록
            t_hist.append(t)
            w_hist.append(w)
            u_hist.append(v)

            if not np.isfinite(w) or abs(w) > 1e4:
                itae += 1e3
                break

        peak_w = max(w_hist) if len(w_hist) > 0 else 0.0
        over_pct = max(
            (peak_w - self.w_ref) / max(self.w_ref, 1e-6), 0.0
        )  # 예: 0.12 = 12%
        over_n = float(np.clip(over_pct, 0.0, 1.0))  # 보상엔 0~1로 정규화해서 사용

        T = self.T_eval
        Ts = settled_time if settled_time is not None else (2 * T)

        # --- reward normalization ---
        T = self.T_eval
        eps = 1e-8

        # 대략적인 기준값(스텝에서 e≈w_ref라 가정하면 ITAE≈0.5*w_ref*T^2)
        ITAE_n = itae / (0.5 * self.w_ref * (T**2) + eps)
        settle_n = Ts / T  # 0~2 정도
        energy_n = energy / (self.Vmax**2 * T + eps)
        du_energy_n = du_energy / (((2 * self.Vmax) ** 2) * self.H + 1e-8)

        if self.normalize_reward:
            reward = -(
                self.w_itae * ITAE_n
                + self.w_settle * settle_n
                + self.w_over * over_n
                + self.w_energy * energy_n
                + self.w_du * du_energy_n
            )
        else:
            reward = -(
                self.w_itae * itae
                + self.w_settle * (Ts / T)
                + self.w_over * over_pct
                + self.w_energy * energy
            )

        obs_next = self._ctx if self.use_context_obs else np.zeros(1, dtype=np.float32)

        info = {
            "Kp": Kp,
            "Ki": Ki,
            "Kd": Kd,
            "itae": itae,
            "settle": Ts,
            "over": overshoot,
            "over_pct": float(over_pct),
            "energy": energy,
            "du_energy": float(du_energy),
            "ctx": self._ctx.copy(),
        }

        if self.return_history:
            s = max(1, self.history_stride)
            info["t"] = np.asarray(t_hist, dtype=np.float32)[::s]
            info["w"] = np.asarray(w_hist, dtype=np.float32)[::s]
            info["u"] = np.asarray(u_hist, dtype=np.float32)[::s]
            info["w_ref"] = float(self.w_ref)

        return obs_next, float(reward), True, False, info

    def close(self):
        pass
