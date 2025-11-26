# pid.py
import numpy as np


class PID:
    def __init__(self, kp, ki, kd, dt, i_clamp=None, out_clamp=None, deriv_tau=0.0):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.dt = dt
        self.i_clamp = i_clamp  # 적분항 제한
        self.out_clamp = out_clamp  # 출력 제한
        self.integral = 0.0
        self.prev_e = None
        self.deriv = 0.0
        self.alpha = (
            dt / (deriv_tau + dt) if deriv_tau > 0 else 1.0
        )  # 미분 저역통과 필터

    def reset(self):
        self.integral = 0.0
        self.prev_e = None
        self.deriv = 0.0

    def step(self, e, de=None):
        # de(=de/dt)를 외부에서 줄 수 있으면 쓰고(예: -theta_dot), 없으면 차분
        if de is None:
            if self.prev_e is None:
                de = 0.0
            else:
                de = (e - self.prev_e) / self.dt
        # 미분 필터링
        self.deriv = self.alpha * de + (1 - self.alpha) * self.deriv

        # 적분
        self.integral += e * self.dt
        if self.i_clamp is not None:
            self.integral = np.clip(self.integral, -self.i_clamp, self.i_clamp)

        # PID 합성
        u = self.kp * e + self.ki * self.integral + self.kd * self.deriv

        # 출력 제한
        if self.out_clamp is not None:
            u = np.clip(u, -self.out_clamp, self.out_clamp)

        self.prev_e = e
        return u
