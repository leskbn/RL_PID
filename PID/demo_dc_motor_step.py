# demo_pid_dc_motor_step.py
import numpy as np
import matplotlib.pyplot as plt
import argparse


# ---------- DC Motor model (armature-controlled) ----------
class DCMotor:
    """
    J dω/dt = Kt*i - b*ω
    L di/dt = v - R*i - Ke*ω
    """

    def __init__(
        self,
        J=0.02,
        b=0.01,
        Kt=1.0,
        Ke=0.1,
        R=2.0,
        L=0.5,
        Vmax=12.0,
        dt=0.01,
        w0=0.0,
        i0=0.0,
    ):
        self.J, self.b, self.Kt, self.Ke, self.R, self.L = J, b, Kt, Ke, R, L
        self.Vmax = float(Vmax)
        self.dt = float(dt)
        self.w = float(w0)
        self.i = float(i0)

    def state(self):
        return self.w, self.i

    def _f(self, w, i, v):
        v = float(np.clip(v, -self.Vmax, self.Vmax))
        dw = (self.Kt * i - self.b * w) / self.J
        di = (v - self.R * i - self.Ke * w) / self.L
        return dw, di

    def step(self, v):
        h = self.dt
        w, i = self.w, self.i
        k1w, k1i = self._f(w, i, v)
        k2w, k2i = self._f(w + 0.5 * h * k1w, i + 0.5 * h * k1i, v)
        k3w, k3i = self._f(w + 0.5 * h * k2w, i + 0.5 * h * k2i, v)
        k4w, k4i = self._f(w + h * k3w, i + h * k3i, v)
        self.w = w + (h / 6.0) * (k1w + 2 * k2w + 2 * k3w + k4w)
        self.i = i + (h / 6.0) * (k1i + 2 * k2i + 2 * k3i + k4i)
        return self.state()


# ---------- PID with derivative filter + anti-windup ----------
class PID:
    def __init__(self, kp, ki, kd, dt, i_clamp=None, out_clamp=None, deriv_tau=0.02):
        self.kp, self.ki, self.kd = float(kp), float(ki), float(kd)
        self.dt = float(dt)
        self.i_clamp = i_clamp
        self.out_clamp = out_clamp
        self._i = 0.0
        self._prev_e = None
        self._d = 0.0
        self._alpha = dt / (deriv_tau + dt) if deriv_tau > 0 else 1.0

    def reset(self):
        self._i = 0.0
        self._prev_e = None
        self._d = 0.0

    def step(self, e, de=None):
        if de is None:
            de = 0.0 if self._prev_e is None else (e - self._prev_e) / self.dt
        # derivative on measurement (LPF)
        self._d = self._alpha * de + (1.0 - self._alpha) * self._d
        # integral + anti-windup
        self._i += e * self.dt
        if self.i_clamp is not None:
            self._i = float(np.clip(self._i, -self.i_clamp, +self.i_clamp))
        # control
        u = self.kp * e + self.ki * self._i + self.kd * self._d
        if self.out_clamp is not None:
            u = float(np.clip(u, -self.out_clamp, +self.out_clamp))
        self._prev_e = float(e)
        return u


# ---------- metrics ----------
def compute_metrics(t, w, w_ref, u, band_ratio=0.02, hold_time=0.5):
    t = np.asarray(t)
    w = np.asarray(w)
    u = np.asarray(u)
    band = band_ratio * w_ref
    # overshoot
    peak = np.max(w)
    overshoot = max(0.0, (peak - w_ref))
    overshoot_pct = 100.0 * max(0.0, (peak / w_ref - 1.0)) if w_ref != 0 else 0.0
    # rise time (10%->90%)
    tr_low, tr_high = 0.1 * w_ref, 0.9 * w_ref
    try:
        t10 = t[np.where(w >= tr_low)[0][0]]
        t90 = t[np.where(w >= tr_high)[0][0]]
        rise_time = max(0.0, t90 - t10)
    except IndexError:
        rise_time = np.nan
    # peak time
    peak_time = t[np.argmax(w)]
    # settling time (enter band and stay for hold_time)
    in_band = np.abs(w - w_ref) < band
    settle_time = np.nan
    if np.any(in_band):
        # find first index after which hold_time of continuous in_band holds
        needed = int(np.ceil(hold_time / (t[1] - t[0])))
        for k in range(len(in_band) - needed):
            if in_band[k] and np.all(in_band[k : k + needed]):
                settle_time = t[k]
                break
    # ITAE & energy
    itae = np.trapz(t * np.abs(w_ref - w), t)
    energy = np.trapz(u * u, t)
    return dict(
        overshoot=overshoot,
        overshoot_pct=overshoot_pct,
        rise_time=rise_time,
        peak_time=peak_time,
        settle_time=settle_time,
        itae=itae,
        energy=energy,
    )


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kp", type=float, default=1)
    ap.add_argument("--ki", type=float, default=0.5)
    ap.add_argument("--kd", type=float, default=0.01)
    ap.add_argument("--wref", type=float, default=1.0, help="rad/s")
    ap.add_argument("--T", type=float, default=12.0, help="sim time [s]")
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--vmax", type=float, default=12.0)
    ap.add_argument("--deriv_tau", type=float, default=0.02)
    ap.add_argument("--i_clamp", type=float, default=10.0)
    args = ap.parse_args()

    # plant & controller
    plant = DCMotor(dt=args.dt, Vmax=args.vmax)
    pid = PID(
        args.kp,
        args.ki,
        args.kd,
        dt=args.dt,
        i_clamp=args.i_clamp,
        out_clamp=args.vmax,
        deriv_tau=args.deriv_tau,
    )
    pid.reset()
    plant.w, plant.i = 0.0, 0.0

    H = int(np.round(args.T / args.dt))
    t = np.arange(1, H + 1) * args.dt
    w_log, u_log = [], []

    for _ in range(H):
        w, i = plant.state()
        e = args.wref - w
        u = pid.step(e)  # voltage command
        u = float(np.clip(u, -args.vmax, args.vmax))
        plant.step(u)
        w_log.append(plant.w)
        u_log.append(u)

    # metrics
    m = compute_metrics(t, np.array(w_log), args.wref, np.array(u_log))
    print(
        f"[Kp={args.kp:.3f}, Ki={args.ki:.3f}, Kd={args.kd:.3f}]  "
        f"Overshoot={m['overshoot_pct']:.1f}%  Rise={m['rise_time']:.3f}s  "
        f"Peak={m['peak_time']:.3f}s  Settle={m['settle_time']:.3f}s  "
        f"ITAE={m['itae']:.3f}  Energy={m['energy']:.3f}"
    )

    # plot
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 5))
    ax1.plot(t, w_log, label="ω")
    ax1.plot(t, [args.wref] * len(t), "k--", lw=1, label="ω_ref")
    ax1.set_ylabel("ω [rad/s]")
    ax1.grid(True)
    ax1.legend(loc="best")
    ax2.plot(t, u_log, label="u=V")
    ax2.set_ylabel("V [V]")
    ax2.set_xlabel("time [s]")
    ax2.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
