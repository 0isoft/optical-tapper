# analogFEs/rxfe.py
from core.signal import Signal
import numpy as np

q = 1.602176634e-19

def one_pole_lpf(x, fs, fc):
    if fc is None or fc <= 0: return x
    a = np.exp(-2*np.pi*fc/fs); b = 1 - a
    y = np.empty_like(x, float); acc = 0.0
    xr = np.asarray(x, float)
    for i, xi in enumerate(xr):
        acc = a*acc + b*xi
        y[i] = acc
    return y

def one_pole_hpf(x, fs, fc):
    if fc is None or fc <= 0: return x
    a = np.exp(-2*np.pi*fc/fs)
    y = np.empty_like(x, float)
    xr = np.asarray(x, float)
    yprev = 0.0; xprev = xr[0]
    for i, xi in enumerate(xr):
        yi = a*(yprev + xi - xprev)
        y[i] = yi; yprev = yi; xprev = xi
    return y

class CTLE_1z1p:
    def __init__(self, fs, fz, fp, gain=1.0):
        self.fs = fs
        self.g  = float(gain)
        self.az = np.exp(-2*np.pi*fz/fs)
        self.ap = np.exp(-2*np.pi*fp/fs)
        self.x1 = 0.0; self.y1 = 0.0
    def __call__(self, x):
        y = np.empty_like(x, float)
        xr = np.asarray(x, float)
        az, ap, g = self.az, self.ap, self.g
        x1 = self.x1; y1 = self.y1
        for i, xi in enumerate(xr):
            v  = (1-az)*xi + az*x1
            yi = g*v - ap*y1
            y[i] = yi; x1 = xi; y1 = yi
        self.x1 = x1; self.y1 = y1
        return y

class DCRestore:
    def __init__(self, eps=1e-5):
        self.mu = float(eps)
        self.m  = 0.0
    def __call__(self, x):
        y = np.empty_like(x, float)
        m = self.m; mu = self.mu
        xr = np.asarray(x, float)
        for i, xi in enumerate(xr):
            m += mu * (xi - m)
            y[i] = xi - m
        self.m = m
        return y

class RxFE:
    def __init__(self, fs,
                 R_A_per_W=0.8,
                 rx_bw_hz=0.8e9,
                 ac_hz=1e5,
                 tia_in_noise_A_per_sqrtHz=2e-12,
                 limiter=None,
                 ctle_fz=None, ctle_fp=None, ctle_gain=1.0,
                 rin_db_per_hz=None,
                 R_TIA_ohm=10e3, post_gain=1.0):
        self.fs   = float(fs)
        self.R    = float(R_A_per_W)
        self.rx_bw= float(rx_bw_hz)
        self.ac   = float(ac_hz)
        self.en   = float(tia_in_noise_A_per_sqrtHz)
        self.rin  = rin_db_per_hz
        self.RTIA = float(R_TIA_ohm)
        self.G    = float(post_gain)
        self.lim  = limiter

        # Build CTLE only if meaningfully enabled
        self.ctle = None
        if (ctle_fz is not None) and (ctle_fp is not None) and (ctle_gain is not None) and (ctle_gain != 1.0):
            self.ctle = CTLE_1z1p(self.fs, float(ctle_fz), float(ctle_fp), float(ctle_gain))

    def __call__(self, fiber_field: Signal) -> Signal:
        fs = self.fs
        xopt = fiber_field.x.real
        P = np.abs(xopt)**2                     # optical power [W]
        I_sig = self.R * P                      # PD current [A]

        N = I_sig.size
        Iavg = max(float(np.mean(I_sig)), 1e-15)

        # --- White noise per-sample (to Nyquist), then shape by filters ---
        sigma_sh = np.sqrt(2*q*Iavg * (fs/2.0))    # A_rms per sample
        sigma_th = self.en * np.sqrt(fs/2.0)       # A_rms per sample

        i_shot = sigma_sh * np.random.randn(N)
        i_th   = sigma_th * np.random.randn(N)

        i_rin = 0.0
        if self.rin is not None:
            rin_lin  = 10**(self.rin/10.0)
            # simple RIN model around average power/current
            sigma_rin = self.R * Iavg * np.sqrt(rin_lin * (fs/2.0))
            i_rin     = sigma_rin * np.random.randn(N)

        i_total = I_sig + i_shot + i_th + i_rin

        # --- Filter chain (current domain) ---
        v = i_total
        v = one_pole_hpf(v, fs, self.ac)
        if self.ctle is not None:
            v = self.ctle(v)
        v = one_pole_lpf(v, fs, self.rx_bw)

        # --- TIA + post gain to volts ---
        v = v * self.RTIA * self.G

        # Optional limiter (e.g., limiting amplifier emulation)
        if self.lim is not None:
            lo, hi = self.lim
            v = np.clip(v, lo, hi)

        return Signal(x=v.astype(np.complex128), fs=fs, unit="V",
                      meta={**fiber_field.meta, "domain":"electrical"})
