from core.signal import Signal
import numpy as np

# ---- helpers ----

def dBm_to_mW(dBm): return 10**(dBm/10)

def one_pole_lpf(x, fs, fc):
    if fc <= 0: return x
    a = np.exp(-2*np.pi*fc/fs); b = 1 - a
    y = np.empty_like(x, float); acc = 0.0
    for i, xi in enumerate(x):
        acc = a*acc + b*xi
        y[i] = acc
    return y

def one_pole_hpf(x, fs, fc):
    if fc <= 0: return x
    a = np.exp(-2*np.pi*fc/fs); y = np.empty_like(x, float)
    yprev = 0.0
    xprev = x[0]                # ← seed with first sample to start near steady state
    for i, xi in enumerate(x):
        yi = a*(yprev + xi - xprev)
        y[i] = yi; yprev = yi; xprev = xi
    return y

class CTLE_1z1p:
    def __init__(self, fs, fz, fp, gain=1.0):
        self.fs, self.g = fs, gain
        self.az = np.exp(-2*np.pi*fz/fs)
        self.ap = np.exp(-2*np.pi*fp/fs)
        self.x1 = 0.0; self.y1 = 0.0
    def __call__(self, x):
        y = np.empty_like(x, float)
        for i, xi in enumerate(x):
            v = (1-self.az)*xi + self.az*self.x1
            yi = self.g*v - self.ap*self.y1
            y[i] = yi; self.x1 = xi; self.y1 = yi
        return y

#limiting amplifier to restor dc value
class DCRestore:
    def __init__(self, eps=1e-5):
        self.mu = float(eps)   # ~ 1 / time-constant-in-samples
        self.m = 0.0
    def __call__(self, x):
        y = np.empty_like(x, dtype=float)
        m = self.m
        mu = self.mu
        for i, xi in enumerate(x):
            m += mu * (xi - m)    # slow average of the baseline
            y[i] = xi - m         # remove it
        self.m = m
        return y
    
# ---- RX FE ----
q = 1.602176634e-19

def rx_ok_vs_sensitivity(OMA_rx_dBm, sens_OMA_dBm):
    return OMA_rx_dBm >= sens_OMA_dBm  # True if above sensitivity

def rx_noise_for_Q(R_A_per_W, OMA_rx_mW, ENBW_Hz, Q=7.0, Iavg_A=None, i_th_A_per_sqrtHz=None):
    q = 1.602176634e-19
    OMA_W = OMA_rx_mW*1e-3
    eye_A = R_A_per_W * OMA_W
    # choose thermal term to meet target Q after shot noise
    ish = 0.0 if Iavg_A is None else np.sqrt(max(0.0, 2*q*Iavg_A*ENBW_Hz))
    sigma_needed = eye_A / Q
    ith = 0.0 if i_th_A_per_sqrtHz is None else i_th_A_per_sqrtHz*np.sqrt(ENBW_Hz)
    # if thermal is free, set it to hit sigma_needed in quadrature:
    if i_th_A_per_sqrtHz is None:
        ith = np.sqrt(max(0.0, sigma_needed**2 - ish**2))
        i_th_A_per_sqrtHz = ith/np.sqrt(ENBW_Hz)
    sigma = np.sqrt(ish**2 + ith**2)
    return {"eye_A": eye_A, "sigma_A": sigma, "Q": (eye_A/max(sigma,1e-30)),
            "i_th_A_per_sqrtHz": i_th_A_per_sqrtHz}



class RxFE:
    def __init__(self, fs, R_A_per_W=0.8, rx_bw_hz=0.8e9, ac_hz=1e5,
                 tia_in_noise_A_per_sqrtHz=2e-12, limiter=None,
                 ctle_fz=None, ctle_fp=None, ctle_gain=1.0,
                 rin_db_per_hz=None):
        self.fs=fs; self.R=R_A_per_W; self.rx_bw=rx_bw_hz; self.ac=ac_hz
        self.en=tia_in_noise_A_per_sqrtHz; self.rin=rin_db_per_hz
        self.ctle = (CTLE_1z1p(fs, ctle_fz, ctle_fp, ctle_gain)
                     if (ctle_fz is not None and ctle_fp is not None) else None)
        self.lim = limiter

    def __call__(self, fiber_field: Signal) -> Signal:
        fs = self.fs
        P = np.abs(fiber_field.x.real)**2
        I_sig = self.R * P

        # white-ish noise over Nyquist (quick model)
        Bn = fs/2
        ENBW_rx = 1.57 * self.rx_bw   # ~π/2 * fc for 1-pole LPF
        # shot noise
        Iavg = max(I_sig.mean(), 1e-12)
        i_shot = np.sqrt(2*q*Iavg*ENBW_rx) * np.random.randn(len(I_sig))
        # thermal
        i_th   = self.en * np.sqrt(ENBW_rx) * np.random.randn(len(I_sig))
        # RIN (if used)
        if self.rin is not None:
            rin_lin = 10**(self.rin/10)
            i_rin = self.R * (P * np.sqrt(rin_lin*ENBW_rx) * np.random.randn(len(P)))
        else:
            i_rin = 0.0
        v = I_sig + i_shot + i_th + i_rin

        v = one_pole_hpf(v, fs, self.ac)
        if self.ctle: v = self.ctle(v)
        v = one_pole_lpf(v, fs, self.rx_bw)
        if self.lim:
            lo, hi = self.lim
            v = np.clip(v, lo, hi)

        return Signal(x=v.astype(np.complex128), fs=fs, unit="A→V(a.u.)",
              meta={**fiber_field.meta, "domain": "electrical"})
