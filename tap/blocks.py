# tap/blocks.py
import numpy as np
from core.signal import Signal

q = 1.602176634e-19

def one_pole_lpf(x, fs, fc):
    if fc is None or fc <= 0: return x
    a = np.exp(-2*np.pi*fc/fs); b = 1 - a
    y = np.empty_like(x, float); acc = 0.0
    for i, xi in enumerate(x.real):
        acc = a*acc + b*xi
        y[i] = acc
    return y

def one_pole_hpf(x, fs, fc):
    if fc is None or fc <= 0: return x
    a = np.exp(-2*np.pi*fc/fs)
    y = np.empty_like(x, float)
    yprev = 0.0; xprev = x[0]
    for i, xi in enumerate(x):
        yi = a*(yprev + xi - xprev)
        y[i] = yi; yprev = yi; xprev = xi
    return y

class CTLE_1z1p:
    def __init__(self, fs, fz, fp, gain=1.0):
        self.az = np.exp(-2*np.pi*fz/fs) if fz else 0.0
        self.ap = np.exp(-2*np.pi*fp/fs) if fp else 0.0
        self.g = gain; self.x1 = 0.0; self.y1 = 0.0
    def __call__(self, x):
        if (self.az==0.0 and self.ap==0.0): return x
        y = np.empty_like(x, float)
        for i, xi in enumerate(x):
            v = (1-self.az)*xi + self.az*self.x1
            yi = self.g*v - self.ap*self.y1
            y[i] = yi; self.x1 = xi; self.y1 = yi
        return y

class OpticalTapAFE:
    """Very weak optical coupling → PD → TIA(+CTLE/HPF/LPF) with noises."""
    def __init__(self, spec):
        self.s = spec
    def __call__(self, fiber_field: Signal) -> Signal:
        fs = fiber_field.fs
        # tiny optical power from leakage
        P_main = np.abs(fiber_field.x.real)**2
        coup_lin = 10**(self.s.coup_dB/10)            # power coupling
        P_tap = P_main * coup_lin
        # PD current
        I_sig = self.s.R_A_per_W * P_tap
        # Noise BW ~ single-pole
        ENBW = 1.57 * self.s.tia_bw_Hz
        Iavg = max(I_sig.mean(), 1e-15)
        i_shot = np.sqrt(2*q*Iavg*ENBW) * np.random.randn(len(I_sig))
        i_th   = self.s.in_therm_A_per_sqrtHz*np.sqrt(ENBW)*np.random.randn(len(I_sig))
        if self.s.rin_dB_per_Hz is not None:
            rin_lin = 10**(self.s.rin_dB_per_Hz/10)
            i_rin = self.s.R_A_per_W*(P_tap*np.sqrt(rin_lin*ENBW)*np.random.randn(len(P_tap)))
        else:
            i_rin = 0.0
        v = I_sig + i_shot + i_th + i_rin
        # analog chain
        v = one_pole_hpf(v, fs, self.s.ac_hz)
        if self.s.ctle_fz_Hz and self.s.ctle_fp_Hz:
            v = CTLE_1z1p(fs, self.s.ctle_fz_Hz, self.s.ctle_fp_Hz, self.s.ctle_gain)(v)
        v = one_pole_lpf(v, fs, self.s.tia_bw_Hz)
        return Signal(x=v.astype(np.complex128), fs=fs, unit="Vtap", meta={"domain":"electrical"})

class EMTapAFE:
    """EM probe as delayed, bandlimited mixture of d/dt of TX activity."""
    def __init__(self, spec):
        self.s = spec
    def __call__(self, tx_activity: np.ndarray, fs: float) -> Signal:
        # tx_activity can be Post-TXFE power or driver “voltage” proxy
        # E-probe ~ activity itself, H-probe ~ derivative
        ddt = np.gradient(tx_activity) * fs
        v = self.s.gain_E*tx_activity + self.s.gain_H*ddt
        # delay
        d = int(round(self.s.delay_s * fs))
        if d>0: v = np.pad(v, (d,0))[:len(v)]
        # BW limit + noise
        v = one_pole_lpf(v, fs, self.s.bw_Hz)
        v += self.s.noise_V_per_sqrtHz*np.sqrt(fs/2)*np.random.randn(len(v))
        return Signal(x=v.astype(np.complex128), fs=fs, unit="Vem", meta={"domain":"electrical"})
