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
    def __init__(self, spec, R_TIA_ohm=10e3, post_gain=1.0):
        self.s = spec
        self.RTIA = R_TIA_ohm
        self.G = post_gain

    def __call__(self, fiber_field: Signal) -> Signal:
        fs = fiber_field.fs
        P_main = np.abs(fiber_field.x.real)**2
        coup_lin = 10**(self.s.coup_dB/10)
        P_tap = P_main * coup_lin

        I_sig = self.s.R_A_per_W * P_tap  # A
        v = I_sig * self.RTIA             # V at TIA output

        # HPF / CTLE / LPF
        v = one_pole_hpf(v, fs, self.s.ac_hz)
        if self.s.ctle_fz_Hz and self.s.ctle_fp_Hz:
            v = CTLE_1z1p(fs, self.s.ctle_fz_Hz, self.s.ctle_fp_Hz, self.s.ctle_gain)(v)
        v = one_pole_lpf(v, fs, self.s.tia_bw_Hz)
        v *= self.G

        # Add output-referred noise (after filtering): ENBW ≈ 1.57*BW
        ENBW = 1.57 * self.s.tia_bw_Hz
        # shot + input current noise (A/√Hz) → multiply by RTIA to V/√Hz
        Iavg = max(I_sig.mean(), 1e-15)
        v_shot = (np.sqrt(2*q*Iavg*ENBW) * self.RTIA) * np.random.randn(len(v))
        v_th   = (self.s.in_therm_A_per_sqrtHz*np.sqrt(ENBW) * self.RTIA) * np.random.randn(len(v))
        if self.s.rin_dB_per_Hz is not None:
            rin_lin = 10**(self.s.rin_dB_per_Hz/10)
            v_rin = (self.s.R_A_per_W*np.sqrt(rin_lin*ENBW)*P_tap*self.RTIA) * np.random.randn(len(v))
        else:
            v_rin = 0.0
        v = v + v_shot + v_th + v_rin

        return Signal(x=v.astype(np.complex128), fs=fs, unit="Vtap", meta={"domain":"electrical"})

# tap/em_bandpass.py
import numpy as np
from core.signal import Signal

def biquad_bp_coef(fs, f0, Q, gain=1.0):
    # RBJ-style band-pass (constant skirt gain, peak gain = Q)
    w0 = 2*np.pi*f0/fs
    alpha = np.sin(w0)/(2*Q)
    b0 =   alpha
    b1 =   0.0
    b2 = - alpha
    a0 =   1 + alpha
    a1 =  -2*np.cos(w0)
    a2 =   1 - alpha
    # normalize and add overall gain
    b0, b1, b2 = (gain*b0/a0, gain*b1/a0, gain*b2/a0)
    a1, a2 = (a1/a0, a2/a0)
    return b0, b1, b2, a1, a2

class EMTapAFE:
    """
    EM probe model:
      v = [a*x + b*dx/dt] * BandPass(f0,Q)  (delay + AWGN)
    """
    def __init__(self, spec, f0_Hz=2.5e9, Q=2.0, gain=0.03, delay_s=0.8e-9,
                 noise_Vrth=1e-4, preblend=(0.8, 0.2), input_target_rms=0.05):
        self.f0 = f0_Hz; self.Q = Q; self.gain = gain
        self.delay_s = delay_s
        self.noise = noise_Vrth
        self.a, self.b = preblend
        self.spec = spec
        self.input_target_rms = input_target_rms

    def __call__(self, tx_activity: np.ndarray, fs: float) -> Signal:
        # safe center freq
        f0 = min(self.f0, 0.25*fs)
        f0 = max(f0, 1e-3)

        x = tx_activity.astype(float)
        dxdt = np.gradient(x) * fs
        u = self.a*x + self.b*dxdt

        # normalize prefilter drive
        rms = np.sqrt(np.mean(u*u) + 1e-30)
        u *= (self.input_target_rms / rms)

        # band-pass biquad
        b0,b1,b2,a1,a2 = biquad_bp_coef(fs, f0, self.Q, gain=self.gain)
        y = np.zeros_like(u)
        x1=x2=y1=y2=0.0
        for n, xn in enumerate(u):
            yn = b0*xn + b1*x1 + b2*x2 - a1*y1 - a2*y2
            y[n] = yn; x2=x1; x1=xn; y2=y1; y1=yn

        # delay
        d = int(round(self.delay_s * fs))
        if d > 0:
            y = np.pad(y, (d,0))[:len(y)]

        # optional post LPF to emulate probe/front-end BW
        if self.spec.bw_Hz:
            a = np.exp(-2*np.pi*self.spec.bw_Hz/fs); b = 1-a
            acc = 0.0; z = np.empty_like(y)
            for i, yi in enumerate(y):
                acc = a*acc + b*yi
                z[i] = acc
            y = z

        # noise scaled to ENBW of the passband
        ENBW = max(f0/self.Q, 1.0)
        y += self.noise * np.sqrt(ENBW) * np.random.randn(len(y))
        target_rms = 0.05  # 50 mV RMS
        r = np.sqrt(np.mean(y*y) + 1e-30)
        y *= (target_rms / r)
        return Signal(x=y.astype(np.complex128), fs=fs, unit="Vem", meta={"domain":"electrical"})
