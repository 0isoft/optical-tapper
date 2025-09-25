# tap/adc_fpga.py
import numpy as np
from core.signal import Signal

class ADCQuant:
    def __init__(self, fs=None, nbits=10, vref=1.0, jitter_rms_s=None):
        self.fs=fs; self.nbits=nbits; self.vref=vref; self.jit=jitter_rms_s
    def __call__(self, sig: Signal) -> Signal:
        fs = sig.fs if self.fs is None else self.fs
        x  = sig.x.real.copy()
        # (optional) aperture jitter ≈ noise ~ 2π f_rms * jitter * slew
        # cheap version: add white noise proportional to |dx/dt|
        if self.jit:
            dxdt = np.gradient(x)*fs
            sigma = np.abs(dxdt) * self.jit
            x = x + sigma*np.random.randn(len(x))
        # quantize
        vmax = self.vref
        xn = np.clip(x / vmax, -1, +1)
        qlevels = 2**self.nbits
        q = np.round((xn+1)*(qlevels-1)/2) / ((qlevels-1)/2) - 1
        vq = vmax * q
        return Signal(x=vq.astype(np.complex128), fs=fs, unit=sig.unit, meta={**sig.meta})

class DCRestore:
    """Digital baseline wander remover: very slow HPF via IIR average."""
    def __init__(self, eps=1e-6):
        self.eps=eps
    def __call__(self, x):
        m=0.0; y=np.empty_like(x, float)
        a=1.0-self.eps
        for i,xi in enumerate(x):
            m=a*m+(1-a)*xi
            y[i]=xi-m
        return y

class SimpleFusionLMS:
    """w = [w_opt, w_em] adapted to minimize slicer error on training chunk."""
    def __init__(self, mu=1e-3, train_syms=2000):
        self.mu=mu; self.N=train_syms
    def fuse_and_adapt(self, v_opt: np.ndarray, v_em: np.ndarray, sps: int, off: int):
        # symbol-center samples
        idx = off + np.arange(min(len(v_opt),len(v_em))//sps) * sps
        idx = idx[idx < len(v_opt)]
        z1 = v_opt[idx]; z2 = v_em[idx]
        # init weights
        w = np.array([1.0, 0.0])   # start optical-only
        out = np.zeros_like(z1)
        Ntr = min(self.N, len(z1))
        # Use 66b header decision-aided: first, coarse threshold on optical only
        thr = 0.5*(np.percentile(z1,15)+np.percentile(z1,85))
        # LMS
        for n in range(Ntr):
            y = w[0]*z1[n] + w[1]*z2[n]
            d = 1.0 if y>=thr else 0.0   # DA target (robustify later with 66b mask)
            e = d - y
            w += self.mu * e * np.array([z1[n], z2[n]])
            out[n] = y
        return w
