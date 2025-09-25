from core.signal import Signal
import numpy as np

def dBm_to_mW(dBm): return 10**(dBm/10)

def tx_levels_from_datasheet(Pavg_dBm, ER_dB, scale_to_P1=1.0):
    ER = 10**(ER_dB/10)
    Pavg = dBm_to_mW(Pavg_dBm)       # mW
    P0 = 2*Pavg/(ER+1)
    P1 = ER*P0
    OMA = P1 - P0
    # normalize if you want P1_sim = scale_to_P1
    s = scale_to_P1 / P1
    return {"P0": P0, "P1": P1, "OMA": OMA, "scale": s,
            "P0_sim": s*P0, "P1_sim": s*P1, "OMA_sim": s*OMA}

def one_pole_lpf(x, fs, fc):
    if fc <= 0: return x
    a = np.exp(-2*np.pi*fc/fs); b = 1 - a
    y = np.empty_like(x, float); acc = 0.0
    for i, xi in enumerate(np.real(x)):
        acc = a*acc + b*xi
        y[i] = acc
    return y

class TxFE:
    def __init__(self, fs, tx_bw_hz=0.8e9, Pscale=1.0):
        self.fs, self.tx_bw, self.Pscale = fs, tx_bw_hz, Pscale
    def __call__(self, tx_field: Signal) -> Signal:
        P = np.abs(tx_field.x.real)**2 * self.Pscale
        P = one_pole_lpf(P, self.fs, self.tx_bw)
        E = np.sqrt(np.maximum(P, 0.0))
        return Signal(x=E.astype(np.complex128), fs=self.fs, unit="a.u.",
                      meta={**tx_field.meta})
