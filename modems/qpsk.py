import numpy as np
from dataclasses import dataclass
from core.signal import Signal

def text_to_bits(s: str) -> np.ndarray:
    b = np.frombuffer(s.encode("utf-8"), dtype=np.uint8)
    return np.unpackbits(b).astype(np.uint8)

def bits_to_qpsk_gray(bits: np.ndarray) -> np.ndarray:
    """Convert bits to QPSK symbols using Gray coding"""
    if bits.size % 2:  # pad
        bits = np.append(bits, 0)
    b0, b1 = bits[0::2], bits[1::2]
    
    # Gray code mapping: (b0,b1) -> (I,Q)
    # 00 -> (+1,+1)  [1st quadrant]
    # 01 -> (-1,+1)  [2nd quadrant] 
    # 11 -> (-1,-1)  [3rd quadrant]
    # 10 -> (+1,-1)  [4th quadrant]
    
    I = np.where((b0==0) & (b1==0), +1,  # 00 -> +1
         np.where((b0==0) & (b1==1), -1,  # 01 -> -1
         np.where((b0==1) & (b1==1), -1,  # 11 -> -1
                                     +1))) # 10 -> +1
    
    Q = np.where((b0==0) & (b1==0), +1,  # 00 -> +1
         np.where((b0==0) & (b1==1), +1,  # 01 -> +1
         np.where((b0==1) & (b1==1), -1,  # 11 -> -1
                                     -1))) # 10 -> -1
    
    return (I + 1j*Q) / np.sqrt(2.0)

def upsample_symbols(symbols: np.ndarray, sps: int) -> np.ndarray:
    x = np.zeros(symbols.size * sps, dtype=complex) #just inseert some zeros
    x[::sps] = symbols #one nonzero every 8 samples
    return x

#RRC is to combat ISI (matched filterat tx and rx), spikes are turned into smooth pulses
def rrc(beta: float, sps: int, span_sym: int) -> np.ndarray:
    N = span_sym * sps
    t = (np.arange(-N/2, N/2 + 1)) / sps
    h = np.zeros_like(t, dtype=float)
    for i, ti in enumerate(t):
        if np.isclose(ti, 0.0):
            h[i] = 1 - beta + 4*beta/np.pi
        elif beta>0 and np.isclose(abs(ti), 1/(4*beta)):
            h[i] = (beta/np.sqrt(2))*(((1+2/np.pi)*np.sin(np.pi/(4*beta))) + ((1-2/np.pi)*np.cos(np.pi/(4*beta))))
        else:
            num = np.sin(np.pi*ti*(1-beta)) + 4*beta*ti*np.cos(np.pi*ti*(1+beta))
            den = np.pi*ti*(1 - (4*beta*ti)**2)
            h[i] = num/den
    h = h/np.sqrt(np.sum(h**2))
    return h

@dataclass
class QPSKModulator:
    Rs: float          # symbol rate [baud]
    sps: int = 8       # samples per symbol
    beta: float = 0.25 # RRC roll-off
    span_sym: int = 8

    def modulate_text(self, text: str) -> Signal:
        bits = text_to_bits(text)
        syms = bits_to_qpsk_gray(bits)
        base = upsample_symbols(syms, self.sps)
        h = rrc(self.beta, self.sps, self.span_sym)
        x = np.convolve(base, h, mode="same")
        fs = self.Rs * self.sps
        return Signal(x=x, fs=fs, unit="a.u.", meta={"syms": syms, "rrc": h, "sps": self.sps})

    def modulate_bits(self, bits: np.ndarray) -> Signal:
        pad_syms = 8
        if bits.size % 2:
            bits = np.append(bits, 0)
        syms = bits_to_qpsk_gray(bits)              # payload symbols only
        syms_p = np.concatenate([
            np.zeros(pad_syms, complex),
            syms,
            np.zeros(pad_syms, complex),
        ])
        n_total = syms_p.size

        base = np.zeros(n_total * self.sps, dtype=complex)  # <-- use padded length
        base[::self.sps] = syms_p

        h = rrc(self.beta, self.sps, self.span_sym)
        gd = (len(h) - 1) // 2
        x_full = np.convolve(base, h, mode="full")
        x = x_full[gd : gd + len(base)]

        fs = self.Rs * self.sps
        meta = {
            "syms": syms,               # payload only
            "rrc": h,
            "sps": self.sps,
            "pad_syms": pad_syms,
            "n_syms_total": int(n_total)
        }
        return Signal(x=x, fs=fs, unit="a.u.", meta=meta)
