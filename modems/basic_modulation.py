import numpy as np
from dataclasses import dataclass
from core.signal import Signal

# --- utilities (duplicate-free: if you already have text_to_bits elsewhere, import it) ---

def text_to_bits(s: str) -> np.ndarray:
    #UTF-8 encode and unpack to bits (MSB first)
    b = np.frombuffer(s.encode("utf-8"), dtype=np.uint8)
    return np.unpackbits(b).astype(np.uint8)

def nrz_rect(bits: np.ndarray, sps: int, high: float, low: float) -> np.ndarray:
    # Rectangular NRZ pulse shaping (repeat each bit 'sps' times)
    # map 1 - high, 0 -'low'.
    # and return a real-valued array at sample rate fs = Rs * sps.

    levels = np.where(bits > 0, high, low).astype(float)

    return np.repeat(levels, sps)

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
class OOKModulator:
    Rs: float
    sps: int = 8
    P1: float = 1.0
    P0: float = 0.0  # consider small bias, e.g., 0.05..0.2 for realistic ER

    def modulate_bits(self, bits: np.ndarray, return_power: bool = False) -> Signal:
        """
        Bits (0/1) → NRZ OOK power → optical field E = sqrt(P).
        Expects bits as a 1-D numpy array of {0,1}.
        """
        b = np.asarray(bits).astype(np.uint8).ravel()
        # (optional) sanity: assert only 0/1
        if np.any((b != 0) & (b != 1)):
            raise ValueError("modulate_bits expects {0,1} values")

        # rectangular NRZ power
        P = nrz_rect(b, self.sps, self.P1, self.P0)  # real, ≥0
        # optical field (envelope) for IM/DD modeling
        E = np.sqrt(P).astype(np.float64)

        fs = self.Rs * self.sps
        sig_field = Signal(x=E.astype(np.complex128), fs=fs, unit="a.u.",
                           meta={"bits": b, "sps": self.sps, "coding": "OOK-NRZ"})

        if return_power:
            sig_field.meta["power_trace"] = P  # optional stash
        return sig_field

    def modulate_text(self, text: str, return_power: bool = False) -> Signal:
        """
        Convenience: UTF-8 text → bits → same as modulate_bits.
        """
        bits = text_to_bits(text)
        return self.modulate_bits(bits, return_power=return_power)