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
    """
    Basic IM/DD OOK (On–Off Keying) modulator.

    Parameters
    ----------
    Rs : float
        Symbol rate [baud] (bits per second for OOK).
    sps : int
        Samples per symbol (oversampling factor).
    P1 : float
        'On' optical power level (arbitrary units for now).
    P0 : float
        'Off' optical power level (can be zero or a small bias).
        In practice, lasers have a bias (not truly zero), but we keep it generic.
    """
    Rs: float
    sps: int = 8
    P1: float = 1.0
    P0: float = 0.0  # set small bias if you want strictly positive power (e.g., 0.05)

    def modulate_text(self, text: str, return_power: bool = False) -> Signal:
        """
        Text → bits → NRZ OOK power → optical field.

        - We model IM/DD correctly: the photodiode later measures |E|^2 = P.
        - The fiber/channel in your simulator acts on the optical field E, so we output E = sqrt(P).
        - E is returned as a complex array but purely real and non-negative here.
        """
        bits = text_to_bits(text)                # 0/1 bits
        P = nrz_rect(bits, self.sps, self.P1, self.P0)  # optical power vs time (real, ≥0)

        # Optical field envelope: E = sqrt(P). For a real IM/DD source, phase is 0 here.
        E = np.sqrt(P).astype(np.float64)

        fs = self.Rs * self.sps                  # sample rate
        sig_field = Signal(x=E.astype(np.complex128), fs=fs, unit="a.u.",
                           meta={"bits": bits, "sps": self.sps, "coding": "OOK-NRZ"})

        if return_power:
            # Optional convenience: also return a power Signal alongside the field
            sig_power = Signal(x=P, fs=fs, unit="W(a.u.)",
                               meta={"bits": bits, "sps": self.sps, "coding": "OOK-NRZ"})
            # You can return a tuple if you prefer. Keeping API simple: put power in meta instead.
            sig_field.meta["power_trace"] = P  # lightweight stash
        return sig_field
