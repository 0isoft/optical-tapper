# utils/tap_specs.py
from dataclasses import dataclass

@dataclass
class TapOpticalSpec:
    coup_dB: float = -35.0         # optical coupling from fiber to tap
    R_A_per_W: float = 0.7         # PD responsivity @ 1310 nm
    tia_bw_Hz: float = 1.0e9       # TIA -3 dB
    ac_hz: float = 5e3             # AC-coupling HPF (very low)
    ctle_fz_Hz: float | None = None
    ctle_fp_Hz: float | None = None
    ctle_gain: float = 1.0
    in_therm_A_per_sqrtHz: float = 1.5e-12  # TIA input current noise
    rin_dB_per_Hz: float | None = None      # optional RIN

     # --- APD knobs (None => behaves like a PIN) ---
    pd_kind: str = "pin"          # "pin" or "apd"
    M: float = 1.0                # APD gain; use >1 only if pd_kind == "apd"
    kA: float = 0.3               # ionization ratio for McIntyre excess noise
    Idark_A: float = 50e-9        # APD dark current (A) at chosen bias
    apd_bw_alpha: float = 0.7     # BW scales as / M^alpha

@dataclass
class TapEMSpec:
    gain_E: float = 0.1            # scaling for “electric” probe
    gain_H: float = 0.05           # scaling for “magnetic” probe
    delay_s: float = 1.0e-9        # relative delay to optical path
    bw_Hz: float = 1.0e9           # probe front-end bandwidth
    noise_V_per_sqrtHz: float = 1e-3

@dataclass
class TapADCSpec:
    fs: float | None = None        # None => use sim fs
    nbits: int = 10
    vref: float = 1.0              # full-scale peak
    jitter_rms_s: float | None = None  # aperture jitter (optional)

@dataclass
class TapFusionSpec:
    # simple 2-tap scalar weights for [opt, em]
    lms_mu: float = 1e-3
    train_syms: int = 2000         # use early payload (or preamble) to adapt
