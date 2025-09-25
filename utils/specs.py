# specs.py
from dataclasses import dataclass
import numpy as np

def dBm_to_mW(dBm: float) -> float:
    return 10**(dBm/10)

@dataclass
class TxSpec:
    Pavg_dBm: float      # average TX optical power (AVG)
    ER_dB: float         # extinction ratio (min)
    tx_bw_mult: float    # TX 1-pole BW as multiple of Rs, e.g. 5.0

@dataclass
class RxSpec:
    R_A_per_W: float     # photodiode responsivity
    sens_OMA_dBm: float  # sensitivity OMA (at target BER)
    rx_bw_mult: float    # RX 1-pole BW as multiple of Rs
    ac_hz: float         # AC-coupling cutoff
    ctle_fz_mult: float | None = None  # e.g. 0.25
    ctle_fp_mult: float | None = None  # e.g. 1.0
    ctle_gain: float = 1.0

@dataclass
class FiberSpec:
    L_km: float
    alpha_db_per_km: float
    conn_loss_dB: float = 0.0  # total extra loss (two LC pairs etc.)

@dataclass
class LinkSpec:
    Rs: float          # symbol rate (baud)
    sps: int
    tx: TxSpec
    rx: RxSpec
    fiber: FiberSpec
