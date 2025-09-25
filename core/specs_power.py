# specs_power.py
import numpy as np

def dBm_to_mW(dBm: float) -> float:
    return 10**(dBm/10)

def levels_from_avg_ER(Pavg_dBm: float, ER_dB: float):
    ER = 10**(ER_dB/10)
    Pavg_mW = dBm_to_mW(Pavg_dBm)
    P0 = 2*Pavg_mW/(ER+1)
    P1 = ER*P0
    OMA = P1 - P0
    return P0, P1, OMA

def apply_loss_mW(P_mW: float, L_dB: float) -> float:
    return P_mW * 10**(-L_dB/10)
