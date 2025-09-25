# specs_noise.py
import numpy as np
q = 1.602176634e-19

def enbw_1pole(fc_hz: float) -> float:
    return 1.57*fc_hz

def thermal_for_Q(R_A_per_W: float, OMA_rx_mW: float, ENBW_Hz: float, Q=7.0, Iavg_A=None):
    OMA_W = OMA_rx_mW*1e-3
    eye_A = R_A_per_W * OMA_W
    ish = 0.0 if Iavg_A is None else np.sqrt(max(0.0, 2*q*Iavg_A*ENBW_Hz))
    sigma_needed = eye_A / Q
    ith = np.sqrt(max(0.0, sigma_needed**2 - ish**2))
    i_th_A_per_sqrtHz = ith / np.sqrt(ENBW_Hz)
    return float(i_th_A_per_sqrtHz)
