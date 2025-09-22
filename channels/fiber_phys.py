import numpy as np
from dataclasses import dataclass
from core.signal import Signal
from typing import Optional

c = 299_792_458.0

@dataclass
class FiberPhysConfig:
    L_km: float = 0.005           # fiber length=5m
    alpha_db_per_km: float = 0.2  # attenuation
    lam_nm: float = 1550.0        # wavelength
    D_ps_nm_km: float = 17.0      # dispersion
   
    # --- Link-only stochastic knobs (optional) ---
    snr_db: Optional[float] = None       # keep for experiments; normally None for short passive fiber
    rin_db_per_hz: Optional[float] = None  # e.g., -155 dB/Hz (DFB typical). None → no RIN
    noise_bw_hz: float = 0.0             # integrate RIN over this bandwidth (e.g., ~symbol rate)

    freq_offset_hz: float = 0.0          # constant Δf (useful if you emulate a carrier)
    linewidth_Hz: float = 0.0     # laser phase noise (optional)

def _beta2_from_D(D_ps_nm_km: float, lam_nm: float) -> float:
    # D [ps/(nm·km)] -> beta2 [s^2/m]
    D = D_ps_nm_km * 1e-6        # s/(m·m)
    lam = lam_nm * 1e-9          # m
    return -(lam**2/(2*np.pi*c)) * D

class FiberPhysChannel:
    def __init__(self, cfg: FiberPhysConfig):
        self.cfg = cfg
        self.beta2 = _beta2_from_D(cfg.D_ps_nm_km, cfg.lam_nm)

    def _apply_cd(self, x: np.ndarray, fs: float, L_m: float) -> np.ndarray:
        # Skip if dispersion over this length is negligible
        if abs(self.beta2) * L_m < 1e-24:
            return x

        N = len(x)
        
        # Fixed padding calculation
        pad = min(max(64, N//8), 256)  # reasonable padding based on signal length
        
        # Zero-pad symmetrically
        xpad = np.pad(x, pad_width=pad, mode='constant', constant_values=0)
        Np = len(xpad)
        
        # Frequency vector (use fftshift for proper centering)
        f = np.fft.fftshift(np.fft.fftfreq(Np, d=1/fs))
        
        # Dispersion transfer function
        H = np.exp(-0.5j * (2*np.pi*f)**2 * self.beta2 * L_m)
        H = np.fft.ifftshift(H)  # shift back for FFT
        
        # Apply dispersion in frequency domain
        X = np.fft.fft(xpad)
        Y = X * H
        ypad = np.fft.ifft(Y)
        
        # Extract original length, preserving phase alignment
        return ypad[pad:pad+N]

    def _apply_phase_noise(self, x: np.ndarray, fs: float) -> np.ndarray:
        lw = self.cfg.linewidth_Hz
        if lw <= 0:
            return x
        # Wiener phase noise: var(Δφ) per sample = 2πΔν / fs
        var = 2*np.pi*lw / fs
        dphi = np.sqrt(var) * np.random.randn(len(x))
        return x * np.exp(1j*np.cumsum(dphi))

    def __call__(self, sig: Signal) -> Signal:
        cfg = self.cfg
        y = sig.x.astype(complex)
        fs = sig.fs
        L_m = cfg.L_km * 1e3

        # (Optional) emulate a carrier frequency offset
        if cfg.freq_offset_hz != 0.0:
            n = np.arange(len(y))
            y *= np.exp(1j * 2*np.pi * cfg.freq_offset_hz * n / fs)

        # Laser phase noise (tiny over 5 m, but keep for completeness)
        y = self._apply_phase_noise(y, fs)

        # Chromatic dispersion (on field) – negligible at 5 m, but modeled
        y = self._apply_cd(y, fs, L_m)

        # Power attenuation: field scales by sqrt(power loss)
        loss_db = cfg.alpha_db_per_km * cfg.L_km
        att_field = 10**(-loss_db / 20.0)
        y *= att_field

        # --- Link-only noise ---
        # Prefer RIN over AWGN for physical realism; otherwise none.
        if cfg.snr_db is not None:
            # (kept for experiments) add complex AWGN referenced to pre-attenuation Es
            Es = np.mean(np.abs(sig.x)**2)
            snr_lin = 10**(cfg.snr_db/10)
            sigma = np.sqrt(Es/(2*snr_lin)) * att_field
            n = sigma * (np.random.randn(len(y)) + 1j*np.random.randn(len(y)))
            y = y + n
        elif (cfg.rin_db_per_hz is not None) and (cfg.noise_bw_hz > 0):
            # RIN ~ dB/Hz → linear; multiplicative intensity noise approximation.
            rin_lin = 10**(cfg.rin_db_per_hz/10)
            # Relative amplitude std ~ sqrt(RIN * B); inject as small field noise
            sigma_rel = np.sqrt(rin_lin * cfg.noise_bw_hz)
            # Scale noise roughly to local amplitude; normalize to average power to keep units sane
            P_avg = np.mean(np.abs(y)**2) + 1e-18
            rel = (y / np.sqrt(P_avg))
            n = sigma_rel * (np.random.randn(len(y)) + 1j*np.random.randn(len(y))) * rel
            y = y + n

        return Signal(x=y, fs=fs, unit="a.u.", meta={**sig.meta, "loss_db": loss_db})
