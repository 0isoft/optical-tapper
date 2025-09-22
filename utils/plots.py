import numpy as np
import matplotlib.pyplot as plt
from core.signal import Signal

def constellation(sig: Signal, sps: int, group_delay: int, title="Constellation"):
    centers = np.arange(group_delay, group_delay + sps*len(sig.meta.get("syms", [])), sps)
    centers = centers[centers < len(sig.x)]
    z = sig.x[centers]
    plt.figure()
    plt.scatter(z.real, z.imag, s=8, alpha=0.6)
    plt.gca().set_aspect('equal', 'box')
    plt.title(title); plt.xlabel("I"); plt.ylabel("Q"); plt.grid(True)
    plt.savefig("constellation.png")

def power_db(sig: Signal, title="Optical power |E|^2 (dB)"):
    P = np.abs(sig.x)**2
    P_dB = 10*np.log10(P + 1e-15)
    plt.figure()
    plt.plot(P_dB, linewidth=0.9)
    plt.title(title); plt.xlabel("Sample index"); plt.ylabel("Power (dB, rel.)"); plt.grid(True)
    plt.savefig("power_db.png")

def power_linear(sig: Signal, title="Optical power |E|^2 (dB)"):
    P = np.abs(sig.x)**2
    plt.figure()
    plt.plot(P, linewidth=0.9)
    plt.title(title); plt.xlabel("Sample index"); plt.ylabel("Power (linear)"); plt.grid(True)
    plt.savefig("power_linear.png")


def iq_time(sig: Signal, n: int = 2000, title="I/Q vs time"):
    i = sig.x.real[:n]; q = sig.x.imag[:n]
    t = np.arange(len(i))/sig.fs
    plt.figure(); plt.plot(t, i, label="I"); plt.plot(t, q, label="Q")
    plt.title(title); plt.xlabel("Time [s]"); plt.ylabel("Field (a.u.)"); plt.grid(True); plt.legend(); 
    plt.savefig("iq_time.png")

def spectrum(sig: Signal, title="Magnitude spectrum (dB)"):
    X = np.fft.fftshift(np.fft.fft(sig.x))
    f = np.fft.fftshift(np.fft.fftfreq(len(sig.x), d=1/sig.fs))
    mag = 20*np.log10(np.abs(X)/np.max(np.abs(X)) + 1e-15)
    plt.figure(); plt.plot(f, mag)
    plt.title(title); plt.xlabel("Frequency [Hz]"); plt.ylabel("Mag [dB rel]"); plt.grid(True); 
    plt.savefig("spectrum.png")

def constellation_at_centers(tx_like: Signal, sig: Signal, title="Constellation @ centers"):
    sps = tx_like.meta["sps"]
    gd = (len(tx_like.meta["rrc"]) - 1)//2  # group delay samples
    centers = np.arange(gd, gd + sps*len(tx_like.meta["syms"]), sps)
    centers = centers[centers < len(sig.x)]
    z = sig.x[centers]
    plt.figure(); plt.scatter(z.real, z.imag, s=8, alpha=0.6)
    plt.gca().set_aspect('equal','box'); plt.title(title); plt.xlabel("I"); plt.ylabel("Q"); plt.grid(True); 
    plt.savefig("constellation_at_centers.png")