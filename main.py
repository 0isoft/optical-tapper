from pathlib import Path
import numpy as np
from utils.specs import LinkSpec, TxSpec, RxSpec, FiberSpec
from core.build_chain import build_blocks
from core.signal import Signal
from phy.ethernet66 import encode_64b66b, align_66b_and_descramble, bits_to_bytes
from decoders.ook import OOKDecoder
from utils.ook_vis import _symvals, _kmeans_thr, eye_plot, plot_waveforms, nrz_from_bits
from analogFEs.rxfe import DCRestore
from utils.tap_specs import TapOpticalSpec, TapEMSpec, TapADCSpec, TapFusionSpec
from tap.adc_fpga import ADCQuant, DCRestore, SimpleFusionLMS
from tap.blocks import OpticalTapAFE, EMTapAFE

def main():
    # ---- declare a module + fiber you found online ----
    spec = LinkSpec(
        Rs=50e6, sps=8,
        tx=TxSpec(
            Pavg_dBm=-4.0,   # pick mid-point in {-8.2..+0.5} dBm
            ER_dB=3.5,
            tx_bw_mult=6.0   # single-pole BW ~ 6*Rs (keeps eye open)
        ),
        rx=RxSpec(
            R_A_per_W=0.8,
            sens_OMA_dBm=-12.6,
            rx_bw_mult=6.0,
            ac_hz=1e4,
            ctle_fz_mult=None,   # enable later when you tighten BW
            ctle_fp_mult=None,
            ctle_gain=1.0
        ),
        fiber=FiberSpec(
            L_km=0.005, alpha_db_per_km=0.35, conn_loss_dB=0.5  # example extra loss
        )
    )

    modem, txfe, fiber, rxfe = build_blocks(spec)
    fs = modem.Rs * modem.sps
    sps = modem.sps
    opt_spec = TapOpticalSpec(coup_dB=-40, R_A_per_W=0.7, tia_bw_Hz=1.2*modem.Rs,
                          ac_hz=5e3, ctle_fz_Hz=None, ctle_fp_Hz=None, ctle_gain=1.0)
    em_spec  = TapEMSpec(gain_E=0.08, gain_H=0.03, delay_s=0.8e-9, bw_Hz=1.5*modem.Rs,
                     noise_V_per_sqrtHz=5e-4)
    adc_spec = TapADCSpec(fs=None, nbits=10, vref=0.5, jitter_rms_s=None)
    fus_spec = TapFusionSpec(lms_mu=5e-4, train_syms=3000)


    opt_afe = OpticalTapAFE(opt_spec)
    em_afe  = EMTapAFE(em_spec)
    adc     = ADCQuant(**adc_spec.__dict__)
    dcfix   = DCRestore(eps=5e-7)
    fuser   = SimpleFusionLMS(mu=fus_spec.lms_mu, train_syms=fus_spec.train_syms)
        # ---- TX → FE → fiber → RX ----
    text = "Hello world, the quick brown fox jumps over the lazy dog."
    print(text)
    bits66 = encode_64b66b(text.encode("utf-8"))
    tx0 = modem.modulate_bits(bits66)
    tx1 = txfe(tx0)
    mid = fiber(tx1)
    rx  = rxfe(mid)               # electrical Signal (complex container, real-valued)


    #tap signal
    P_txfe = np.abs(tx1.x.real)**2 
    opt_v  = opt_afe(mid).x.real 
    em_v   = em_afe(P_txfe, mid.fs).x.real 
    opt_d  = adc(Signal(x=opt_v.astype(np.complex128), fs=mid.fs, unit="V", meta={"domain":"electrical"})).x.real   
    em_d   = adc(Signal(x=em_v.astype(np.complex128),  fs=mid.fs, unit="V", meta={"domain":"electrical"})).x.real


    # statistics to check what happens within pcb block
    coup_lin = 10**(opt_spec.coup_dB/10)          # power coupling factor
    P_main   = np.abs(mid.x.real)**2               # power in main fiber
    P_tap    = P_main * coup_lin                   # << tapped optical power (into PD)
    P_tx_rect = nrz_from_bits(tx0.meta["bits"], sps, hi=modem.P1, lo=modem.P0)  # ideal TX NRZ power
    P_txfe    = np.abs(tx1.x.real)**2                                           # post-TXFE power
    P_fiber   = P_main                                                          # after fiber (power loss only)
    V_opt     = opt_v                                                           # optical-tap AFE output (voltage)
    V_em      = em_v  
    
    def pwr_db(x):      # average power in dB rel. to P_tx_rect avg
        return 10*np.log10(np.mean(x)/ (np.mean(P_tx_rect)+1e-18) + 1e-18)

    print("\n[PCB/tap levels]")
    print(f"  <P_tx_rect>        = {np.mean(P_tx_rect):.3e} (ref 0 dB)")
    print(f"  <P_txfe>            = {np.mean(P_txfe):.3e}  ({pwr_db(P_txfe):+6.2f} dB rel TX)")
    print(f"  <P_fiber>           = {np.mean(P_fiber):.3e}  ({pwr_db(P_fiber):+6.2f} dB rel TX)")
    print(f"  <P_tap>             = {np.mean(P_tap):.3e}  ({pwr_db(P_tap):+6.2f} dB rel TX)")
    Vrms_opt = np.sqrt(np.mean(V_opt**2)); Vpp_opt = np.max(V_opt)-np.min(V_opt)
    Vrms_em  = np.sqrt(np.mean(V_em**2));  Vpp_em  = np.max(V_em)-np.min(V_em)
    print(f"  V_opt (rms/pp)      = {Vrms_opt:.3e} / {Vpp_opt:.3e}")
    print(f"  V_em  (rms/pp)      = {Vrms_em:.3e} / {Vpp_em:.3e}\n")

    Nzoom = int(5e-6 * fs)  # 5 µs window for clarity

    fs = mid.fs
    t  = np.arange(len(mid.x)) / fs
    plot_waveforms(
        t[:Nzoom],
        [
            P_tx_rect[:Nzoom],
            P_txfe[:Nzoom],
            P_fiber[:Nzoom],
            P_tap[:Nzoom],     # NEW: tapped optical power
            V_opt[:Nzoom],     # NEW: optical AFE voltage
            V_em[:Nzoom],      # NEW: EM AFE voltage
        ],
        [
            "TX NRZ (rect P)",
            "Post-TXFE P",
            "Post-fiber P",
            "Tapped optical P",
            "Optical AFE V",
            "EM AFE V",
        ],
        "Tap branch: powers & voltages (first 5 µs)"
    )

    # --- eye diagrams at the PCB outputs (pre-ADC) ---
    eye_plot(V_opt, sps, span_sym=2, ntraces=400, title="Eye @ Optical AFE output (pre-ADC)")
    eye_plot(V_em,  sps, span_sym=2, ntraces=400, title="Eye @ EM AFE output (pre-ADC)")

    # digital DC restore, optical tap
    opt_dr = dcfix(opt_d)
    em_dr  = dcfix(em_d)
    # coarse timing from optical only (robust & simple)
    
    best=None
    for off in range(sps):
        sy = _symvals(opt_dr, sps, off, reduce="center")
        thr = _kmeans_thr(sy); rb = (sy >= thr).astype(np.uint8)
        _, _, mask = align_66b_and_descramble(rb)
        sc = int(mask.sum())
        if (best is None) or (sc > best["score"]):
            best = {"off": off, "thr": float(thr), "score": sc}
    best_off = best["off"]

    # learn fusion weights on early portion (decision-aided LMS)
    w = fuser.fuse_and_adapt(opt_dr, em_dr, sps=sps, off=best_off)
    # fuse (full stream)
    fused = w[0]*opt_dr + w[1]*em_dr

    # decode from fused
    tap_sig = Signal(x=fused.astype(np.complex128), fs=mid.fs, unit="V", meta={"domain":"electrical"})
    tap_dec = OOKDecoder(sps=sps, offset=best_off, reduce="center")
    tap_bits, _, _ = tap_dec.decode(tap_sig)

    off66, payload_bits, mask = align_66b_and_descramble(tap_bits)
    text_out = bits_to_bytes(payload_bits).decode("utf-8", errors="replace")
    print(f"[tap] weights opt/em = {w}")
    print(f"[tap] 66b valid headers from optical tap: {int(mask.sum())}")
    print(f"[tap] Decoded text from optical tap: {text_out}")




    # --- digital DC restore  -- on rx (not optical tap)
    restorer = DCRestore(eps=1e-6)            # ~1e6-sample time constant; tune as needed
    v_dc = restorer(rx.x.real)                # restored waveform (numpy array)

    # Put it back into a Signal so downstream code knows it's electrical
    rx_dc = Signal(x=v_dc.astype(np.complex128), fs=rx.fs, unit=rx.unit,
                meta={**rx.meta, "domain": "electrical"})

    # -------- quick timing sweep on the *restored* waveform
    sps = modem.sps
    best = None
    for off in range(sps):
        sy  = _symvals(v_dc, sps, off, reduce="center")
        thr = _kmeans_thr(sy)
        rb  = (sy >= thr).astype(np.uint8)
        _, _, mask = align_66b_and_descramble(rb)
        score = int(mask.sum())
        if (best is None) or (score > best["score"]):
            best = {"off": off, "thr": float(thr), "score": score}
    best_off = best["off"]

    # -------- decode using the *restored* Signal
    dec = OOKDecoder(sps=modem.sps, offset=best_off, reduce="center")
    raw_bits, _, _ = dec.decode(rx_dc)
    off66, payload_bits, mask = align_66b_and_descramble(raw_bits)
    text_out = bits_to_bytes(payload_bits).decode("utf-8", errors="replace")
    print(f"66b valid headers (best off={best_off}):", int(mask.sum()))
    print("Decoded (descrambled) text:", text_out)

    # -------- plots (also use restored waveform)
    fs = modem.Rs * modem.sps
    t  = np.arange(len(rx_dc.x)) / fs
    P_tx_rect = nrz_from_bits(tx0.meta["bits"], sps, hi=modem.P1, lo=modem.P0)
    P_txfe    = np.abs(tx1.x.real)**2
    P_fiber   = np.abs(mid.x.real)**2
    v_rx      = v_dc   # restored electrical

    plot_waveforms(t[:int(5e-6*fs)],
                [P_tx_rect[:int(5e-6*fs)], P_txfe[:int(5e-6*fs)],
                    P_fiber[:int(5e-6*fs)], v_rx[:int(5e-6*fs)]],
                ["TX NRZ (rect P)", "Post-TXFE P", "Post-fiber P", "RX electrical (DC-restored)"],
                "OOK chain (first 5 µs)")

    eye_plot(v_rx, sps, span_sym=2, ntraces=300, title="RX eye after analog FE + DC restore")

if __name__ == "__main__":
    main()