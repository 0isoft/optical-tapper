from pathlib import Path
import numpy as np
from utils.specs import LinkSpec, TxSpec, RxSpec, FiberSpec
from core.build_chain import build_blocks
from core.signal import Signal
from phy.ethernet66 import encode_64b66b, align_66b_and_descramble, bits_to_bytes
from decoders.ook import OOKDecoder
from utils.ook_vis import _symvals, _kmeans_thr, eye_plot, plot_waveforms, nrz_from_bits
from analogFEs.rxfe import DCRestore

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

        # ---- TX → FE → fiber → RX ----
    text = "Hello world, the quick brown fox jumps over the lazy dog."
    print(text)
    bits66 = encode_64b66b(text.encode("utf-8"))
    tx0 = modem.modulate_bits(bits66)
    tx1 = txfe(tx0)
    mid = fiber(tx1)
    rx  = rxfe(mid)               # electrical Signal (complex container, real-valued)

    # --- digital DC restore (slow baseline removal)
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