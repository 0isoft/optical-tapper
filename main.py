from pathlib import Path
from datetime import datetime
from modems.qpsk import QPSKModulator, bits_to_qpsk_gray
from modems.basic_modulation import OOKModulator
from channels.fiber_phys import FiberPhysConfig, FiberPhysChannel
from utils.plots import iq_time, spectrum, constellation_at_centers, power_db, power_linear
from utils.io import save_signal, load_signal
from decoders.ook import OOKDecoder
from decoders.qpsk import QPSKIdealDecoder, qpsk_gray_slicer
from phy.ethernet66 import encode_64b66b, align_66b_and_descramble, bits_to_bytes
import numpy as np

def main():
    outdir = Path("data"); outdir.mkdir(exist_ok=True)

    text = "Hello world, the quick brown fox jumps over the lazy dog"


    #modem = OOKModulator(Rs=50e6, sps=8, P1=1.0, P0=0.0)

    modem = QPSKModulator(Rs=50e6, #symbol rate = 50 Msym/s, each QPSK symbol lasts 20 ns
                          sps=8,   # 8 samples per second (simulate cont. waveform)
                          beta=0.25, #rolloff factor for RRC
                          span_sym=8) #filter spans 8 symbols (64 taps)


    bits66=encode_64b66b(text.encode("utf-8"))
    tx = modem.modulate_bits(bits66) #complex baseband optical field envelope 



    chan = FiberPhysChannel(FiberPhysConfig(
        L_km=0.005, #5m length of fiber
        alpha_db_per_km=0.2, #industry standard
        lam_nm=1550.0, #industry standard but depends on laser
        D_ps_nm_km=17.0, #industry standard
        snr_db=None,  
        rin_db_per_hz=None,       # typical DFB ballpark; optional
        #noise budget to be calculated from photodiode shot noise, TIA thermal noise, ADC quantization noise, fiber noise=0 for short fiber

        noise_bw_hz=50e6,           # ~ symbol rate or Rx noise BW
        freq_offset_hz=0.0, #this is sum of chromatic dispersion, freq offset (SI), laser freq offset, PD/TIA mismatches
        linewidth_Hz=0.0 #of laser (double check depending on chosen laser)
        #Kerr nonlinearities assumed negligible
    ))

    rx = chan(tx) #complex baseband envelope of optical field ~ electrical field



    scale = np.vdot(tx.x, rx.x) / np.vdot(tx.x, tx.x)  # complex least-squares gain
    err = rx.x - scale * tx.x
    print("Chan LS gain:", scale)
    print("RMS err:", np.sqrt(np.mean(np.abs(err)**2)))

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    save_signal(outdir/f"tx_field_{stamp}.npz", tx)
    save_signal(outdir/f"rx_field_{stamp}.npz", rx)

    # ---- VISUALIZE side-by-side behaviors ----
    iq_time(tx, title="TX field: I/Q vs time")
    iq_time(rx, title="RX field: I/Q vs time")
    spectrum(tx, title="TX spectrum (dB rel)")
    spectrum(rx, title="RX spectrum (dB rel)")
    constellation_at_centers(tx, tx, title="TX constellation @ centers")
    constellation_at_centers(tx, rx, title="RX constellation @ centers")
    power_db(rx, title="RX optical power |E|^2 (dB)")
    power_linear(rx, title="RX power, linear")


    #decoder = OOKDecoder(sps=modem.sps, offset=0, reduce="mean") 
    #bits, text_out, info = decoder.decode(rx)
    #print("Decoded text:", text_out)
    #print("Threshold info:", info)


    dec = QPSKIdealDecoder(sps=tx.meta["sps"], h_tx=tx.meta["rrc"], do_phase_est=True)
    bits_out, text_out, info = dec.decode(rx)

    print("Decoded (descrambled) text:", text_out)
    print("Info:", info)

    


    ##########################
    print("sanity test")
    msg = b"Hello world, the quick brown fox jumps over the lazy dog"
    tx_bits = encode_64b66b(msg)

    # TX mapping only (no upsample, no filter, no channel):
    syms = bits_to_qpsk_gray(tx_bits)
    # RX slicer only:
    rx_bits = qpsk_gray_slicer(syms)
    

    # 66b align + descramble:
    off66, payload_bits, mask = align_66b_and_descramble(rx_bits)
    print("blocks:", tx_bits.size//66, "valid headers:", int(mask.sum()))
    rec = bits_to_bytes(payload_bits)
    print("decoded:", rec[:len(msg)])
    

if __name__ == "__main__":
    main()
