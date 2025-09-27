# main.py
from pathlib import Path
import numpy as np

from utils.specs import LinkSpec, TxSpec, RxSpec, FiberSpec
from core.build_chain import build_blocks
from core.signal import Signal
from phy.ethernet66 import encode_64b66b, align_66b_and_descramble, bits_to_bytes
from decoders.ook import OOKDecoder
from utils.ook_vis import _symvals, _kmeans_thr, eye_plot, plot_waveforms, nrz_from_bits

# --- disambiguate DC restorers (name collision fix)
from analogFEs.rxfe import DCRestore as RxDC
from tap.adc_fpga   import ADCQuant, DCRestore as TapDC, SimpleFusionLMS

from tap.blocks import OpticalTapAFE, EMTapAFE
from utils.tap_specs import TapOpticalSpec, TapEMSpec, TapADCSpec, TapFusionSpec

from utils.metrics import (
    symbol_centers, eye_stats_da, empirical_ber_aligned, plot_ber_points, plot_snr_bars, align_symbols_and_bits,
    payload_symbol_indices, eye_stats_da_from_indices
)

from utils.sanity_checks import (
    seed_everything, check_fs_sps, check_bandwidths, check_power_scaling,
    check_coupling, adc_headroom, symbol_centers_trace, eye_from_symbols,
    q_and_ber, masked_payload, check_ac_hpf, warn_low_valid66
)


def dbm_to_w(dBm: float) -> float:
    return 1e-3 * (10 ** (dBm / 10.0))


def main():
    seed_everything(42) #this is for whenever we will have random draws
    spec = LinkSpec(
        Rs=50e6,  #1GHz symbol rate, each symbol period is 1ns, 
        #and each symbol is on/off with a constant optical level, but
        # rise/fall times will be modelled in tx/rx frontends
        sps=8,  # resolution for sampling (granulairty of sim)
        tx=TxSpec(
            Pavg_dBm=-4.0, #average launch optical power of transmitter
            #specified by vendor (-4dBm=0.398mW). 
            # for huawei module, transmit power varies from -8.2 to 0.5dBm
            ER_dB=3.5, #this is the ratio of 1 to 0 optical levels (in powers)
            # ER=3.5dB is vendor spec for huawei module
            tx_bw_mult=0.9,  # fscaling factor such that transmitter filter has a 
            #cutoff around 0.9GHz. datasheet doesnt give such specs on bandwidth
            #this is inferred from rise fall time specs
        ),
        rx=RxSpec(
            R_A_per_W=0.8, #didoe responsivity (current per optical watt received)
            # value picked as "typical" for InGaAs 1310nm diode.
            # if 100uW hits pd we get 80uA avg photocurrent. inferred from datasheet not
            #explicitly listed
            sens_OMA_dBm=-12.6, #optical modulatuon amplitude is inferred from Rx sensityvity
            # which is a vendor spec (-14.4dBm for huawei module). OMA  is  difference between
            #1s and 0s optical powers (modulation swing)
            rx_bw_mult=3.0,   #reating the receiver as a lumped system acting as a filter
            # it has its limitations, but SFP+  TIAs have multi-GHz bandwidth. this shouldnt
            # be a limiting concern for our  system. 3GHz is way above what we care about
            ac_hz=3e4, # within the receiver we model a high pass filter which removes DC baseline wandering
            # this should be tiny, to only block the DC wander (30KHz was deemed enough)

            #CTLE analog filter inside rx to boost high frequencies, attneuate low ones,
            # purpose is to counteract  ISI from chromatic dispersion, pcb  traces etc
            # these specs are more or less guesses, vendors dont specify specs
            ctle_fz_mult=0.5,
            ctle_fp_mult=2.5,
            ctle_gain=1.5,
        ),
        fiber=FiberSpec(
            L_km=0.005, alpha_db_per_km=0.35, conn_loss_dB=1.0
        ),
    )

    modem, txfe, fiber, rxfe = build_blocks(spec)
    #instantiates the analog frontends (tx,rx) and the fiber as objects
    # which then will influence the behavior of the signal

    fs  = modem.Rs * modem.sps
    sps = modem.sps

    # ---- tap specs (unchanged) ----
    opt_spec = TapOpticalSpec(
        coup_dB=-30, #<0.1% of optical power is picked off -40dB would be more realistic
        #given the covert tap (invisibility to main receiver)
        R_A_per_W=0.8, #we will pick an InGaAs PIN to match this spec
        tia_bw_Hz=1.2 * modem.Rs, #we will  have a TIA+frontend that can
        #ideally support 20% more than the symbol rate (to avoid too much ISI)
        ac_hz=5e3, #HPF wander removal (long term drift)
        ctle_fz_Hz=None, ctle_fp_Hz=None, ctle_gain=1.0, #not in the passsive tap probe

        #pd_kind="apd",
        #M=10.0,
        #kA=0.3,
        #apd_bw_alpha=0.7,
        #in_therm_A_per_sqrtHz=1.5e-12,
        #rin_dB_per_Hz=None
    )
    em_spec  = TapEMSpec(
        gain_E=0.08, gain_H=0.03, delay_s=0.8e-9, bw_Hz=1.5 * modem.Rs,
        noise_V_per_sqrtHz=5e-4
        #these numbers are more arbitrary (dependance on geometry, trace distances, stackup, shielding..)
    )
    adc_spec = TapADCSpec(fs=None, nbits=10, vref=0.08, jitter_rms_s=None)
    fus_spec = TapFusionSpec(lms_mu=5e-4, train_syms=3000)

    # ---- tap AFEs / helpers ----
    opt_afe = OpticalTapAFE(opt_spec)
    em_afe  = EMTapAFE(
        em_spec,
        f0_Hz=2.5e9,   # will be clamped to 0.25*fs internally
        Q=1.5,
        gain=0.03,
        delay_s=0.8e-9,
        noise_Vrth=1e-4,
        preblend=(0.8, 0.2),
        input_target_rms=0.05,  # ~50 mV rms drive
    )

    # DC blocks (disambiguated): TapDC for tap streams, RxDC for main RX
    # very slow IIR average + subtraction, epsilon sets time constant 
    dcfix    = TapDC(eps=5e-7) 
    restorer = RxDC(eps=1e-6)

    # simple fusion LMS, pick weights to linearly combine paths from tap 
    # so slicer makes fewer errors
    # if EM branch is noise, w_em is near zero. if EM adds useful SNR, w_em grows
    fuser = SimpleFusionLMS(mu=fus_spec.lms_mu, train_syms=fus_spec.train_syms)

    # ---- health checks ----
    check_fs_sps(fs=fs, Rs=modem.Rs, sps=modem.sps)
    check_bandwidths(
        tx_bw_hz=spec.tx.tx_bw_mult * spec.Rs,
        rx_bw_hz=spec.rx.rx_bw_mult * spec.Rs,
        fs=fs, label="front-ends"
    )
    check_coupling(opt_spec.coup_dB)
    check_ac_hpf(spec.rx.ac_hz,  spec.Rs, tag="normal RX")
    check_ac_hpf(opt_spec.ac_hz, spec.Rs, tag="tap optical")

    # ---- TX → FE → fiber ----
    text = (
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nunc finibus sapien id cursus maximus. "
        "Sed eget libero nunc. Nam in ornare justo, ac aliquet mi. Proin sed sem eget felis pulvinar viverra vel a odio. "
        "Proin venenatis vitae nisi sed efficitur. Etiam fermentum efficitur orci. Proin lobortis volutpat auctor. "
        "Aliquam at lacinia erat. Phasellus tempor diam ultricies mollis condimentum. Nulla in dui vitae lorem fringilla "
        "dignissim eget nec eros. Nulla facilisi. Nam ornare ligula quis auctor suscipit. Praesent varius euismod metus, "
        "in convallis urna elementum sed. Cras tincidunt eget mi ac blandit. Etiam id sodales tellus, a feugiat elit. "
        "Donec diam velit, sodales sit amet faucibus quis, iaculis vitae felis. Fusce fringilla turpis et quam sagittis "
        "convallis. Nunc efficitur porttitor ipsum, ac consectetur libero. Phasellus nec elementum diam, et tincidunt elit. "
        "Sed quis nibh hendrerit, posuere nunc vel, tincidunt nulla. Praesent finibus fermentum neque non pretium. "
        "Donec vitae consequat tortor, id ornare diam. Cras sed lacus mi. Nullam pellentesque elit nulla, at pellentesque "
        "sapien finibus vitae. Class aptent taciti sociosqu ad litora torquent per conubia nostra, per inceptos himenaeos. "
        "Donec massa ante, viverra vitae sollicitudin nec, facilisis vel orci. Vivamus quis dictum justo. Cras vitae risus "
        "tempus, pulvinar elit vel, viverra est. Aenean ultricies diam ac faucibus fermentum."
    )
    print(text)

    bits66 = encode_64b66b(text.encode("utf-8"))

    tx0 = modem.modulate_bits(bits66)  # NRZ field envelope (a.u.)
    tx1 = txfe(tx0)                    # TX analog FE: power LPF, sqrt → field

    # driver "activity" for EM probe model
    I_drv = np.sqrt(np.maximum(np.abs(tx1.x.real) ** 2, 0.0))
    X1 = I_drv
    X2 = np.gradient(I_drv) * tx1.fs
    tx_activity = 0.8 * X1 + 0.2 * X2

    mid = fiber(tx1)  # linear fiber loss/phase in your build

    # ---- scale sim field to physical watts at RX ----
    Ptx_avg_W  = dbm_to_w(spec.tx.Pavg_dBm)
    loss_dB    = spec.fiber.alpha_db_per_km * spec.fiber.L_km + spec.fiber.conn_loss_dB
    Prx_avg_W  = Ptx_avg_W * (10 ** (-loss_dB / 10.0))
    P_unit_avg = float(np.mean(np.abs(mid.x.real) ** 2)) + 1e-30  # sim avg power (a.u.)
    kW         = Prx_avg_W / P_unit_avg

    mid_phys = Signal(
        x=(np.sqrt(kW) * mid.x).astype(np.complex128),
        fs=mid.fs, unit="a.u.", meta={**mid.meta, "units_to_W": kW}
    )
    check_power_scaling(Ptx_avg_W, loss_dB, P_unit_avg, kW)

    # ---- main RX FE (now includes RTIA/post-gain in your updated RxFE) ----
    rx = rxfe(mid_phys)  # returns electrical volts after TIA/filters

    # ---- tap branches (pre/post ADC) ----
    opt_v = opt_afe(mid_phys).x.real
    em_v  = em_afe(tx_activity, tx1.fs).x.real

    # use per-path vref for headroom checks (NO more vref mixup)
    vref_opt = 0.02  # 20 mV full-scale (tap-opt ADC)
    vref_em  = 0.30  # 300 mV full-scale (tap-em ADC)

    adc_opt = ADCQuant(fs=adc_spec.fs, nbits=adc_spec.nbits, vref=vref_opt)
    adc_em  = ADCQuant(fs=adc_spec.fs, nbits=adc_spec.nbits, vref=vref_em)

    opt_d = adc_opt(Signal(x=opt_v.astype(np.complex128), fs=mid.fs, unit="V", meta={"domain": "electrical"})).x.real
    em_d  = adc_em (Signal(x=em_v.astype(np.complex128),  fs=mid.fs, unit="V", meta={"domain": "electrical"})).x.real

    # headroom (pre & post) with correct vref per path
    adc_headroom(opt_v, vref_opt, "tap-opt pre-ADC")
    adc_headroom(em_v,  vref_em,  "tap-em  pre-ADC")
    adc_headroom(opt_d, vref_opt, "tap-opt post-ADC")
    adc_headroom(em_d,  vref_em,  "tap-em  post-ADC")

    # ---- PCB/tap level diagnostics ----
    coup_lin  = 10 ** (opt_spec.coup_dB / 10.0)
    P_main    = np.abs(mid_phys.x.real) ** 2               # main fiber power
    P_tap     = P_main * coup_lin
    P_tx_rect = nrz_from_bits(tx0.meta["bits"], sps, hi=modem.P1, lo=modem.P0)
    P_txfe    = np.abs(tx1.x.real) ** 2
    P_fiber   = P_main
    V_opt     = opt_v
    V_em      = em_v

    def pwr_db(x):
        return 10 * np.log10(np.mean(x) / (np.mean(P_tx_rect) + 1e-18) + 1e-18)

    print("\n[PCB/tap levels]")
    print(f"  <P_tx_rect>        = {np.mean(P_tx_rect):.3e} (ref 0 dB)")
    print(f"  <P_txfe>           = {np.mean(P_txfe):.3e}  ({pwr_db(P_txfe):+6.2f} dB rel TX)")
    #print(f"  <P_fiber>          = {np.mean(P_fiber):.3e}  ({pwr_db(P_fiber):+6.2f} dB rel TX)")
    def mW_to_dBm(mW): return 10*np.log10(mW + 1e-30)
    print(f"  P_rx_avg  ≈ {1e3*np.mean(P_fiber):.3f} mW  ({mW_to_dBm(1e3*np.mean(P_fiber)):+.2f} dBm)")
    print(f"  P_tap_avg ≈ {1e6*np.mean(P_tap):.3f} µW  ({mW_to_dBm(1e3*np.mean(P_tap)):+.2f} dBm)")
    print(f"  <P_tap>            = {np.mean(P_tap):.3e}  ({pwr_db(P_tap):+6.2f} dB rel TX)")
    Vrms_opt = np.sqrt(np.mean(V_opt ** 2)); Vpp_opt = np.max(V_opt) - np.min(V_opt)
    Vrms_em  = np.sqrt(np.mean(V_em  ** 2)); Vpp_em  = np.max(V_em)  - np.min(V_em)
    print(f"  V_opt (rms/pp)     = {Vrms_opt:.3e} / {Vpp_opt:.3e}")
    print(f"  V_em  (rms/pp)     = {Vrms_em :.3e} / {Vpp_em :.3e}\n")

    # ---- quick waveform window ----
    Nzoom = int(5e-6 * fs)
    t = np.arange(len(mid.x)) / fs
    plot_waveforms(
        t[:Nzoom],
        [
            P_tx_rect[:Nzoom],
            P_txfe[:Nzoom],
            P_fiber[:Nzoom],
            P_tap[:Nzoom],
            V_opt[:Nzoom],
            V_em[:Nzoom],
        ],
        [
            "TX NRZ (rect P)",
            "Post-TXFE P",
            "Post-fiber P",
            "Tapped optical P",
            "Optical AFE V",
            "EM AFE V",
        ],
        "Tap branch: powers & voltages (first 5 µs)",
    )

    # ---- eyes for tap (pre-ADC voltages) ----
    #eye_plot(V_opt, sps, span_sym=2, ntraces=400, title="Eye @ Optical AFE output (pre-ADC)")
    #eye_plot(V_em,  sps, span_sym=2, ntraces=400, title="Eye @ EM AFE output (pre-ADC)")

    #optical output, before adc
    eye_plot(
    V_opt, sps,
    span_sym=2, ntraces=400,
    title="Eye @ Optical AFE output (pre-ADC)",
    yscale=1.0, ylabel="V"
    )

    # EM AFE output (same: volts, though mostly noise)
    eye_plot(
        V_em, sps,
        span_sym=2, ntraces=400,
        title="Eye @ EM AFE output (pre-ADC)",
        yscale=1.0, ylabel="V"
    )

    # ---- digital DC restore (tap paths) ----
    opt_dr = dcfix(opt_d)
    em_dr  = dcfix(em_d)

    # ---- coarse timing from optical tap only ----
    best = None
    for off in range(sps):
        sy  = _symvals(opt_dr, sps, off, reduce="mean")
        thr = _kmeans_thr(sy)
        rb  = (sy >= thr).astype(np.uint8)
        _, _, mask = align_66b_and_descramble(rb)
        sc = int(mask.sum())
        if (best is None) or (sc > best["score"]):
            best = {"off": off, "thr": float(thr), "score": sc}
    best_off = best["off"]

    # ---- learn fusion weights, fuse full stream ----
    w = fuser.fuse_and_adapt(opt_dr, em_dr, sps=sps, off=best_off)
    fused = w[0] * opt_dr + w[1] * em_dr

    # ---- decode from fused tap stream ----
    tap_sig = Signal(x=fused.astype(np.complex128), fs=mid.fs, unit="V", meta={"domain": "electrical"})
    tap_dec = OOKDecoder(sps=sps, offset=best_off, reduce="mean")
    tap_bits, _, _ = tap_dec.decode(tap_sig)

    # tap: alignment-aware payload extraction
    _, tap_payload, tap_mask = align_66b_and_descramble(tap_bits)
    _, ref_payload, ref_mask = align_66b_and_descramble(bits66)
    #payload aligned arrays
    tx_aln, tap_aln = masked_payload(ref_payload, ref_mask, tap_payload, tap_mask)


    # print tap stats
    print(f"[tap] weights opt/em = {w}")
    print(f"[tap] 66b valid headers from optical tap: {int(tap_mask.sum())}")
    tap_text = bits_to_bytes(tap_payload).decode("utf-8", errors="replace")
    print(f"[tap] Decoded text from optical tap (descrambled): {tap_text}")

    # tap eye/Q estimate (use optical-only restored stream for stability)
    sym_tap = symbol_centers_trace(opt_dr, sps, best_off)
    est_tap = eye_from_symbols(sym_tap)
    Qt, bert = q_and_ber(est_tap["mu0"], est_tap["mu1"], est_tap["s0"], est_tap["s1"])
    print(f"[sanity] TAP eye: mu0={est_tap['mu0']:.4g}, mu1={est_tap['mu1']:.4g}, "
          f"s0={est_tap['s0']:.3g}, s1={est_tap['s1']:.3g}, Q≈{Qt:.2f}, BER_th≈{bert:.2e}")

    ################## standard receiver ##################

    # ---- digital DC restore on main RX (uses RxDC) ----
    v_dc = restorer(rx.x.real)
    rx_dc = Signal(x=v_dc.astype(np.complex128), fs=rx.fs, unit=rx.unit,
                   meta={**rx.meta, "domain": "electrical"})

    # ---- quick timing sweep on restored waveform ----
    best_rx = None
    for off in range(sps):
        sy  = _symvals(v_dc, sps, off, reduce="mean")
        thr = _kmeans_thr(sy)
        rb  = (sy >= thr).astype(np.uint8)
        _, _, mask = align_66b_and_descramble(rb)
        sc = int(mask.sum())
        if (best_rx is None) or (sc > best_rx["score"]):
            best_rx = {"off": off, "thr": float(thr), "score": sc}
    best_off_rx = best_rx["off"]

    # ---- decode using restored RX signal ----
    dec = OOKDecoder(sps=sps, offset=best_off_rx, reduce="mean")
    raw_bits, _, _ = dec.decode(rx_dc)

    # RX: alignment-aware payload extraction
    _, rx_payload, rx_mask = align_66b_and_descramble(raw_bits)
    tx_aln_rx, rx_aln = masked_payload(ref_payload, ref_mask, rx_payload, rx_mask)

    # human-readable RX text (for sanity)
    rx_text = bits_to_bytes(rx_payload).decode("utf-8", errors="replace")
    print(f"66b valid headers (best off={best_off_rx}): {int(rx_mask.sum())}")
    print("Decoded (descrambled) text (RX):", rx_text)

    # ---- plots on RX ----
    t  = np.arange(len(rx_dc.x)) / fs
    P_tx_rect = nrz_from_bits(tx0.meta["bits"], sps, hi=modem.P1, lo=modem.P0)
    P_txfe    = np.abs(tx1.x.real) ** 2
    P_fiber   = np.abs(mid.x.real) ** 2
    v_rx      = v_dc

    t_start = int(100e-6 * fs)      # index at 100 µs
    t_end   = int(105e-6 * fs)      # index at 105 µs

    plot_waveforms(
        t[t_start:t_end],
        [
            P_tx_rect[t_start:t_end],
            P_txfe[t_start:t_end],
            P_fiber[t_start:t_end],
            v_rx[t_start:t_end],
        ],
        ["TX NRZ (rect P)", "Post-TXFE P", "Post-fiber P", "RX electrical (DC-restored)"],
        "OOK chain (100–105 µs)"
    )
    #eye_plot(v_rx, sps, span_sym=2, ntraces=300, title="RX eye after analog FE + DC restore")
    eye_plot(
    v_rx, sps,
    span_sym=2, ntraces=300,
    title="RX eye after analog FE + DC restore",
    yscale=1.0, ylabel="V"
    )
    
    
    # ---- theoretical Q/BER for RX ----
    sym_rx_trace = symbol_centers_trace(v_dc, sps, best_off_rx)
    esr = eye_from_symbols(sym_rx_trace)
    Qr, berr = q_and_ber(esr["mu0"], esr["mu1"], esr["s0"], esr["s1"])
    print(f"[sanity] RX eye: mu0={esr['mu0']:.4g}, mu1={esr['mu1']:.4g}, "
          f"s0={esr['s0']:.3g}, s1={esr['s1']:.3g}, Q≈{Qr:.2f}, BER_th≈{berr:.2e}")

    # --- TAP eye sanity stats ---
    sym_tap_trace = symbol_centers_trace(opt_dr, sps, best_off)
    est_tap = eye_from_symbols(sym_tap_trace)
    Qt, bert = q_and_ber(est_tap["mu0"], est_tap["mu1"], est_tap["s0"], est_tap["s1"])
    print(f"[sanity] TAP eye: mu0={est_tap['mu0']:.4g}, mu1={est_tap['mu1']:.4g}, "
      f"s0={est_tap['s0']:.3g}, s1={est_tap['s1']:.3g}, Q≈{Qt:.2f}, BER_th≈{bert:.2e}")

if __name__ == "__main__":
    main()
