# build_chain.py
from utils.specs import LinkSpec
from core.specs_power import levels_from_avg_ER, apply_loss_mW
from core.specs_noise import enbw_1pole, thermal_for_Q
from modems.basic_modulation import OOKModulator
from analogFEs.txfe import TxFE
from analogFEs.rxfe import RxFE
from channels.fiber_phys import FiberOnly

def build_blocks(spec: LinkSpec):
    Rs, sps = spec.Rs, spec.sps
    fs = Rs*sps

    # ------- TX power levels from datasheet -------
    P0_tx_mW, P1_tx_mW, OMA_tx_mW = levels_from_avg_ER(spec.tx.Pavg_dBm, spec.tx.ER_dB)

    # scale to sim “a.u.” so P1≈1.0 (keep ratios)
    scale = 1.0 / P1_tx_mW
    P0_sim = scale*P0_tx_mW
    P1_sim = scale*P1_tx_mW
    OMA_tx_sim = scale*OMA_tx_mW

    modem = OOKModulator(Rs=Rs, sps=sps, P1=P1_sim, P0=P0_sim)

    # ------- Fiber loss (power) -------
    LdB_fiber = spec.fiber.alpha_db_per_km*spec.fiber.L_km + spec.fiber.conn_loss_dB
    fiber = FiberOnly(L_km=spec.fiber.L_km, alpha_db_per_km=spec.fiber.alpha_db_per_km)
    # Note: your FiberOnly already attenuates the field; connector loss is easy to
    # apply as an extra scalar on the field if you want:
    fiber.extra_loss_field = 10**(-spec.fiber.conn_loss_dB/20) if hasattr(fiber, 'extra_loss_field') else None

    # ------- RX sensitivity → thermal noise ----------
    OMA_rx_mW = apply_loss_mW(OMA_tx_mW, LdB_fiber)
    fc_rx = spec.rx.rx_bw_mult * Rs
    ENBW = enbw_1pole(fc_rx)
    # rough average current (optional; pass None)
    Iavg_A = None
    i_th = thermal_for_Q(spec.rx.R_A_per_W, OMA_rx_mW, ENBW, Q=7.0, Iavg_A=Iavg_A)

    # ------- Blocks ----------
    txfe = TxFE(fs=fs, tx_bw_hz=spec.tx.tx_bw_mult*Rs)
    rxfe = RxFE(fs=fs,
                R_A_per_W=spec.rx.R_A_per_W,
                rx_bw_hz=fc_rx,
                ac_hz=spec.rx.ac_hz,
                tia_in_noise_A_per_sqrtHz=i_th,
                ctle_fz=None if spec.rx.ctle_fz_mult is None else spec.rx.ctle_fz_mult*Rs,
                ctle_fp=None if spec.rx.ctle_fp_mult is None else spec.rx.ctle_fp_mult*Rs,
                ctle_gain=spec.rx.ctle_gain)

    return modem, txfe, fiber, rxfe
