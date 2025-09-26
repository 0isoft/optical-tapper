# build_chain.py
from utils.specs import LinkSpec
from core.specs_power import levels_from_avg_ER, apply_loss_mW, dBm_to_mW
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
    # vendor specs average power and ER, this is then used to determine P0, P1 and OMA
    # we expect for a -4dBm abg and ER=3.5dB (the  vendor spec) to get P0=0.23mW, P1=0.53mW and OMA=0.29mW

    #normalize optical power units such that '1' level is a dimensionless 1
    # we have to remember to scale back to physical watts when injecting noise
    scale = 1.0 / P1_tx_mW
    P0_sim = scale*P0_tx_mW
    P1_sim = scale*P1_tx_mW

    modem = OOKModulator(Rs=Rs, sps=sps, P1=P1_sim, P0=P0_sim)

    # ------- Fiber loss (power) -------
    LdB_fiber = spec.fiber.alpha_db_per_km*spec.fiber.L_km + spec.fiber.conn_loss_dB
    fiber = FiberOnly(L_km=spec.fiber.L_km, alpha_db_per_km=spec.fiber.alpha_db_per_km)
    # Note: your FiberOnly already attenuates the field; connector loss is easy to

    if hasattr(fiber, "extra_loss_field"):
        fiber.extra_loss_field = 10 ** (-spec.fiber.conn_loss_dB / 20.0)

    # ------- RX sensitivity → thermal noise ----------
    #fiber and connectors attenuate power (loss is specified in dB, it takes effect on mW)
    fc_rx = spec.rx.rx_bw_mult * Rs
    # the -3dB cutoff of receivers primary LPF ties the bandwidth to data rate
    # exmaple Rs=1GHz, f_c=3GHz (wide, we dont care about this too much)
    ENBW = enbw_1pole(fc_rx)
    #equivalent noise bandwidth is here to turn noise spectral density into RMS noise
    # example ENBW = 4.7GHz if f_c is 3GHz (wider bandwidth leads to bigger ENBW = more total noise power)
    Iavg_A = None
    OMA_sens_mW = dBm_to_mW(spec.rx.sens_OMA_dBm) 
    OMA_sens_W = OMA_sens_mW * 1e-3   
    ER_lin = 10 ** (spec.tx.ER_dB / 10.0)
    P1_sens_W = OMA_sens_W * ER_lin / (ER_lin - 1.0)
    P0_sens_W = OMA_sens_W * 1.0    / (ER_lin - 1.0)
    Pavg_sens_W = 0.5 * (P1_sens_W + P0_sens_W)
    R = spec.rx.R_A_per_W
    Iavg_sens_A = R * Pavg_sens_W  

    i_th = thermal_for_Q(R, OMA_sens_mW, ENBW, Q=7.0, Iavg_A=Iavg_sens_A)


    #this solves for receiver's input referred thermal noise such that it produces a desired Q factor
    #(given OMA and bandwidth). it just sets a noise (thermal) s.t. we achieve a desired Q factor
    # q factor here being the eye amplitude divided by the noise RMS. larger Q means better separation between 1/0 levels
    #BER=1/2 erfc(Q/sqrt(2))
    #the specific value of 7 is pulled out of gpt ass 

    # ------- Blocks ----------
    txfe = TxFE(fs=fs, tx_bw_hz=spec.tx.tx_bw_mult * Rs)

    rxfe = RxFE(
        fs=fs,
        R_A_per_W=spec.rx.R_A_per_W,
        rx_bw_hz=fc_rx,
        ac_hz=spec.rx.ac_hz,
        tia_in_noise_A_per_sqrtHz=i_th,                 # <-- fixed by sensitivity
        ctle_fz=None if spec.rx.ctle_fz_mult is None else spec.rx.ctle_fz_mult * Rs,
        ctle_fp=None if spec.rx.ctle_fp_mult is None else spec.rx.ctle_fp_mult * Rs,
        ctle_gain=spec.rx.ctle_gain
    )

    return modem, txfe, fiber, rxfe
