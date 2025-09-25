from core.signal import Signal
import numpy as np

class FiberOnly:
    def __init__(self, L_km=1.4, alpha_db_per_km=0.35):
        self.loss_db = L_km * alpha_db_per_km
        self.att = 10**(-self.loss_db/20)  # field attenuation

    def __call__(self, tx_field: Signal) -> Signal:
        y = tx_field.x * self.att
        return Signal(x=y, fs=tx_field.fs, unit="a.u.",
                      meta={**tx_field.meta, "loss_db": self.loss_db})
