import numpy as np, json
from core.signal import Signal

def save_signal(path: str, sig: Signal) -> None:
    # Save complex field as real/imag, plus fs/unit/meta
    meta_json = json.dumps(sig.meta, default=str)
    np.savez_compressed(path,
        x_real=sig.x.real, x_imag=sig.x.imag,
        fs=np.float64(sig.fs), unit=str(sig.unit),
        meta=str(meta_json)
    )

def load_signal(path: str) -> Signal:
    z = np.load(path, allow_pickle=False)
    x = z["x_real"] + 1j*z["x_imag"]
    fs = float(z["fs"])
    unit = z["unit"].astype(str)
    meta = json.loads(z["meta"].astype(str))
    return Signal(x=x, fs=fs, unit=unit, meta=meta)
