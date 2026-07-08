"""Sliding-window inference helper.

FeTS/BraTS crops (~128-192 per axis) fit in a single forward pass. ImageCAS
volumes after 0.5mm resampling generally don't, so val/test evaluation for
CAS needs patch-based sliding-window inference to reconstruct a full-volume
prediction before computing Dice/HD95/clDice — a single model(images) call
on the whole volume is not an option there.
"""
from monai.inferers import SlidingWindowInferer

_inferer_cache = {}


def sliding_window_predict(model, image, roi_size=(96, 96, 96), sw_batch_size=4, overlap=0.25):
    """Runs model over `image` (B, C, H, W, D) via sliding-window patches and
    stitches the result into a full-volume output (same spatial shape as `image`).
    """
    key = (roi_size, sw_batch_size, overlap)
    inferer = _inferer_cache.get(key)
    if inferer is None:
        inferer = SlidingWindowInferer(roi_size=roi_size, sw_batch_size=sw_batch_size, overlap=overlap)
        _inferer_cache[key] = inferer
    return inferer(image, model)
