"""
utils.py — Shared utilities for stain normalization pipeline.
"""

from PIL import Image
from ot_norm import OTSinkhornNormalizer
from baselines import get_baseline_normalizer


class NormalizationTransform:
    """
    PIL-compatible transform that applies a stain normalizer.
    Falls back to the original image if normalization fails (e.g. all-background patch).
    """
    def __init__(self, normalizer):
        self.normalizer = normalizer

    def __call__(self, img):
        try:
            norm_img = self.normalizer.transform(img)
            return Image.fromarray(norm_img)
        except Exception:
            return img  # fallback to original on failure


def build_normalizer(norm_method, target_img_path):
    """
    Instantiate and fit a stain normalizer to a reference image.

    Args:
        norm_method (str): One of 'ot', 'macenko', 'reinhard', 'vahadane', 'stainnet'.
        target_img_path (str): Path to the reference/target image.

    Returns:
        normalizer: Fitted normalizer object with a .transform(img) method.
    """
    print(f"Initializing '{norm_method}' normalizer...")

    if norm_method == 'ot':
        normalizer = OTSinkhornNormalizer()
    else:
        normalizer = get_baseline_normalizer(norm_method)

    target_img = Image.open(target_img_path).convert('RGB')
    normalizer.fit(target_img)
    print(f"Normalizer '{norm_method}' fitted to: {target_img_path}")
    return normalizer
