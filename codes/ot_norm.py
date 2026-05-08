import numpy as np
import ot
from PIL import Image

class OTSinkhornNormalizer:
    def __init__(self, reg_e=0.1, subsample_size=300, background_intensity=240, luminosity_threshold=0.8):
        """
        Optimal Transport Stain Normalizer using Sinkhorn Transport.
        
        Args:
            reg_e (float): Entropic regularization parameter.
            subsample_size (int): Number of tissue pixels to subsample for estimating the transport plan.
            background_intensity (int): Background intensity I0 for OD conversion.
            luminosity_threshold (float): Threshold to distinguish tissue from background.
        """
        self.reg_e = reg_e
        self.subsample_size = subsample_size
        self.background_intensity = background_intensity
        self.luminosity_threshold = luminosity_threshold
        self.target_od_subsampled = None
        self.ot_mapping = None

    def rgb_to_od(self, img):
        """Convert RGB image to Optical Density (OD) space."""
        # Convert to float and replace zeros to avoid log(0)
        img = np.array(img).astype(np.float64)
        img[img == 0] = 1.0
        
        # Beer-Lambert law
        od = -np.log(img / self.background_intensity)
        # Handle pixels that were brighter than background (gives negative OD)
        od[od < 0] = 0.0
        return od

    def od_to_rgb(self, od):
        """Convert Optical Density (OD) back to RGB space."""
        img = self.background_intensity * np.exp(-od)
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img

    def get_tissue_mask(self, img):
        """
        Create a boolean mask where True indicates tissue and False indicates background.
        Uses a simple mean luminosity threshold.
        """
        img_np = np.array(img).astype(np.float64) / 255.0
        luminosity = np.mean(img_np, axis=-1)
        # Background is bright white/gray, tissue is darker and colored
        return luminosity < self.luminosity_threshold

    def fit(self, target_img):
        """
        Fit the normalizer to a target reference image.
        
        Args:
            target_img (PIL.Image or np.ndarray): Reference image.
        """
        od = self.rgb_to_od(target_img)
        mask = self.get_tissue_mask(target_img)
        
        # Reshape to list of pixels (N, 3) and extract only tissue pixels
        tissue_pixels = od[mask]
        
        # Subsample target tissue pixels
        if tissue_pixels.shape[0] > self.subsample_size:
            idx = np.random.choice(tissue_pixels.shape[0], self.subsample_size, replace=False)
            self.target_od_subsampled = tissue_pixels[idx, :]
        else:
            self.target_od_subsampled = tissue_pixels

    def transform(self, source_img):
        """
        Normalize the source image to match the target color distribution.
        
        Args:
            source_img (PIL.Image or np.ndarray): Source image to normalize.
            
        Returns:
            np.ndarray: Normalized RGB image.
        """
        if self.target_od_subsampled is None:
            raise ValueError("Normalizer must be fitted to a target image before transforming.")
            
        source_od = self.rgb_to_od(source_img)
        mask = self.get_tissue_mask(source_img)
        original_shape = source_od.shape
        
        # Extract source tissue pixels
        tissue_pixels = source_od[mask]
        
        if tissue_pixels.shape[0] == 0:
            # If no tissue found, return original image to avoid errors
            return np.array(source_img)
        
        # Subsample source tissue pixels for transport plan estimation
        if tissue_pixels.shape[0] > self.subsample_size:
            idx = np.random.choice(tissue_pixels.shape[0], self.subsample_size, replace=False)
            source_od_subsampled = tissue_pixels[idx, :]
        else:
            source_od_subsampled = tissue_pixels
            
        # Fit EMD Transport mapping on subsampled tissue pixels
        ot_mapping = ot.da.EMDTransport()
        ot_mapping.fit(Xs=source_od_subsampled, Xt=self.target_od_subsampled)
        
        # Apply mapping ONLY to tissue pixels
        normalized_tissue_od = ot_mapping.transform(Xs=tissue_pixels)
        
        # Reconstruct full OD image
        normalized_od = np.copy(source_od)
        normalized_od[mask] = normalized_tissue_od
        
        # Convert back to RGB
        normalized_rgb = self.od_to_rgb(normalized_od)
        
        return normalized_rgb

def normalize_batch(source_images, target_img, normalizer_kwargs=None):
    """
    Utility to normalize a batch of images to a single target image.
    """
    if normalizer_kwargs is None:
        normalizer_kwargs = {}
        
    normalizer = OTSinkhornNormalizer(**normalizer_kwargs)
    normalizer.fit(target_img)
    
    normalized_images = []
    for img in source_images:
        norm_img = normalizer.transform(img)
        normalized_images.append(norm_img)
        
    return normalized_images
