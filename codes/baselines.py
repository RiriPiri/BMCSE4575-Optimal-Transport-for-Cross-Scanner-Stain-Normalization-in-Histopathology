import numpy as np
import cv2
import torch
import torch.nn as nn
from sklearn.decomposition import DictionaryLearning
import warnings
from sklearn.exceptions import ConvergenceWarning

try:
    import torchstain
except ImportError:
    torchstain = None

class ReinhardNormalizer:
    def __init__(self):
        self.target_means = None
        self.target_stds = None
        
    def fit(self, target_img):
        """Fit Reinhard normalizer using LAB color space means and stds."""
        target_img = np.array(target_img)
        lab = cv2.cvtColor(target_img, cv2.COLOR_RGB2LAB).astype(np.float32)
        
        self.target_means = lab.mean(axis=(0, 1))
        self.target_stds = lab.std(axis=(0, 1))
        
    def transform(self, source_img):
        """Normalize the source image using Reinhard's method."""
        if self.target_means is None:
            raise ValueError("Normalizer must be fitted first.")
            
        source_img = np.array(source_img)
        lab = cv2.cvtColor(source_img, cv2.COLOR_RGB2LAB).astype(np.float32)
        
        means = lab.mean(axis=(0, 1))
        stds = lab.std(axis=(0, 1))
        
        # Avoid division by zero
        stds[stds == 0] = 1.0
        
        # Scale and shift
        for i in range(3):
            lab[:, :, i] = ((lab[:, :, i] - means[i]) * (self.target_stds[i] / stds[i])) + self.target_means[i]
            
        # Clip to valid LAB ranges before conversion back to RGB
        lab[:, :, 0] = np.clip(lab[:, :, 0], 0, 255)
        lab[:, :, 1] = np.clip(lab[:, :, 1], 0, 255)
        lab[:, :, 2] = np.clip(lab[:, :, 2], 0, 255)
        
        lab = lab.astype(np.uint8)
        norm_img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return norm_img


class MacenkoNormalizerWrapper:
    def __init__(self):
        if torchstain is None:
            raise ImportError("Please install torchstain to use MacenkoNormalizer: pip install torchstain")
        self.normalizer = torchstain.normalizers.MacenkoNormalizer(backend='numpy')
        
    def fit(self, target_img):
        target_img = np.array(target_img)
        self.normalizer.fit(target_img)
        
    def transform(self, source_img):
        source_img = np.array(source_img)
        norm_img, _, _ = self.normalizer.normalize(I=source_img, stains=True)
        return norm_img

class VahadaneNormalizer:
    def __init__(self):
        self.target_dict = None
        self.target_max_c = None
        
    def get_stain_matrix(self, img, alpha=0.1):
        """Extract stain matrix using Dictionary Learning (alternative to SPAMS)"""
        # Convert to OD
        img = np.array(img).astype(np.float64)
        img[img == 0] = 1.0
        od = -np.log(img / 255.0)
        
        # Filter background (OD < 0.15)
        od_pixels = od.reshape(-1, 3)
        od_pixels = od_pixels[np.all(od_pixels > 0.15, axis=1)]
        
        if len(od_pixels) == 0:
            return np.eye(3), np.zeros(3)
            
        # Dictionary Learning to find W (stain matrix, 2 stains x 3 channels)
        # Transpose to fit sklearn (n_samples, n_features) -> we want 2 components from 3 features
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            dl = DictionaryLearning(n_components=2, alpha=alpha, fit_algorithm='cd', max_iter=3, random_state=42)
            dl.fit(od_pixels)
        
        # Ensure positive stain matrix
        W = dl.components_  # (2, 3)
        W = np.maximum(W, 0)
        
        # Normalize rows to sum to 1
        W = W / np.linalg.norm(W, axis=1)[:, None]
        
        # Get concentrations for max estimation
        C = dl.transform(od_pixels) # (n_samples, 2)
        max_c = np.percentile(C, 99, axis=0)
        
        return W, max_c
        
    def fit(self, target_img):
        self.target_dict, self.target_max_c = self.get_stain_matrix(target_img)
        
    def transform(self, source_img):
        if self.target_dict is None:
            raise ValueError("Fit the normalizer first.")
            
        img = np.array(source_img).astype(np.float64)
        img[img == 0] = 1.0
        od = -np.log(img / 255.0)
        original_shape = od.shape
        
        od_pixels = od.reshape(-1, 3)
        
        # Get source stain matrix
        source_dict, source_max_c = self.get_stain_matrix(source_img)
        
        # Calculate source concentrations using pseudo-inverse
        # OD = C * W -> C = OD * W_inv
        # where W is (2, 3), we need W.T pseudo inverse
        try:
            C_source = np.linalg.lstsq(source_dict.T, od_pixels.T, rcond=None)[0].T # (N, 2)
        except:
            return source_img # Fallback if singular
            
        C_source = np.maximum(C_source, 0)
        
        # Scale concentrations
        scale = self.target_max_c / (source_max_c + 1e-6)
        C_source_scaled = C_source * scale
        
        # Reconstruct OD
        od_norm = np.dot(C_source_scaled, self.target_dict)
        
        # Convert back to RGB
        od_norm = od_norm.reshape(original_shape)
        norm_img = 255.0 * np.exp(-od_norm)
        norm_img = np.clip(norm_img, 0, 255).astype(np.uint8)
        
        return norm_img

class StainNetArchitecture(nn.Module):
    """
    StainNet architecture (1x1 Convolutions / Pixel-wise MLP)
    Reference: Kang et al. 2021
    """
    def __init__(self):
        super().__init__()
        self.rgb_trans = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=1)
        )
        
    def forward(self, x):
        return self.rgb_trans(x)

class StainNetNormalizer:
    def __init__(self, model_weights_path='stainnet_weights/camelyon16_dataset/StainNet-Public-centerUni_layer3_ch32.pth'):
        self.model = StainNetArchitecture()
        if model_weights_path:
            self.model.load_state_dict(torch.load(model_weights_path, map_location='cpu'))
        self.model.eval()
        
    def fit(self, target_img):
        # StainNet is pre-trained, no fit required per image unless adapting
        pass
        
    def transform(self, source_img):
        img = np.array(source_img).astype(np.float32) / 255.0
        # (H, W, C) -> (1, C, H, W)
        tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            out = self.model(tensor)
        
        out_img = out.squeeze(0).permute(1, 2, 0).numpy()
        out_img = np.clip(out_img * 255.0, 0, 255).astype(np.uint8)
        return out_img

def get_baseline_normalizer(method='macenko', model_weights_path=None):
    method = method.lower()
    if method == 'macenko':
        return MacenkoNormalizerWrapper()
    elif method == 'reinhard':
        return ReinhardNormalizer()
    elif method == 'vahadane':
        return VahadaneNormalizer()
    elif method == 'stainnet':
        if model_weights_path is not None:
            return StainNetNormalizer(model_weights_path)
        else:
            return StainNetNormalizer()
    else:
        raise ValueError(f"Unknown baseline method: {method}")
