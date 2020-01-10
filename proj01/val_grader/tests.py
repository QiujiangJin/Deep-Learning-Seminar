from .grader import Grader, Case

import numpy as np
import torch

def upsample(img, factor):
    w, h = img.size
    return img.resize((int(w*factor), int(h*factor)))
    
def downsample(img, factor):
    return upsample(img, 1./factor)

def get_data_loader(path_name, batch_size=1):
    
    from pathlib import Path
    from PIL import Image

    path = Path(path_name)
    
    def _loader():
        for img_path in path.glob('*.jpg'):
            img = Image.open(img_path)
            yield img

    return _loader
        
class PerceptualLoss(torch.nn.Module):
    """https://towardsdatascience.com/pytorch-implementation-of-perceptual-losses-for-real-time-style-transfer-8d608e2e9902"""
    def __init__(self, vgg):
        super().__init__()
        self.vgg_features = vgg.features
        self.layers = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3",
            '22': "relu4_3"
        }
        
    def forward(self, x):
        outputs = dict()
        for name, module in self.vgg_features._modules.items():
            x = module(x)
            if name in self.layers:
                outputs[self.layers[name]] = x.detach()
        return outputs


class SuperResolutionGrader(Grader):
    """Super-resolution"""
    def __init__(self, *args, **kwargs):
        factor = 4
        super().__init__(*args, **kwargs)
        
        superresolve = self.module.superresolve
        
        data_loader = get_data_loader('data')
        
        self.rel_l1 = []
        self.l1 = []
        self.ssim = []
        self.perceptual = []
        self.diversity = []
        
        from itertools import combinations
        from skimage.measure import compare_ssim as _compare_ssim
        from torchvision.models import vgg
        from torchvision.transforms import functional as TF
        _vgg16 = vgg.vgg16(pretrained=True)
        _perceptual = PerceptualLoss(_vgg16).eval()
        _tensor = lambda x: TF.normalize(TF.to_tensor(x),mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])[None]
        _numpy = lambda x: np.array(x,dtype=float)/255.
        
        compare_l1 = lambda a, b: np.abs(_numpy(a) - _numpy(b)).mean()
        compare_ssim = lambda a, b: _compare_ssim(_numpy(a), _numpy(b), multichannel=True)
        def compare_perceptual(a, b):
            a_features = _perceptual(_tensor(a))
            b_features = _perceptual(_tensor(b))
            loss = 0.
            for name in a_features:
                loss += float((a_features[name] - b_features[name]).abs().mean())
            
            return loss

        for img in data_loader():
            w, h = img.size
            img_low = downsample(img, factor)

            img_rec = superresolve(img_low)
            
            img_up = upsample(img_low, factor)
            img_recs = [superresolve(img_low, seed=seed) for seed in range(10)]
            img_rec_low = downsample(img_rec, factor)
            
            assert img_rec.size == img_up.size, "Superresolved image has wrong resolution"
            
            rec_l1 = compare_l1(img, img_rec)
            rec_diversity = np.mean([compare_l1(img_rec1, img_rec2) for img_rec1, img_rec2 in combinations(img_recs, 2)])
            low_rel_l1 = compare_l1(img_low, img_rec_low) / _numpy(img_low).mean()
            up_l1 = compare_l1(img, img_up)
            rec_ssim = compare_ssim(img, img_rec)
            up_ssim = compare_ssim(img, img_up)
            rec_perceptual = compare_perceptual(img, img_rec)
            up_perceptual = compare_perceptual(img, img_up)
            
            self.l1.append(rec_l1)
            self.rel_l1.append(low_rel_l1)
            self.ssim.append(rec_ssim)
            self.perceptual.append(rec_perceptual)
            self.diversity.append(rec_diversity)
            
    
        print ("L1: %.3f, Low-res rel L1: %.3f, SSIM: %.3f, Perceptual: %.3f, Diversity: %.3f"\
            %(np.mean(self.l1), np.mean(self.rel_l1), np.mean(self.ssim), np.mean(self.perceptual), np.mean(self.diversity)))
    
    @Case(score=30)
    def test_rel_l1_low(self, low=0.05, high=0.1):
        """Relative L1 distance for Low resolution"""
        return np.clip(high-np.mean(self.rel_l1), 0, high-low) / (high-low)

    @Case(score=10)
    def test_l1(self, low=0.0, high=0.034):
        """L1 distance"""
        return np.clip(high-np.mean(self.l1), 0, high-low) / (high-low)
        
    @Case(score=10)
    def test_ssim(self, low=0.795, high=1.0):
        """SSIM"""
        return np.clip(np.mean(self.ssim)-low, 0, high-low) / (high-low)
        
    @Case(score=10)
    def test_perceptual(self, low=0.0, high=3.134):
        """Perceptual loss"""
        return np.clip(high-np.mean(self.perceptual), 0, high-low) / (high-low)
    
    @Case(score=30)
    def test_diversity(self, low=0.0, high=0.05):
        """Diversity"""
        return np.clip(np.mean(self.diversity)-low, 0, high-low) / (high-low)
