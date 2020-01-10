from .grader import Grader, Case
from .utils import ConfusionMatrix

import numpy as np

def get_data_loader(path_name, batch_size=1):
    
    import json
    from pathlib import Path
    from skimage.io import imread
    
    path = Path(path_name)

    with open(str(path / 'color_map.json'), 'r') as f:
        color_map = json.load(f)
    
    with open(str(path / 'data.txt'), 'r') as f:
        files = f.read().split('\n')

    def _loader():
        for file_path in files:
            img = imread(str(path/'images'/file_path))
            raw_mask = imread(str(path/'masks'/file_path))
            
            mask = - np.ones(raw_mask.shape[:2], dtype=int)

            for line in color_map:
                idx = np.all(raw_mask == line['rgb_values'], axis=2)
                mask[idx] = line['id']
            
            yield img, mask

    return _loader


class SegmentationGrader(Grader):
    """Semantic segmentation"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        segment = self.module.segment
        self.c = ConfusionMatrix(size=22)
        
        data_loader = get_data_loader('data')
        
        for img, label in data_loader():
            self.c.add(segment(img),label)
            # self.c.add(label,label)


    @Case(score=50)
    def test_global_accuracy(self, low=0, high=0.8):
        """Global Accuracy"""
        return np.clip(self.c.global_accuracy-low,0,high-low) / (high-low)

    @Case(score=50)
    def test_average_accuracy(self, low=0, high=0.75):
        """Average Accuracy"""
        return np.clip(self.c.average_accuracy-low,0,high-low) / (high-low)
