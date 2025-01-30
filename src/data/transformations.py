import torch

from tsl.data import ImputationDataset
from tsl.data import HORIZON
from tsl.data.batch_map import BatchMapItem
from tsl.data.preprocessing import  StandardScaler
from torch_geometric.transforms import BaseTransform


class CustomTransform(BaseTransform):
    def __init__(self):
        super().__init__()

    def normalize(self, data, label):
        data[label] = data.transform['x'](data[label])
        return data
    
    def mask_input(self, data):
        data['x'] *= data['mask']
        return data

    def mask_y(self, data):
        data['y'] = torch.where(data['og_mask'], data.target['y'], 0)
        return data
    
    def __call__(self, data):
        data = self.mask_input(data)
        data = self.mask_y(data)
        return data
    
class ImputatedDataset(ImputationDataset):
    def __init__(self, og_mask, **kwargs):
        super().__init__(**kwargs)
        self.add_covariate(
            name='x_interpolated',
            value=torch.zeros(og_mask.shape),
            pattern='t n f',
            add_to_input_map=True,
            synch_mode=HORIZON,
            preprocess=False)
        
        self.add_covariate(
            name='og_mask',
            value=og_mask,
            pattern='t n f',
            add_to_input_map=True,
            synch_mode=HORIZON,
            preprocess=False)

        self.auxiliary_map['x_interpolated'] = BatchMapItem('x_interpolated',
                                                       synch_mode=HORIZON,
                                                       pattern='t n f',
                                                       preprocess=True)
        
        self.auxiliary_map['og_mask'] = BatchMapItem('og_mask',
                                                       synch_mode=HORIZON,
                                                       pattern='t n f',
                                                       preprocess=False)

class CustomScaler(StandardScaler):
    
    def fit(self, x, mask=None, keepdims=True):
        size = int(x.shape[0] * 0.7)
        x = x[:size].unsqueeze(0)
        self.bias = torch.mean(x, dim=1, keepdims=keepdims)[0]
        self.scale = torch.std(x, dim=1, keepdims=keepdims)[0]
        return self
    