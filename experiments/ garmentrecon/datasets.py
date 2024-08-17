import os
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms

from matplotlib import pyplot as plt
from matplotlib.colors import to_rgba

from glob import glob

_CMAP = {
    "帽": {"alias": "hat", "color": "#F7815D"},
    "领": {"alias": "collar", "color": "#F9D26D"},
    "肩": {"alias": "shoulder", "color": "#F23434"},
    "袖片": {"alias": "sleeve", "color": "#C4DBBE"},
    "袖口": {"alias": "cuff", "color": "#F0EDA8"},
    "衣身前中": {"alias": "body front", "color": "#8CA740"},
    "衣身后中": {"alias": "body back", "color": "#4087A7"},
    "衣身侧": {"alias": "body side", "color": "#DF7D7E"},
    
    "底摆": {"alias": "hem", "color": "#DACBBD"},
    "腰头": {"alias": "belt", "color": "#DABDD1"},
    "裙前中": {"alias": "skirt front", "color": "#46B974"},
    "裙后中": {"alias": "skirt back", "color": "#6B68F5"},
    "裙侧": {"alias": "skirt side", "color": "#D37F50"},

    "橡筋": {"alias":"ruffles", "color": "#A8D4D2"},
    "木耳边": {"alias":"ruffles", "color": "#A8D4D2"},
    "袖笼拼条": {"alias":"ruffles", "color": "#A8D4D2"},
    "荷叶边": {"alias":"ruffles", "color": "#A8D4D2"},
    "绑带": {"alias":"ruffles", "color": "#A8D4D2"}
}

_PANEL_CLS = [
    '帽','领','肩','袖片','袖口','衣身前中','衣身后中','衣身侧','底摆','腰头','裙前中','裙后中','裙侧','橡筋','木耳边','袖笼拼条','荷叶边','绑带']


_PANEL_COLORS = np.array(
    [(0., 0., 0., 0.)] + [to_rgba(_CMAP[_PANEL_CLS[idx]]['color']) for idx in range(len(_PANEL_CLS))]
)


def _positional_encoding(input, num_freq, min_freq=2, max_freq=10, include_input=False):
    
    if not torch.is_tensor(input): input = torch.tensor(input)

    scaled = 2 * torch.pi * input  # scale to [0, 2pi]
    freqs = 2 ** torch.linspace(min_freq, max_freq, num_freq, device=input.device)
        
    scaled = scaled[..., None] * freqs  # [..., "input_dim", "num_freqs"]
    scaled = scaled.view(*scaled.shape[:-2], -1)  # [..., "input_dim" * "num_freqs"]
        
    pos_feat = torch.sin(torch.cat([scaled, scaled + torch.pi / 2.0], dim=-1))

    if include_input: pos_feat = torch.cat([pos_feat, input], dim=-1)
    
    return pos_feat

    
class SXDImageDataset(Dataset):
    def __init__(self, data_root, reso=1024, split='train', uv_type='local'):
        
        self.all_files = glob(os.path.join(data_root, split, '*.npz'))
        print("Load %s split from %s, %d items."%(split, data_root, len(self.all_files)))
                    
        self.split = split
        self.reso = reso
        self.num_classes = len(_PANEL_CLS) + 1  # count for background class
        self.num_freq = 4
        
        assert uv_type in ['local', 'global', 'both'], \
            "Unsupported uv_type %s, supported types include [local, global, both]"%uv_type
        self.uv_type = uv_type
                
        if self.split == 'train':
            # data augmentation during training (square crop and random flip for batch training)        
            self.data_transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(
                        self.reso, interpolation=transforms.InterpolationMode.NEAREST_EXACT,
                        scale=(0.25, 1.0), ratio=(0.5, 2.0)
                    ),            
                    transforms.RandomHorizontalFlip()
                ]
            )
        elif self.split == 'val':
            # data augmentation during validation and test,
            self.data_transform = transforms.Compose(
                [
                    transforms.Resize(
                        self.reso, interpolation=transforms.InterpolationMode.NEAREST_EXACT),
                    transforms.RandomCrop(
                        self.reso, padding=0, pad_if_needed=True, 
                        fill=0, padding_mode='constant')
                ]
            )
        elif self.split == 'test':
            self.data_transform = transforms.Resize(
                [self.reso, self.reso], 
                interpolation=transforms.InterpolationMode.NEAREST_EXACT)
        else:
            raise ValueError('Unsupported split type: %s. Please choose between train, val and test.'%(self.split))
                
        print('*** data transform: ', self.data_transform)
                
    def __len__(self):
        return len(self.all_files)


    def __getitem__(self, idx):
        data = np.load(self.all_files[idx])
        
        geo_arr, seg_arr, uv_arr = data['geo'], data['seg'], data['uv']
        
        if self.uv_type == 'global':
            H, W = geo_arr.shape[0], geo_arr.shape[1]
            uv_arr = np.stack(np.meshgrid(
                np.linspace(0., 1.0, W),
                np.linspace(0., 1.0, H)
            ), axis=-1) * ((seg_arr[..., None] > 0).astype(np.float32))
        elif self.uv_type == 'both':
            H, W = geo_arr.shape[0], geo_arr.shape[1]
            global_uv_arr = np.stack(np.meshgrid(
                np.linspace(0., 1.0, W),
                np.linspace(0., 1.0, H)
            ), axis=-1) * ((seg_arr[..., None] > 0).astype(np.float32))
            uv_arr = np.concatenate([uv_arr, global_uv_arr], axis=-1)
        else:
            uv_arr = uv_arr
        
        uv_arr = _positional_encoding(uv_arr, self.num_freq).numpy()
        
        data_tensor = torch.from_numpy(
            np.concatenate([
                geo_arr, seg_arr[..., None], uv_arr], axis=2).transpose(2, 0, 1)).to(torch.float32)
        
        transformed_tensor = self.data_transform(data_tensor)
        
        geo_arr = transformed_tensor[:3]        # (3, H, W)
        seg_arr = transformed_tensor[3].long()  # (H, W)
        uv_arr = transformed_tensor[4:]         # (2, H, W) or (N_encoded, H, W) range [-1, 1]
        
        return geo_arr, seg_arr, uv_arr