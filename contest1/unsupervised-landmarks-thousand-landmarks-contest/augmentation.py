import cv2
import sys
import torch
import random
import numpy as np

from typing import List, Dict, Any

sys.path.append('./')
from hack_utils import NUM_PTS

    
class Augmenter(object):
    name = 'augmenter'
    params_example = dict()
    
    def __init__(self, with_inv=False, elem_name='image'):
        self._with_inv = with_inv
        self.elem_name = elem_name
        
    def _get_params(self, sample):
        return self.params_example.copy()
        
    def _set_params(self, sample, params):
        if self._with_inv:
            if 'params' in sample:
                sample['params'].update(params)
            else:
                sample['params'] = params
            sample['augmenter'] = self.name
    
    def __call__(self, sample):
        params = self._get_params(sample)
        self._set_params(sample, params)
        
        return sample
    
    def inv(self, sample):
        return sample
    

class RandomAugmentation(Augmenter):
    name = 'random'
    params_example = dict()
    
    def __init__(self, augmenters_list: List[Augmenter], probs: List[float], with_inv=False, elem_name='image'):
        super().__init__(with_inv=with_inv, elem_name=elem_name)
        self._augmenters = augmenters_list
        self._probs = probs
        
        for augmenter in self._augmenters:
            self.params_example.update(augmenter.params_example)
        
    def __call__(self, sample):
        params = self._get_params(sample)
        self._set_params(sample, params)
        
        augmenter = np.random.choice(self._augmenters, p=self._probs)
            
        return augmenter(sample)
    
    def inv(self, sample):
        for augmenter in self._augmenters:
            if augmenter.name == sample['augmenter']:
                return augmenter.inv(sample)
        return sample


class AffineAugmenter(Augmenter):
    name = 'affine'
    params_example = dict(angle=0., scale=0., x_center=0, y_center=0, x_offset=0, y_offset=0)
    
    def __init__(self, min_scale=0.9, max_scale=1.1, max_offset=0.1, rotate=True, with_inv=False, elem_name='image'):
        super().__init__(with_inv=with_inv, elem_name=elem_name)
        self._min_scale = min_scale
        self._max_scale = max_scale
        self._max_offset = max_offset
        self._rotate = rotate
        
    def __call__(self, sample):
        params = self._get_params(sample)
        self._set_params(sample, params)
        rotation = self.rotation_matrix(**params)
        sample = self._apply_rotation(sample=sample, rotation=rotation)
        
        return sample
        
    def _get_params(self, sample):
        params = {}
        h, w, c = sample[self.elem_name].shape
        
        if self._rotate:
            params['angle'] = random.random() * 90 - 45.
        else:
            params['angle'] = 0
            
        params['scale'] = self._min_scale + random.random() * (self._max_scale - self._min_scale)
        
        params['x_center'] = int(w // 2)
        params['y_center'] = int(h // 2)
        params['x_offset'] = random.randint(-int(self._max_offset * w), int(self._max_offset * w))
        params['y_offset'] = random.randint(-int(self._max_offset * h), int(self._max_offset * h))
        
        return params
    
    def _apply_rotation(self, sample, rotation):
        image = np.array(sample[self.elem_name])
        h, w, c = image.shape
        
        image = cv2.warpAffine(image, rotation, (w, h), borderValue=(128, 128, 128))
        sample[self.elem_name] = image
        
        if 'landmarks' in sample:
            landmarks = np.array(sample['landmarks']).reshape(-1, 2)
            landmarks_matrix = np.ones((NUM_PTS, 3))
            landmarks_matrix[:, :-1] = landmarks
            landmarks = np.dot(rotation, landmarks_matrix.astype(float).T).T
            sample['landmarks'] = torch.tensor(landmarks.reshape(-1), dtype=torch.float32)
            
        return sample

    @staticmethod
    def rotation_matrix(x_center=0, y_center=0, angle=0, scale=1, x_offset=0, y_offset=0, **kwargs):
        rotation = cv2.getRotationMatrix2D((x_center, y_center), angle, scale)
        rotation[:, 2] += [x_offset, y_offset]
        return rotation
    
    @staticmethod
    def inv_rotation_matrix(rotation):
        rotation_full = np.zeros((3, 3))
        rotation_full[:-1, :] = rotation
        rotation_full[-1, -1] = 1

        inv_rotation = np.linalg.inv(rotation_full.astype(float))[:-1, :]
        return inv_rotation
    
    def inv(self, sample):
        params = {k: float(v) for k, v in sample['params'].items()}
        rotation = self.rotation_matrix(**params)
        inv_rotation = self.inv_rotation_matrix(rotation)
        
        return self._apply_rotation(sample=sample, rotation=inv_rotation)
    
    
class CropAugmenter(Augmenter):
    name = 'crop'
    params_example = dict(scale=0., new_w=0, new_h=0, x_offset=0, y_offset=0)
    
    def __init__(self, min_scale=0.8, with_inv=False, elem_name='image'):
        super().__init__(with_inv=with_inv, elem_name=elem_name)
        self._min_scale = min_scale
        
    def _get_params(self, sample):
        params = {}
        h, w, c = sample[self.elem_name].shape
        
        params['scale'] = self._min_scale + random.random() * (1 - self._min_scale)
        params['new_w'] = int(params['scale'] * w)
        params['new_h'] = int(params['scale'] * h)
        
        params['x_offset'] = random.randint(0, w - params['new_w'])
        params['y_offset'] = random.randint(0, h - params['new_h'])
        
        return params
    
    def __call__(self, sample):
        params = self._get_params(sample)
        self._set_params(sample, params)

        image = np.array(sample[self.elem_name])
        h, w, c = image.shape
        image = cv2.resize(image[
                params['y_offset']:params['y_offset'] + params['new_h'], 
                params['x_offset']:params['x_offset'] + params['new_w']
            ], 
            (w, h),
        )
        sample[self.elem_name] = image
        
        if 'landmarks' in sample:
            landmarks = sample['landmarks'].reshape(-1, 2)
            landmarks[:, 0] = (landmarks[:, 0] - params['x_offset']) * w / params['new_w']  # x
            landmarks[:, 1] = (landmarks[:, 1] - params['y_offset']) * h / params['new_h']  # y
            
            sample['landmarks'] = landmarks.reshape(-1)
        
        return sample
    
    def inv(self, sample):
        params = sample['params']
        
        image = np.array(sample[self.elem_name]).transpose((1, 2, 0))
        h, w, c = image.shape
        
        small_image = cv2.resize(image, (params['new_w'], params['new_h']))
        inv_image = np.full((h, w, c), 128, dtype=np.uint8)
        
        inv_image[
            params['y_offset']:params['y_offset'] + params['new_h'], 
            params['x_offset']:params['x_offset'] + params['new_w'],
            :
        ] = small_image
        sample[self.elem_name] = inv_image
        
        if 'landmarks' in sample:
            landmarks = sample['landmarks'].reshape(-1, 2)
            landmarks[:, 0] = landmarks[:, 0] * params['new_w'] / w + params['x_offset']  # x
            landmarks[:, 1] = landmarks[:, 1] * params['new_h'] / h + params['y_offset']  # y
            
            sample['landmarks'] = landmarks.reshape(-1)
        
        return sample
    

class BrightnessContrastAugmenter(Augmenter):
    name = 'brightness_contrast'
    params_example = dict(brightness=0., contrast=0.)
    
    def __init__(self, brightness=0.3, contrast=0.3, with_inv=False, elem_name='image'):
        super().__init__(with_inv=with_inv, elem_name=elem_name)
        self._brightness = brightness
        self._contrast = contrast
        
    def _get_params(self, sample):
        params = {}
        
        params['brightness'] = 2 * (random.random() - 0.5) * self._brightness
        params['contrast'] = 1 + 2 * (random.random() - 0.5) * self._contrast
        
        return params
    
    def __call__(self, sample):
        params = self._get_params(sample)
        self._set_params(sample, params)

        image = np.array(sample[self.elem_name]).astype(float)
        
        image += params['brightness'] * 255
        image = (image - 128) * params['contrast'] + 128
        image = np.clip(image, 0, 255).astype(np.uint8)
        
        sample[self.elem_name] = image
        
        return sample
    
    def inv(self, sample):
        params = {k: float(v) for k, v in sample['params'].items()}
        
        inv_image = np.array(sample[self.elem_name]).astype(float)
        inv_image = (inv_image - 128) / params['contrast'] + 128
        inv_image = inv_image - params['brightness'] * 255
        inv_image = np.clip(inv_image, 0, 255).astype(np.uint8)
        
        sample[self.elem_name] = inv_image
        
        return sample

    
class BlurAugmenter(Augmenter):
    name = 'blur'
    params_example = dict(kernel=0)
    
    def __init__(self, max_kernel=5, with_inv=False, elem_name='image'):
        super().__init__(with_inv=with_inv, elem_name=elem_name)
        self._max_kernel = max_kernel
        
    def _get_params(self, sample):
        params = {}
        
        params['kernel'] = random.randint(0, self._max_kernel // 2) * 2 + 1
        
        return params
    
    def __call__(self, sample):
        params = self._get_params(sample)
        self._set_params(sample, params)
        
        if params['kernel'] == 1:
            return sample
        
        image = np.array(sample[self.elem_name])
        image = cv2.GaussianBlur(image, (params['kernel'], params['kernel']), params['kernel'] // 2)
        
        sample[self.elem_name] = image
        
        return sample

    
def get_i_from_dict(i: int, d: dict):
    sample = {}
    for key, val in d.items():
        if isinstance(val, dict):
            sample[key] = get_i_from_dict(i, val)
        else:
            sample[key] = val[i]
    return sample


def batch2samples(batch, size):
    for i in range(size):
        yield get_i_from_dict(i, batch)

        
def constant_augmenter(augmenter: Augmenter, const_params_dict: Dict[str, Any]) -> Augmenter:
    def _get_params(sample):
        return const_params_dict.copy()
    augmenter._get_params = _get_params
    
    return augmenter
        