import cv2
import sys
import random
import numpy as np

import torch

root_path = './'
sys.path.append(root_path)

from hack_utils import NUM_PTS


class AffineAugmenter(object):
    def __init__(self, min_scale=0.9, max_scale=1.1, max_offset=0.1, rotate=True, with_inv=False, elem_name='image'):
        self._min_scale = min_scale
        self._max_scale = max_scale
        self._max_offset = max_offset
        self._rotate = rotate
        self._with_inv = with_inv
        self.elem_name = elem_name
        
    def __call__(self, sample):
        params = self._get_params(sample)
        rotation = self.rotation_matrix(**params)
        sample = self._apply_rotation(sample=sample, rotation=rotation)
        
        if self._with_inv:
            sample['affine_params'] = params
        
        return sample
        
    def _get_params(self, sample):
        params = {}
        h, w, c = sample[self.elem_name].shape
        
        if self._rotate:
            params['angle'] = random.random() * 90 - 45
        else:
            params['angle'] = 0
            
        params['scale'] = self._min_scale + random.random() * (self._max_scale - self._min_scale)
        
        params['x_center'] = w // 2
        params['y_center'] = h // 2
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
            sample['landmarks'] = torch.tensor(landmarks.reshape(-1))
            
        return sample

    @staticmethod
    def rotation_matrix(x_center=0, y_center=0, angle=0, scale=1, x_offset=0, y_offset=0):
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
        sample['affine_params'] = {k: float(v) for k, v in sample['affine_params'].items()}
        rotation = self.rotation_matrix(**sample['affine_params'])
        inv_rotation = self.inv_rotation_matrix(rotation)
        
        return self._apply_rotation(sample=sample, rotation=inv_rotation)
    
    
class CropAugmenter(object):
    def __init__(self, min_scale=0.8, with_inv=False, elem_name='image'):
        self._min_scale = min_scale
        self.elem_name = elem_name
        self._with_inv = with_inv
        
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
        
        if self._with_inv:
            sample['crop_params'] = params
        
        return sample
    
    def inv(self, sample):
        params = sample['crop_params']#{k: float(v) for k, v in sample['crop_params'].items()}
        
        image = np.array(sample[self.elem_name])
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
        