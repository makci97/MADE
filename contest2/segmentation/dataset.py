import torch
import os, json
import numpy as np

from PIL import Image
from matplotlib.path import Path
from torch.utils.data import Dataset


class SegmentationDataset(Dataset):

    def __init__(self, data_path, config_file=None, transforms=None, val_split=0.95, is_train=True):
        super(SegmentationDataset, self).__init__()
        self.data_path = data_path
        self.transforms = transforms

        with open(config_file, 'r') as f:
            self.marks = json.load(f)
            if is_train:
                self.marks = self.marks[:int(len(self.marks) * val_split)]
            else:
                self.marks = self.marks[int(len(self.marks) * val_split):]

    def __len__(self):
        return len(self.marks)

    def __getitem__(self, idx):
        item = self.marks[idx]
        img_path = os.path.join(self.data_path, item["file"])
        img = Image.open(img_path).convert('RGB')
        w, h = img.size

        box_coords = item['nums']
        boxes = []
        labels = []
        masks = []
        for box in box_coords:
            points = np.array(box['box'])
            x0, y0 = np.min(points[:, 0]), np.min(points[:, 1])
            x2, y2 = np.max(points[:, 0]), np.max(points[:, 1])
            boxes.append([x0, y0, x2, y2])
            labels.append(1)

            # Здесь мы наши 4 точки превращаем в маску
            # Это нужно, чтобы кроме bounding box предсказывать и, соответственно, маску :)
            nx, ny = w, h
            poly_verts = points
            x, y = np.meshgrid(np.arange(nx), np.arange(ny))
            x, y = x.flatten(), y.flatten()
            points = np.vstack((x, y)).T
            path = Path(poly_verts)
            grid = path.contains_points(points)
            grid = grid.reshape((ny, nx)).astype(int)
            masks.append(grid)

        boxes = torch.as_tensor(boxes)
        labels = torch.as_tensor(labels)
        masks = torch.as_tensor(masks)

        target = {
            'boxes': boxes,
            'labels': labels,
            'masks': masks,
        }

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target
