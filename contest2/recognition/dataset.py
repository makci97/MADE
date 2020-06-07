import json, os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from contest2.recognition.common import abc, is_valid_str, convert_to_eng

from contest2.alexyar88_baseline_utils import four_point_transform


class RecognitionDataset(Dataset):

    def __init__(self, data_path, config_file=None, abc=abc, transforms=None, val_split=0.95, is_train=True):
        super(RecognitionDataset, self).__init__()
        self.data_path = data_path
        self.abc = abc
        self.transforms = transforms

        with open(config_file, 'r') as f:
            self.marks = json.load(f)
            if is_train:
                self.marks = self.marks[:int(len(self.marks) * val_split)]
            else:
                self.marks = self.marks[int(len(self.marks) * val_split):]

        final_marks = []
        for elem in self.marks:
            text = elem['text']
            text = convert_to_eng(text.upper())  # samples can have russian characters or lower case
            if is_valid_str(text):
                elem['text'] = text
                final_marks.append(elem)
        self.marks = final_marks

    def __len__(self):
        return len(self.marks)

    def __getitem__(self, idx):
        item = self.marks[idx]
        img_path = os.path.join(self.data_path, item["file"].lstrip('data').lstrip('/'))
        img = cv2.imread(img_path)

        if item['boxed']:
            x_min, y_min, x_max, y_max = item['box']
            img = img[y_min:y_max, x_min:x_max]
        else:
            points = np.clip(np.array(item['box']), 0, None)
            img = four_point_transform(img, points)

        text = item['text']
        seq = self.text_to_seq(text)
        seq_len = len(seq)

        if self.transforms is not None:
            img = self.transforms(img)

        output = {
            'image': img,
            'text': text,
            'seq': seq,
            'seq_len': seq_len
        }

        return output

    def text_to_seq(self, text):
        seq = [self.abc.find(c) + 1 for c in text]
        return seq

    @staticmethod
    def collate_fn(batch):
        images = list()
        seqs = list()
        seq_lens = list()
        for sample in batch:
            images.append(sample["image"])
            # images.append(torch.from_numpy(sample["image"].transpose((2, 0, 1))).float())
            seqs.extend(sample["seq"])
            seq_lens.append(sample["seq_len"])
        images = torch.stack(images)
        seqs = torch.Tensor(seqs).int()
        seq_lens = torch.Tensor(seq_lens).int()
        batch = {"images": images, "seqs": seqs, "seq_lens": seq_lens}
        return batch
