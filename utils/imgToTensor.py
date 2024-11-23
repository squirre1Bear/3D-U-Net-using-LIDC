import torch
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset

# 使用torch.utils.dataset将图像转换为张量。
# 使用之前需要重写里面 init getitem len方法。
class UnetDataset(Dataset):
    def __init__(self, transform):
        super().__init__()
        self.images = np.load(r'E:\LIDC_dataset\train_image_list.npy')
        self.labels = np.load(r'E:\LIDC_dataset\train_label_list.npy')
        self.transform = transform

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]

        image, label = self.transform(image, label)

        image = image.float().cuda()
        label = label.long().cuda()
        return image, label

    def __len__(self):
        return len(self.images)