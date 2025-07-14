import os
import torch

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms


class ImageNetDataset(Dataset):
    def __init__(self, path: str):
        super().__init__()

        self.path = Path(path)
        self.img_path_list = [self.path / p for p in os.listdir(path)]

        self.total = len(os.listdir(path))
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __len__(self):
        return self.total

    def __getitem__(self, index: int) -> torch.Tensor:
        img_path = self.img_path_list[index]
        img = Image.open(img_path)
        img_tensor = self.transform(img)

        return img_tensor
