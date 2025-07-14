import torch

from torch.optim import Adam
from torch.utils.data import DataLoader

from .dataset import ImageNetDataset
from .model_mae import MaskedAutoEncoderViT

device = "cuda" if torch.cuda.is_available() else "cpu"
train_path = ""
test_path = ""
val_path = ""


def train(batch_size: int, epochs: int):
    mae = MaskedAutoEncoderViT(img_size=224)
    mae = mae.to(device).train()

    optimizer = Adam(mae.parameters(), lr=1e-4)

    train_set = ImageNetDataset(train_path)
    # val_set = ImageNetDataset(val_path)

    train_data = DataLoader(train_set, batch_size, shuffle=True)
    # val_data = DataLoader(val_set, batch_size)

    for epoch in epochs:
        train_loss = 0

        for target in train_data:
            target: torch.Tensor = target.to(device)
            optimizer.zero_grad()

            pred = mae(target)
            train_loss += pred.item() / len(train_data)
            pred.backward()

            optimizer.step()

        print(f"[Epoch {epoch}] loss = {train_loss:.4f}")
