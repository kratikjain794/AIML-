from torch.utils.data import DataLoader
from utils.dataset import ColorizationDataset

dataset = ColorizationDataset(
    image_dir="dataset/train_samples"
)

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True
)

L, ab = next(iter(loader))

print("L Shape :", L.shape)

print("ab Shape :", ab.shape)