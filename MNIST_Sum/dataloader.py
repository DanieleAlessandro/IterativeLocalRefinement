import pickle
from torch.utils.data import Dataset, DataLoader


class MNISTDataset(Dataset):
    def __init__(self, file_name):
        super(MNISTDataset).__init__()
        with open(file_name, 'rb') as f:
            self.dataset = pickle.load(f)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


def dataloader(filename, batch_size=32):
    dataset = MNISTDataset(filename)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )