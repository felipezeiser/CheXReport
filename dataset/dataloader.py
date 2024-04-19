from typing import Tuple
import h5py
import json
import os
import random
import numpy as np
import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose

class HDF5Dataset(Dataset):
    """
    Dataset class to load data from HDF5 files and JSON files for captions and lengths.

    Parameters:
        hdf5_path (str): Path to the HDF5 file containing image data.
        captions_path (str): Path to the JSON file containing captions.
        lengths_path (str): Path to the JSON file containing the lengths of captions.
        pad_id (float): Padding ID to use for padding sequences.
        transform (Callable): Transformations to apply to the images.
    """
    def __init__(self, hdf5_path: str, captions_path: str, lengths_path: str, pad_id: float, transform: Compose = None):
        super().__init__()
        self.pad_id = pad_id

        with h5py.File(hdf5_path, 'r') as h5_file:
            self.keys, = h5_file.keys()
            self.images = np.array(h5_file[self.keys])

        with open(captions_path, 'r') as json_file:
            self.captions = json.load(json_file)

        with open(lengths_path, 'r') as json_file:
            self.lengths = json.load(json_file)

        self.transform = transform

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor]:
        image = torch.as_tensor(self.images[index], dtype=torch.float32) / 255.0
        if self.transform:
            image = self.transform(image)

        captions = [torch.tensor(c, dtype=torch.long) for c in self.captions[index]]
        captions = pad_sequence(captions, padding_value=self.pad_id)
        lengths = torch.tensor(self.lengths[index], dtype=torch.long)

        return image, captions, lengths

    def __len__(self) -> int:
        return len(self.images)

class CollatePad:
    """
    Collate function to pad batches of variable lengths to a fixed length.

    Parameters:
        max_len (int): Maximum length to pad the sequences.
        pad_id (float): Padding ID to use for sequences.
    """
    def __init__(self, max_len: int, pad_id: float):
        self.max_len = max_len
        self.pad_id = pad_id

    def __call__(self, batch: list) -> Tuple[Tensor, Tensor, Tensor]:
        images, captions, lengths = zip(*batch)
        images = torch.stack(images)

        captions = pad_sequence(captions, batch_first=True, padding_value=self.pad_id)
        if captions.size(1) < self.max_len:
            captions = torch.nn.functional.pad(captions, (0, self.max_len - captions.size(1)), value=self.pad_id)

        lengths = torch.stack(lengths)
        return images, captions, lengths

if __name__ == "__main__":
    SEED = 9001
    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    generator = torch.Generator()
    generator.manual_seed(SEED)

    path = "/srv/data/guszarzmo/mlproject/data/mscoco_h5/"
    dataset_paths = {split: {"images": f"{path}{split}_images.hdf5",
                             "captions": f"{path}{split}_captions.json",
                             "lengths": f"{path}{split}_lengthes.json"}
                     for split in ["train", "val", "test"]}

    datasets = {split: HDF5Dataset(**paths, pad_id=0) for split, paths in dataset_paths.items()}
    loader_params = {
        "batch_size": 100,
        "shuffle": True,
        "num_workers": 4,
        "worker_init_fn": lambda _: np.random.seed(int(torch.initial_seed()) % (2 ** 32)),
        "generator": generator
    }

    for split, dataset in datasets.items():
        data_loader = DataLoader(dataset, collate_fn=CollatePad(30, 0), **loader_params)
        for images, captions, lengths in tqdm(data_loader, desc=f"Processing {split} data"):
            pass
    print("Data loading complete.")
