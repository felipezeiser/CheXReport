import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torchvision.transforms import Normalize, Compose

from pathlib import Path

from models.cnn_encoder import ImageEncoder
from models.IC_encoder_decoder.transformer import Transformer
from dataset.dataloader_old import HDF5Dataset, collate_padd
from trainer import Trainer
from utils.train_utils import parse_arguments, seed_everything, load_json
from utils.gpu_cuda_helper import select_device

def get_datasets(dataset_dir: str, pad_id: int):
    """
    Load training and validation datasets.

    Parameters:
        dataset_dir (str): Directory path where datasets are located.
        pad_id (int): Padding ID used in the datasets.

    Returns:
        Tuple[DataLoader, DataLoader]: A tuple containing training and validation dataloaders.
    """
    # Define paths
    dataset_dir = Path(dataset_dir)
    paths = {
        "train": ("train_images.hdf5", "train_captions.json", "train_lengthes.json"),
        "val": ("val_images.hdf5", "val_captions.json", "val_lengthes.json")
    }

    # Image transformations
    norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = Compose([norm])

    datasets = {}
    for phase, (img_path, cap_path, len_path) in paths.items():
        hdf5_path = dataset_dir / img_path
        captions_path = dataset_dir / cap_path
        lengthes_path = dataset_dir / len_path

        datasets[phase] = HDF5Dataset(hdf5_path=str(hdf5_path),
                                      captions_path=str(captions_path),
                                      lengthes_path=str(lengthes_path),
                                      pad_id=pad_id,
                                      transform=transform)

    return datasets["train"], datasets["val"]

if __name__ == "__main__":
    # Command line arguments
    args = parse_arguments()
    print(f"Configuring training with dataset directory: {args.dataset_dir}")

    # Device setup
    device = select_device(args.device)
    print(f"Using device: {device}")

    # Load configuration and vocab
    config = load_json(args.config_path)
    vocab = torch.load(Path(args.dataset_dir) / "vocab.pth")
    vocab_size = len(vocab)
    pad_id = vocab.stoi["<pad>"]

    # Seed for reproducibility
    seed_everything(config["seed"])

    # Prepare data loaders
    train_dataset, val_dataset = get_datasets(args.dataset_dir, pad_id)
    loader_params = config["dataloader_params"]
    train_loader = DataLoader(train_dataset, collate_fn=collate_padd(config["max_len"], pad_id), **loader_params)
    val_loader = DataLoader(val_dataset, collate_fn=collate_padd(config["max_len"], pad_id), **loader_params)

    # Models setup
    image_encoder = ImageEncoder(**config["image_encoder_params"])
    transformer = Transformer(vocab_size=vocab_size, pad_id=pad_id, **config["transformer_params"])

    # Optimizers and schedulers setup
    encoder_optimizer = Adam(image_encoder.parameters(), lr=config["encoder_lr"])
    transformer_optimizer = Adam(transformer.parameters(), lr=config["transformer_lr"])
    encoder_scheduler = StepLR(encoder_optimizer, step_size=1, gamma=config["lr_decay"])
    transformer_scheduler = StepLR(transformer_optimizer, step_size=1, gamma=config["lr_decay"])

    # Trainer setup and execution
    trainer = Trainer([encoder_optimizer, transformer_optimizer],
                      [encoder_scheduler, transformer_scheduler],
                      device=device, pad_id=pad_id, epochs=config["epochs"],
                      checkpoints_path=config["checkpoints_path"])

    print("Starting training...")
    trainer.train(image_encoder, transformer, train_loader, val_loader)
    print("Training completed.")
