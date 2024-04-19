import os
import random
import argparse
from argparse import Namespace
import json

import numpy as np
import torch

def parse_arguments() -> Namespace:
    """
    Parse command line arguments for the training script.

    Returns:
        An argparse.Namespace with the parsed options. This includes:
        - dataset_directory: Directory containing the processed MS COCO dataset.
        - config_file_path: Path for the configuration JSON file.
        - device_type: Device to be used for computation ('gpu' or 'cpu').
        - resume_checkpoint: Path to a checkpoint file to resume training.
    """
    parser = argparse.ArgumentParser(description="Training script for CheXReport.")

    parser.add_argument("--dataset_directory", type=str, default="mimic",
                        help="Directory contains processed MIMIC dataset.")

    parser.add_argument("--config_file_path", type=str, default="config.json",
                        help="Path for the configuration JSON file.")

    parser.add_argument("--device_type", type=str, default="gpu", choices=['gpu', 'cpu'],
                        help="Device to be used for computation, either 'gpu' or 'cpu'.")

    parser.add_argument("--resume_checkpoint", type=str, default="",
                        help="Path to a checkpoint file to resume training from.")

    return parser.parse_args()

def seed_everything(seed: int = 42):
    """
    Seed all possible random generators to ensure reproducible results.

    Parameters:
        seed (int): The seed value.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_json(json_path: str) -> dict:
    """
    Load data from a JSON file.

    Parameters:
        json_path (str): The path to the JSON file.

    Returns:
        dict: Data loaded from the JSON file.
    """
    with open(json_path, 'r') as file:
        data = json.load(file)
    return data

# Example usage within a script
if __name__ == "__main__":
    args = parse_arguments()
    print("Configuration loaded:")
    print(f"Dataset directory: {args.dataset_directory}")
    print(f"Config file path: {args.config_file_path}")
    print(f"Device type: {args.device_type}")
    print(f"Resume from checkpoint: {args.resume_checkpoint}")

    # Optionally, set a deterministic behavior
    seed_everything(42)
