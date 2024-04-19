import argparse
import json
import h5py
import numpy as np
import torch
from torch import Tensor
from torch.nn.init import xavier_uniform_
from typing import List, Tuple
from argparse import Namespace

def parse_arguments() -> Namespace:
    """
    Parse command line arguments for the project.

    Returns:
        An argparse.Namespace with the parsed options.
    """
    parser = argparse.ArgumentParser(description="Parse command line arguments for the CheXReport dataset.")
    
    parser.add_argument("--dataset_directory", type=str, default="mimic",
                        help="Root directory containing the MIMIC-CXR dataset files.")
    parser.add_argument("--train_annotations", type=str, default="captions_train.json",
                        help="Path to the training annotations JSON file.")
    parser.add_argument("--val_annotations", type=str, default="captions_test.json",
                        help="Path to the validation annotations JSON file.")
    parser.add_argument("--train_images", type=str, default="images/train",
                        help="Directory containing training images.")
    parser.add_argument("--val_images", type=str, default="images/test",
                        help="Directory containing validation images.")
    parser.add_argument("--output_directory", type=str, default="mimic/",
                        help="Directory to save the processed output files.")
    parser.add_argument("--embedding_directory", type=str, default="../code_swin/glove",
                        help="Directory containing embedding vectors.")
    parser.add_argument("--embedding_dimension", type=int, default=300,
                        help="Dimensionality of the embedding vectors.")
    parser.add_argument("--min_frequency", type=int, default=2,
                        help="Minimum frequency to include a token in the vocabulary.")
    parser.add_argument("--max_caption_length", type=int, default=52,
                        help="Maximum length of captions.")

    return parser.parse_args()

def load_json_file(json_path: str) -> Tuple[List[dict], List[dict]]:
    """
    Load JSON file from a given path.

    Parameters:
        json_path (str): The file path to the JSON file.

    Returns:
        Tuple containing two lists: annotations and images.
    """
    with open(json_path, 'r') as file:
        data = json.load(file)

    return data["annotations"], data["images"]

def save_json_file(json_path: str, data: dict) -> None:
    """
    Save a dictionary to a JSON file.

    Parameters:
        json_path (str): The file path where to save the JSON data.
        data (dict): The data to save in JSON format.
    """
    with open(json_path, 'w') as file:
        json.dump(data, file, indent=4)

def write_h5_dataset(dataset_path: str, dataset_data: np.ndarray, dataset_name: str, dataset_type: str) -> None:
    """
    Write an array to an H5 file.

    Parameters:
        dataset_path (str): Path to the H5 file.
        dataset_data (np.ndarray): Data array to write.
        dataset_name (str): Name for the dataset within the H5 file.
        dataset_type (str): Data type of the dataset.
    """
    with h5py.File(dataset_path, "w") as file:
        file.create_dataset(name=dataset_name, data=dataset_data, shape=np.shape(dataset_data), dtype=dataset_type)

def seed_randomness(worker_id: int):
    """
    Seed the random number generators for PyTorch, NumPy, and the Python random module.
    
    Parameters:
        worker_id (int): The worker identifier, used to seed the generators.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def initialize_unknown_words(tensor: Tensor) -> Tensor:
    """
    Initialize unknown word vectors using Xavier uniform distribution.

    Parameters:
        tensor (Tensor): Tensor to initialize.

    Returns:
        Tensor: Initialized tensor.
    """
    initialized_tensor = torch.ones(tensor.size())
    return xavier_uniform_(initialized_tensor.view(1, -1)).view(-1)
