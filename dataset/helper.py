from typing import List, Tuple
from numpy.typing import NDArray
from utils.types import Captions, ImagesAndCaptions

from collections import defaultdict, Counter
from itertools import chain
import multiprocessing as mp
from functools import partial
from tqdm import tqdm
import re

import numpy as np
from torchtext.vocab import Vocab

from sklearn.model_selection import train_test_split

import cv2

from .utils import initialize_unknown_words

def parse_captions(annotations: List[dict], max_length: int) -> Captions:
    """
    Extrai e processa legendas de uma lista de anotações, adicionando tokens
    de início e fim e truncando legendas que excedam o comprimento máximo permitido.
    """
    captions_by_image = defaultdict(list)
    for annotation in annotations:
        tokens = [token for token in re.split(r"(\W)", annotation["caption"]) if token.strip()]
        truncated_tokens = tokens[:max_length - 2] if len(tokens) > (max_length - 2) else tokens
        captions_by_image[annotation["image_id"]].append(["<sos>"] + truncated_tokens + ["<eos>"])
    return captions_by_image

def map_captions_to_images(image_ids: List[dict], captions_by_image: Captions, images_directory: str) -> ImagesAndCaptions:
    """
    Associa cada imagem a uma lista selecionada aleatoriamente de legendas.
    """
    images_and_captions = {}
    for image in image_ids:
        image_id = image["id"]
        if image_id in captions_by_image:
            selected_indices = np.random.choice(len(captions_by_image[image_id]), size=1, replace=False)
            selected_captions = [captions_by_image[image_id][idx] for idx in selected_indices]
            image_path = f"{images_directory}/{image['file_name']}"
            images_and_captions[image_path] = {"image_id": image_id, "captions": selected_captions}
    return images_and_captions

def load_and_resize_image(image_path: str, resize_dim: Tuple[int, int]) -> NDArray:
    """
    Carrega e redimensiona uma imagem para as dimensões especificadas.
    """
    image = cv2.imread(image_path)
    if image is not None:
        resized_image = cv2.resize(image, resize_dim, interpolation=cv2.INTER_AREA)
        return resized_image.transpose(2, 0, 1)
    return None

def encode_captions(captions: List[List[str]], vocabulary: Vocab) -> Tuple[List[List[int]], List[int]]:
    """
    Codifica as legendas em índices inteiros usando um vocabulário fornecido.
    """
    encoded_captions = []
    lengths = []
    for caption in captions:
        encoded_caption = [vocabulary.stoi[word] for word in caption]
        encoded_captions.append(encoded_caption)
        lengths.append(len(encoded_caption))
    return encoded_captions, lengths

def split_data(original_train: ImagesAndCaptions, original_val: ImagesAndCaptions, seed: int, test_pct: float = 0.15, val_pct: float = 0.15) -> Tuple[ImagesAndCaptions, ImagesAndCaptions, ImagesAndCaptions]:
    """
    Divide os dados em conjuntos de treinamento, validação e teste.
    """
    total_size = len(original_train) + len(original_val)
    val_adjusted_size = int(total_size * val_pct) - len(original_val)
    train_size = int(total_size * (1 - test_pct - val_pct))

    original_train_list = list(original_train.items())
    additional_val_data, remaining_data = train_test_split(original_train_list, train_size=val_adjusted_size, random_state=seed)

    train_data, val_data = train_test_split(remaining_data, train_size=train_size, random_state=seed)
    val_data.extend(additional_val_data)

    return dict(train_data), dict(val_data), original_val

def build_vocabulary(captions: List[chain], vector_dir: str, vector_file: str, min_freq: int = 2) -> Vocab:
    """
    Constrói um vocabulário a partir de uma lista de legendas, filtrando palavras por frequência mínima.
    """
    all_words = list(chain.from_iterable(captions))
    word_counts = Counter(all_words)
    return Vocab(word_counts, min_freq=min_freq, specials=("<unk>", "<pad>", "<sos>", "<eos>"), vectors_cache=vector_dir, vectors=vector_file, unk_init=initialize_unknown_words)

def process_dataset_entries(dataset: Tuple[str, Captions], vocabulary: Vocab, image_dims: Tuple[int, int]) -> Tuple[NDArray, List[List[int]], List[int]]:
    """
    Carrega imagens e codifica legendas de um único conjunto de dados.
    """
    image, captions_data = dataset
    loaded_image = load_and_resize_image(image, image_dims)
    encoded_captions, caption_lengths = encode_captions(captions_data["captions"], vocabulary)
    return loaded_image, encoded_captions, caption_lengths

def process_full_dataset(dataset: ImagesAndCaptions, vocabulary: Vocab, split_label: str, num_processes: int = 4) -> Tuple[NDArray, List[List[List[int]]], List[List[int]]]:
    """
    Processa imagens e legendas para todo o conjunto de dados usando processamento paralelo.
    """
    create_arrays_func = partial(process_dataset_entries, vocabulary=vocabulary, image_dims=(384, 384))
    with mp.Pool(processes=min(num_processes, mp.cpu_count())) as pool:
        results = list(tqdm(pool.imap(create_arrays_func, dataset.items()), total=len(dataset), desc=f"Preparing {split_label} Dataset", unit="image"))
    images, encoded_captions, lengths = zip(*results)
    return np.stack(images), encoded_captions, lengths
