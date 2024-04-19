import os
from argparse import Namespace
from pathlib import Path
from itertools import chain

import torch

from utils.types import ImagesAndCaptions
from utils.train_utils import seed_everything
from dataset.utils import parse_arguments, load_json_file, write_h5_dataset, save_json_file
from dataset.helper import parse_captions, map_captions_to_images, process_full_dataset, split_data, encode_captions, build_vocabulary

def load_data(json_file: str, images_directory: str, max_caption_length: int = 52) -> ImagesAndCaptions:
    """
    Carrega um arquivo JSON de anotações e retorna um dicionário com os IDs das imagens e suas legendas correspondentes.
    """
    annotations, image_ids = load_json_file(json_file)
    captions = parse_captions(annotations, max_caption_length)
    images_with_captions = map_captions_to_images(image_ids, captions, images_directory)
    return images_with_captions

def main():
    SEED = 9001
    seed_everything(seed=SEED)

    args = parse_arguments()  # Parse command line arguments
    dataset_directory = Path(os.path.expanduser(args.dataset_directory))
    output_directory = Path(os.path.expanduser(args.output_directory))
    output_directory.mkdir(parents=True, exist_ok=True)

    train_annotations_path = dataset_directory / args.train_annotations
    validation_annotations_path = dataset_directory / args.val_annotations
    train_images_directory = dataset_directory / args.train_images
    validation_images_directory = dataset_directory / args.val_images

    vector_directory = Path(os.path.expanduser(args.embedding_directory))
    vector_file = vector_directory.glob("*.zip").__next__()
    vector_name = f"{vector_file.stem}.{args.embedding_dimension}d"

    print("Processing annotation files...")
    train_images_and_captions = load_data(str(train_annotations_path), str(train_images_directory), args.max_caption_length)
    validation_images_and_captions = load_data(str(validation_annotations_path), str(validation_images_directory))

    train_set, validation_set, test_set = split_data(train_images_and_captions, validation_images_and_captions, SEED)

    print("Building vocabulary...")
    captions_for_vocab = [chain.from_iterable(dataset["captions"]) for dataset in train_set.values()]
    vocabulary = build_vocabulary(captions_for_vocab, str(vector_directory), vector_name, args.min_frequency)
    torch.save(vocabulary, str(output_directory / "vocab.pth"))

    for dataset, split_name in zip([train_set, validation_set, test_set], ["train", "val", "test"]):
        print(f"Processing {split_name} data split...")
        arrays = process_full_dataset(dataset=dataset, vocabulary=vocabulary, split_label=split_name)
        images, encoded_captions, lengths = arrays
        
        print(f"Number of samples in {split_name} split: {images.shape[0]}")
        write_h5_dataset(dataset_path=str(output_directory / f"{split_name}_images.hdf5"), dataset_name=split_name, dataset_data=images, dataset_type="uint8")
        save_json_file(str(output_directory / f"{split_name}_captions.json"), encoded_captions)
        save_json_file(str(output_directory / f"{split_name}_lengthes.json"), lengths)

        print(f"Data for {split_name} split saved successfully.\n")
        del images, encoded_captions, lengths

    print("\nAll datasets created and saved successfully.")

if __name__ == "__main__":
    main()
