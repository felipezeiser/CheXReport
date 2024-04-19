from argparse import Namespace, ArgumentParser

def parse_arguments() -> Namespace:
    """
    Parse command line arguments for the testing utilities.

    Returns:
        An argparse.Namespace with the parsed options. This includes:
        - dataset_directory: Directory containing the processed MS COCO dataset.
        - output_directory: Directory to save the output files.
        - config_file_path: Path for the configuration JSON file.
        - checkpoint_path: Path to the checkpoint file.
        - device_type: Specifies the device type (e.g., 'gpu', 'cpu', or 'mgpu').
    """
    parser = ArgumentParser(description="Testing utility for CheXReport.")

    parser.add_argument("--dataset_directory", type=str, default="mimic",
                        help="Directory contains processed MS COCO dataset.")

    parser.add_argument("--output_directory", type=str, default="test",
                        help="Directory to save the output files.")

    parser.add_argument("--config_file_path", type=str, default="config.json",
                        help="Path for the configuration JSON file.")

    parser.add_argument("--checkpoint_path", type=str, default="checkpoint_best.pth.tar",
                        help="Path to the checkpoint file.")

    parser.add_argument("--device_type", type=str, default="gpu", choices=['gpu', 'cpu', 'mgpu'],
                        help="Specifies the device type for computation ('gpu', 'cpu', or 'mgpu').")

    return parser.parse_args()

# Example usage within a script
if __name__ == "__main__":
    args = parse_arguments()
    print("Configuration loaded for the project:")
    print(f"Dataset directory: {args.dataset_directory}")
    print(f"Output directory: {args.output_directory}")
    print(f"Config file path: {args.config_file_path}")
    print(f"Checkpoint path: {args.checkpoint_path}")
    print(f"Device type: {args.device_type}")
