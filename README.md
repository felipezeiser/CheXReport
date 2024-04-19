
# CheXReport

## Getting Started

### Download the Dataset
Download the MIMIC-CXR dataset from [PhysioNet](https://physionet.org/content/mimic-cxr/2.0.0/) and add it to the `mimic` folder in your project directory.

### Data Preprocessing
To preprocess the data, run the following command:

```bash
python create_dataset.py
```

### Training
To start the training process, execute:

```bash
python run_train.py --dataset_directory='mimic' --config_file_path='config.json' --device_type='gpu'
```

### Inference
For inference, use the following script:

```bash
python predict.py --dataset_directory='mimic' --output_directory='results' --config_file_path='config.json' --checkpoint_path='1904.0213/checkpoint_best.pth.tar'
```

### Model Weights
Download the weights for the model from this [Google Drive link](https://drive.google.com/file/d/1oAfGJNxJQRN4UOOFCrmpPkdMGDhNBEQm/view?usp=sharing).

## License

## Contact
felipezeiser@gmail.com
```
