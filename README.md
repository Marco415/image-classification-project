# image-classification-project
Image classification using three different models



### Project Overview

This project investigates multiclass land-use and land-cover classification using the EuroSAT multispectral satellite image dataset.


The task is to classify Sentinel-2 satellite image patches into one of ten land-cover classes.



### Dataset

**Dataset source:**

Hugging Face: Ehzoahis/EuroSAT


**URL:**

https://huggingface.co/datasets/Ehzoahis/EuroSAT


The dataset contains 27,000 labelled GeoTIFF images representing ten land-cover classes. The multispectral images contain 13 Sentinel-2 spectral bands.



#### Classes
1. Annual Crop
2. Forest
3. Herbaceous Vegetation
4. Highway
5. Industrial Buildings
6. Pasture
7. Permanent Crop
8. Residential Buildings
9. River
10. SeaLake



### Dataset Installation

The dataset is not included in the repository because of its size.


Download the EuroSAT dataset from the Hugging Face repository and place EuroSAT.zip in:

data/raw/EuroSAT.zip


Extract it so that the directory structure becomes:

    data/
        └── raw/
            └── EuroSAT/
                ├── train/
                ├── val/ 
                └── test/



### Experimental Split

The provided test partition is retained as the final held-out test set.


The provided training partition is stratified into training and validation subsets using random seed 42.


The final split is approximately:

- Training: 16,200 images
- Validation: 5,400 images
- Test: 5,400 images



### Models

Three different model families are compared:

1. ResNet-50
2. EfficientNetV2-S
3. Vision Transformer B/16


All three models use ImageNet pretrained weights and are adapted to accept the 13-band multispectral input.



### Preprocessing

The original GeoTIFF images contain 13 spectral bands at 64×64 pixels.


Images are resized to 224×224 pixels before being supplied to the neural networks.


Band-wise normalization statistics are calculated using the training set only.


Training augmentation consists of horizontal flips, vertical flips and 90-degree rotations.


No augmentation is applied to validation or test data.



### Reproducibility

The random seed is 42.


The experiment records:

- model architecture
- pretrained weights
- input dimensions
- batch size
- gradient accumulation
- optimizer
- learning rate
- learning-rate scheduler
- weight decay
- number of epochs
- early stopping
- loss function
- software versions
- hardware configuration



### Running the Project

**Create Jupyter kernel:**

See KernelCreation.md


Run the notebooks in this order:

1. 01_dataset_preparation.ipynb
2. 02_resnet50_training.ipynb
3. 03_efficientnetv2_training.ipynb
4. 04_vit_b16_training.ipynb
5. 05_final_evaluation.ipynb


The first notebook creates the train, validation and test split files.


The next three notebooks train the three classification models.


The final notebook evaluates the trained models on the common held-out test set and generates the comparison tables, classification reports and confusion matrices.



### Output

The project produces:

- trained model checkpoints
- training and validation curves
- test accuracy
- precision
- recall
- F1-score
- macro F1-score
- per-class metrics
- confusion matrices
- error-analysis examples
- model comparison tables



### Hardware

The experiments are intended to use an NVIDIA GPU where available.


The hardware and software versions used for the final experiments are recorded in the project report.



### Citation

The original EuroSAT dataset and all pretrained models, software libraries and external resources used in the project must be acknowledged in the final report.
