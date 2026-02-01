"""
Brain Tumor Segmentation using U-Net

This file contains all configurations
"""

import os


# Root directory of the project


# Automatically resolve project root directory
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


# Dataset paths

DATASET_ROOT = os.path.join(PROJECT_ROOT, "data")

IMAGE_DIR = os.path.join(DATASET_ROOT, "train_images")
MASK_DIR = os.path.join(DATASET_ROOT, "train_masks")

TEST_IMAGE_DIR = os.path.join(DATASET_ROOT, "test_images")
TEST_MASK_DIR = os.path.join(DATASET_ROOT, "test_masks")



EXPERIMENT_NAME = "unet_brain_segmentation_training_1" # Change this to change the experiment name



# Model and log_dir config

MODEL_DIR = os.path.join(PROJECT_ROOT, "saved_model", EXPERIMENT_NAME)
LOG_DIR = os.path.join(PROJECT_ROOT, "log_dir", EXPERIMENT_NAME)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)



# LOSS & METRICS CONFIGURATION

# Available options:
# "bce_dice" (default)
# If user wants, any loss unction can be included here

LOSS_TYPE = "bce_dice"

# Threshold for binarizing predictions
THRESHOLD = 0.5 # can change this and observe the results


# Testing and inference config

PREDICTION_ROOT = os.path.join(
    PROJECT_ROOT,
    "results",
    EXPERIMENT_NAME,
    "final_evaluation"
)

os.makedirs(PREDICTION_ROOT, exist_ok=True)

