# Brain Tumor Segmentation using U-Net

## Overview

This project implements a 2D U-Net–based deep learning pipeline for brain tumor segmentation from MRI images.  
The primary goal is accurate tumor localization, using principled training, evaluation, 
and model selection practices.

Rather than chasing state-of-the-art scores, this work focuses on:

- baseline implementation in limited hardware
- correct methodology  
- robust evaluation  
- clear interpretation of results

---

## Problem Statement

Brain tumor segmentation is a challenging medical imaging task due to:

- high class imbalance (tumor pixels ≪ background pixels)  
- ambiguous tumor boundaries
- noisy and heterogeneous MRI slices  

Accurate segmentation is critical for:

- treatment planning  
- disease monitoring  
- clinical analysis

---
## System Specifications Used For Training

The training of this model was performed with the following system specifications:

- CPU: AMD Ryzen 5 4600H with Radeon Graphics (3.00 GHz)
- RAM: 16 GB 
- Storage: 1 TB HDD
- GPU: NVIDIA GeForce GTX 1650 Ti

---
## How To Run

1. Clone the repository
```bash
git clone https://github.com/<your-username>/brain_tumor_segmentation.git
cd brain_tumor_segmentation
```

2. Create and activate a virtual environment 

### Windows
```bash
python -m venv venv
venv\Scripts\activate
```
### Linux/macOS
```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```
4. Dataset setup

Arrange the dataset in the following structure:
<p align="center">
  <img src="assets\dataset_setup\dataset_setup.png"/>
</p>

- Each image must have a corresponding mask
- Filenames must match exactly
- Supported formats: .png, .jpg, .jpeg

Before training, the pipeline automatically checks dataset integrity and halts execution if mismatches are detected.

5. Configure experiment parameters

All experiment-level configuration is centralized in ```config.py```.

Key parameters include:
- ```IMAGE_SIZE```
- ```BATCH_SIZE```
- ```LEARNING_RATE```
- ```EPOCHS```
- ```THRESHOLD```
- dataset_paths
- output directories

Modify these values as required before running experiments.

6. Train the model

```bash
python train.py
```

During training:

- progress is displayed using tqdm
- metrics are logged to TensorBoard
- the best model checkpoint is saved automatically based on validation Dice score

To monitor training in real time:
```bash
tensorboard --logdir <LOG_DIR>
```

7. Run inference and evaluation

```bash
python test.py
```
This script produces:
- binary segmentation masks
- segmented images (background removed)
- overlay visualizations (prediction over original image)
- per-image Dice and IoU scores saved to metrics.csv

All outputs are saved under the directory specified by ```PREDICTION_ROOT``` in ```config.py```.

8. Output directory structure

After inference, the output directory will have the following structure:

<p align="center">
  <img src="assets\output_directory_structure\output_directory_structure.png"/>
</p>

### Note:
If the model is to be trained using a GPU, download the appropriate CUDA toolkit version that corresponds to the 
PyTorch version that is being used and run the setup for it. This training uses:
- PyTorch version:  2.7.1 
- CUDA toolkit version 11.8.0

You can find more information in the official PyTorch website regarding compatible PyTorch and CUDA toolkit versions.
- Official PyTorch website: https://pytorch.org
- NVIDIA toolkit archive: https://developer.nvidia.com/cuda-toolkit-archive

Without this, the model will not be trained using GPU.

---
## Project Structure

This following is the folder structure of the project:

<p align="center">
  <img src="assets\project_structure\project_structure.png"/>
</p>

---
## Model Architecture

A 2D U-Net architecture is used with:

- encoder–decoder structure  
- skip connections for spatial detail preservation  
- convolution + BatchNorm + ReLU blocks  
- binary segmentation output (tumor vs background)  

**Why U-Net?**

- proven effectiveness in medical image segmentation  
- strong and interpretable baseline  
- computationally efficient  
- widely accepted in academic literature  

This project uses a standard 2D UNET architecture.

---
## Dataset

- MRI brain images with corresponding tumor masks  
- ~3000 image–mask pairs  
- RGB slices (configurable to grayscale)  
- severe foreground–background imbalance  

### Data Integrity Checks

Before training:

- every image is matched with a corresponding mask  
- missing or mismatched files are detected  
- training proceeds only if dataset integrity is verified  

This ensures reproducibility and correctness.

### Example Image–Mask Pair

The figure below shows a representative sample from the dataset.

<p align="center">
  <img src="assets\dataset_example\1.png" width="350"/>
<span style="display:inline-block; width:20px;"></span>
  <img src="assets\dataset_example\1_mask.png" width="350"/>
</p>

<p align="center">
  <em>
    Left: Original MRI brain slice.  
    Right: Corresponding ground-truth tumor mask.
  </em>
</p>

---

## Data Preprocessing & Augmentation

### Preprocessing

- resize to `256 × 256`  
- normalize using ImageNet statistics  
- convert to tensors  

### Data Augmentation (Applied during runtime in training only)

- horizontal flip  
- vertical flip  
- random 90° rotations  

**Why augmentation?**

- improves generalization  
- reduces overfitting  
- simulates real-world variation

---

## Loss Function

A combined BCE + Dice loss is used

### Why BCE + Dice loss?

The combined BCE + Dice loss was selected based on prior findings in the medical image segmentation literature.  
In particular, the study:

> *"Robustness of different loss functions and their impact
on network’s learning capability"*  
> (arXiv:2110.08322)

reports that hybrid losses combining Dice and BCE terms consistently outperform single-component losses.

This motivated the use of a BCE–Dice formulation in this project.

---

## Evaluation Metrics

### Primary Metric

- Dice Coefficient is treated as  the main performance indicator

### Secondary Metrics

- Intersection over Union (IoU)
- Pixel Accuracy 

Pixel accuracy is inflated due to background dominance and is not used for model selection.

---

## Training Strategy

- train/validation split: 80 / 20
- optimizer: Adam  
- learning rate: `1e-3`  
- batch size: `16`  
- maximum epochs: `50`  
- best model was selected using validation Dice  

### Early Stopping 
Early stopping was used manually to stop the training.
Training was extended beyond the best epoch to confirm convergence.  
No further improvement was observed, and training was stopped to avoid overfitting.

---

## Training Behavior & Interpretation

Validation Dice exhibited:

- early fluctuations  
- recovery after temporary dips  
- a clear peak followed by overfitting  

This behavior was maybe observed due to:

- usage of Dice-based loss  
- applying random data augmentation  
- training on imbalanced medical data  

The best generalization performance was achieved at **Epoch 11**.

---

## Results & Analysis

This section presents both quantitative and qualitative evaluation of the trained U-Net model.  
All results are reported using the best validation checkpoint (Epoch 11), selected based on peak validation Dice score.


## Quantitative Results

The following plots summarize model behavior across training epochs:

- Training vs Validation Dice
- Training vs Validation Accuracy
- Validation IoU

## Training vs Validation Dice
<p align="center">
  <img src="assets\dice\training_dice.png">
  <img src="assets\dice\validation_dice.png">
</p>

<p align="center">
  <em>
    Top: Training Dice score.  
    Bottom: Validation Dice score.
  </em>
</p>

## Training vs Validation Accuracy
<p align="center">
  <img src="assets\accuracy\train_accuracy.png">
  <img src="assets\accuracy\validation_accuracy.png">
</p>

<p align="center">
  <em>
    Top: Training Accuracy.  
    Bottom: Validation Accuracy.
  </em>
</p>

## Validation IOU
<p align="center">
  <img src="assets\iou\validation_iou.png">
</p>

<p align="center">
  <em>
    Validation IOU
  </em>
</p>

### Interpretation

- Validation Dice shows fluctuations, that may be caused when using Dice-based losses and data augmentation.
- A clear peak at Epoch 11 indicates optimal generalization.
- Continued training beyond this point led to overfitting, as training Dice increased while validation Dice declined.

This behavior confirms stable convergence and justifies early stopping based on validation Dice.

---
## Final Results

| Metric               | Value |
|----------------------|-------|
| Best Validation Dice | 0.547 |
| Validation IoU       | ~0.41 |
| Training Dice        | ~0.56 |

### Interpretation

- the model reliably localizes tumor regions  
- boundary precision remains imperfect
- performance is not state-of-the-art but can be considered as a baseline model  

---
## Qualitative Results
To complement numerical metrics, the segmentation outputs were visually inspected.

Each example includes:

1. binary segmentation mask  
2. segmented image (background removed)  
3. overlay visualization (mask over original image)  
4. per-image Dice & IoU stored in CSV 

<p align="center">
  <img src="assets\pipeline_output\pipeline_output.PNG">
</p>

<p align="center">
  <em>
    Outputs from the pipeline.
  </em>
</p>

---

## Success Cases

The following examples illustrate cases where the model performs well.

<p align="center">
  <img src="assets\success_cases\3015.png" width="350"/>
<span style="display:inline-block; width:20px;"></span>
  <img src="assets\success_cases\3026.png" width="350"/>
</p>

### Observations

- The model successfully **localizes tumor regions**
- Main tumor mass is accurately captured
- Shape and spatial consistency are preserved
- Minor boundary inaccuracies do not significantly affect localization

These cases demonstrate the model’s ability to learn robust spatial representations despite class imbalance.

---
## Limitations

This project has the following limitations:

- uses 2D slices only and has no 3D context  
- single-modality input and output  
- no post-processing  
- small tumors remain challenging

---
## Failure Cases

The following examples highlight limitations of the model.

<p align="center">
  <img src="assets\failure_cases\3009.png" width="350"/>
<span style="display:inline-block; width:20px;"></span>
  <img src="assets\failure_cases\3036.png" width="350"/>
</p>

### Observations

- Small or diffuse tumors are partially missed
- Boundary precision degrades in low-contrast regions
- False positives occasionally appear near tissue boundaries

These errors are primarily due to:
- slice-wise 2D processing (lack of 3D context)
- ambiguous tumor boundaries
- severe foreground–background imbalance

---

## Discussion

The qualitative and quantitative results together indicate that:

- The model provides reliable tumor localization
- Dice ≈ 0.55 
- Visual results align well with numerical performance
- Limitations are structural rather than implementation-related

Despite modest Dice values, the model demonstrates consistent tumor localization.

---

## Error Analysis Summary

| Error Type                      | Cause                                                       |
|---------------------------------|-------------------------------------------------------------|
| Missed small tumors             | Low pixel count and Dice sensitivity                        |
| Boundary inaccuracies           | MRI may have ambiguous contrast                             |
| False positives                 | Similar intensity tissues may be present                    |
| Slice inconsistency             | Absence of 3D context                                       |
| Large tumors with black regions | Model could get confused since the region mimics background |


---

## Key Takeaways

- Quantitative metrics and qualitative results are consistent
- The model demonstrates stable generalization
- Performance is appropriate for a baseline study
- Clear directions for improvement are identified

This analysis confirms that the proposed pipeline is methodologically sound, 
reproducible, and suitable as a foundation for future work.

---
## Libraries Used

This project is implemented in Python and uses the following libraries:

- PyTorch
- torchvision
- Pillow (PIL)
- NumPy
- albumentations
- tqdm
- TensorBoard
- scikit-learn
- Custom implementations for Dice score, IoU, and Pixel Accuracy

---

## Future Improvements

Possible extensions include:

- Attention U-Net   
- learning-rate scheduling  
- 3D U-Net   
- post-processing for boundary refinement  

---

## Author

**Raghav Sarangan**  
Computer Science Engineer | Aspiring M.Sc. Student  


