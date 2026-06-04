# XSA-Net: Multi-Modal Deep Learning for Post-Thermal Ablation Recurrence Prediction

A PyTorch-based deep learning framework for predicting local tumor progression (LTP) after thermal ablation treatment using multi-parametric pre- and post-treatment MRI combined with clinical variables.

## Overview

This project addresses the clinical challenge of predicting local tumor progression (LTP) following thermal ablation therapy. By integrating multi-parametric 3D medical imaging (pre- and post-operative MRI sequences) with clinical patient data, the model provides accurate risk stratification to guide clinical decision-making and patient management.

### Key Features

- **Multi-Parametric MRI Integration**: Processes multiple MRI sequences including:
  - Pre-treatment: DWI, T1-weighted, T2-weighted
  - Post-treatment: T1-weighted with contrast
  - Tumor masks for both pre- and post-treatment phases
  
- **3D ResNet Encoders**: Uses 3D ResNet-18/50 backbones for volumetric feature extraction from medical images

- **Cross-Modal Attention**: Implements co-attention mechanisms for effective fusion of multi-sequence imaging data

- **Self-Attention Blocks**: Captures long-range dependencies within each modality and for clinical-imaging feature fusion

- **Clinical Variable Integration**: Fuses imaging features with standardized clinical variables through dedicated normalization and attention mechanisms

- **Survival Analysis**: Supports Cox proportional hazards loss for time-to-event prediction with Efron/Breslow tie handling

## Project Structure

```
.
├── main.py                      # Main training/evaluation script
├── run.py                       # Alternative entry point for training pipeline
├── dataset.py                   # Dataset class for loading 3D medical images and clinical data
├── utils.py                     # Training/validation utilities and metrics
├── coxloss.py                   # Cox proportional hazards loss for survival analysis
├── Cindex.py                    # Concordance index calculation for survival metrics
├── model/                       # Model architecture definitions
│   ├── model.py                 # XSA_Net: Multi-view model with attention mechanisms
│   └── ResNet3D.py             # 3D ResNet backbone implementation

```

## Model Architecture

The `XSA_Net` (Cross-modal Self-Attention Network) model consists of:

1. **Multiple 3D ResNet Encoders**: Each MRI sequence is processed by dedicated 3D ResNet encoders
2. **Co-Plane Attention**: Enables information exchange between different anatomical planes (Sagittal, Coronal, Axial)
3. **Cross-Modal Attention**: Fuses features across different MRI sequences (DWI, T1, T2, post-contrast)
4. **Clinical Feature Integration**: 
   - Clinical variables are normalized and projected to match imaging feature dimensions
   - Self-attention mechanism fuses clinical and imaging features
5. **Prediction Head**: Outputs risk scores for LTP prediction

### Architecture Variants

The model supports multiple configurations via command-line arguments:
- `--post_img_only`: Use only post-treatment imaging
- `--no_post_img`: Exclude post-treatment imaging
- `--no_clic`: Exclude clinical variables
- `--no_co_att`: Disable co-plane attention
- `--encoder_only`: Use encoder outputs directly without attention

## Data Format

The dataset should be organized as follows:
```
data_dir/
├── data/                       # Directory containing 3D image data (.npy files)
│   └── {patient_id}_{modality}.npy
├── baseline.xlsx              # Training set labels and clinical data
├── val.xlsx                   # Validation set labels
├── test1.xlsx                 # Test set 1 labels
└── test2.xlsx                 # Test set 2 labels
```

### Excel File Structure

Each Excel file should contain columns for:
- **ID**: Patient identifier
- **File name**: Reference to image files
- **Center**: Hospital/center identifier
- **Event**: Event indicator (0/1)
- **TimeToEvent**: Time to event or censoring
- **LTP**: Local Tumor Progression indicator (target variable)
- **LTPT**: Time to LTP
- **Clinical variables**: From column 9 onwards (standardized during training)

### Imaging Modalities

The model expects the following modalities (configurable via `SequenceList`):
- `dwipre`: Pre-treatment DWI
- `t2pre`: Pre-treatment T2-weighted
- `t1pre`: Pre-treatment T1-weighted
- `premask`: Pre-treatment tumor mask
- `t1post`: Post-treatment T1-weighted with contrast
- `t1tumourpost`: Post-treatment tumor mask

## Requirements

- Python 3.7+
- PyTorch 1.10+
- torchvision
- NumPy
- pandas
- scikit-learn
- OpenCV (cv2)
- scipy
- tqdm
- matplotlib
- lifelines (for C-index calculation)

## Usage

### Training

```bash
python main.py \
    --data_dir /path/to/data \
    --output_dir /path/to/output \
    --model_name XSA \
    --epochs 300 \
    --lr 3e-5 \
    --batch_size 64 \
    --use_cox_loss \
    --seed 388
```

### Evaluation

```bash
python main.py \
    --eval_only /path/to/model/directory \
    --eval_model chkpt_epoch-94.pt \
    --data_dir /path/to/data
```

### Key Arguments

- `--model_name`: Model architecture ('XSA' for multi-view model, 'res' for baseline ResNet3D)
- `--use_cox_loss`: Enable Cox proportional hazards loss for survival analysis
- `--use_class_weights`: Apply class weighting for imbalanced datasets
- `--augment`: Enable data augmentation (random crop, flip, rotation)
- `--ssl`: Path to self-supervised pre-trained weights
- `--clip_len`: Number of frames/slices to sample (default: 4)
- `--sampling_rate`: Temporal sampling rate (default: 1)
- `--n_TTA`: Number of test-time augmentation samples
- `--emb_dim`: Embedding dimension for attention mechanisms (default: 512)
- `--alpha`: Weighting parameter for feature fusion (default: 0.1)

### Architecture Configuration

- `--backbone`: Backbone architecture (default: 'ResNet3D')
- `--model_depth`: ResNet depth (18 or 50, default: 18)
- `--active_branch`: Active branches for multi-view processing (default: [0, 0, 1])
- `--no_co_att`: Disable co-plane attention
- `--no_cross_modal`: Disable cross-modal attention
- `--no_post_img`: Exclude post-treatment imaging
- `--post_img_only`: Use only post-treatment imaging
- `--no_clic`: Exclude clinical variables
- `--encoder_only`: Use encoder outputs directly

## Loss Functions

- **Binary Cross-Entropy**: Standard classification loss
- **Cox Loss**: Survival analysis loss for time-to-event data with Efron and Breslow tie handling

## Evaluation Metrics

- **Concordance Index (C-index)**: Primary metric for survival analysis
- **AUROC**: Area Under Receiver Operating Characteristic curve
- **AUPR**: Area Under Precision-Recall curve
- **Accuracy** and **Balanced Accuracy**
- **Matthews Correlation Coefficient (MCC)**
- **Precision**, **Recall**, **F1-score**

## Training Details

The training process includes:
- Early stopping based on validation C-index
- Model checkpointing for best performing epochs
- Patient-level aggregation for metrics (multiple samples per patient)
- StandardScaler normalization for clinical variables
- Support for multi-GPU training via DataParallel

## License

This project is for research purposes. Please cite appropriately if used in academic work.

## Acknowledgments

This project uses:
- 3D ResNet implementations adapted from torchvision
- Cox loss implementation with Efron/Breslow tie handling
- Kinetics-400 pre-trained weights for 3D CNN initialization
