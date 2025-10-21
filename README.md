# Leaffliction - Plant Disease Classification

A computer vision project focused on classifying plant diseases through leaf image analysis, developed as part of the 42 School curriculum.

## Project Overview

Leaffliction is a comprehensive machine learning pipeline that analyzes plant leaf images to identify various diseases. The project encompasses data analysis, augmentation, image transformation, and classification using deep learning techniques.

## Key Features

### 1. Data Analysis & Visualization
- Automated dataset exploration and statistical analysis
- Generation of pie charts and bar charts showing class distribution
- Hierarchical directory structure analysis
- Identification of class imbalance issues

### 2. Data Augmentation
- Implementation of 6+ augmentation techniques:
  - Flip (horizontal/vertical)
  - Rotation
  - Skew
  - Shear
  - Crop
  - Distortion
- Automatic dataset balancing
- Preservation of original file naming conventions

### 3. Image Transformation Pipeline
- Advanced image processing using PlantCV or similar libraries
- Multiple transformation techniques:
  - Gaussian blur
  - Masking
  - ROI object detection
  - Object analysis
  - Pseudolandmark detection
  - Color histogram analysis
- Batch processing capabilities
- Flexible CLI with custom arguments

### 4. Classification System
- Deep learning model training for disease recognition
- Separation of training and validation datasets
- Model persistence and deployment
- Real-time prediction with visual output
- **Target accuracy: >90% on validation set (minimum 100 images)**

## Technical Challenges

### Data Management
- Handling imbalanced datasets across multiple plant species and disease types
- Managing large-scale image augmentation without quality loss
- Ensuring proper train/validation split to prevent overfitting

### Image Processing
- Implementing robust leaf segmentation algorithms
- Handling varying image quality, lighting conditions, and backgrounds
- Extracting meaningful features from diverse leaf morphologies

### Model Performance
- Achieving >90% accuracy requirement on validation data
- Preventing overfitting while maintaining generalization
- Optimizing model architecture for multi-class classification

### Infrastructure
- Working within 42's cluster environment constraints
- Managing storage limitations (using goinfre when necessary)
- Creating SHA1 signatures for dataset integrity verification

## Skills Developed

### Computer Vision
- Image preprocessing and enhancement techniques
- Feature extraction from biological samples
- Understanding of color spaces and their applications
- Object detection and segmentation

### Machine Learning
- Dataset preparation and curation
- Data augmentation strategies
- Model architecture selection and tuning
- Training pipeline development
- Overfitting prevention techniques
- Model evaluation and validation

### Software Engineering
- Clean code architecture following coding standards (flake8 for Python)
- CLI design and argument parsing
- Batch processing systems
- File I/O operations and directory management
- Version control with Git
- Documentation and reproducibility

### Domain Knowledge
- Plant pathology basics
- Disease classification categories
- Visual characteristics of plant diseases
- Agricultural computer vision applications

## Project Structure

```
leaffliction/
├── Distribution.[ext]      # Dataset analysis and visualization
├── Augmentation.[ext]      # Image augmentation tool
├── Transformation.[ext]    # Image transformation pipeline
├── train.[ext]            # Model training script
├── predict.[ext]          # Inference and prediction
└── signature.txt          # SHA1 hash of dataset
```

## Requirements

- Python 3.x (recommended) or language of choice
- Machine learning libraries (TensorFlow/PyTorch/Keras)
- Image processing libraries (OpenCV, PIL, PlantCV)
- Visualization libraries (Matplotlib, Seaborn)
- Code must follow flake8 standards if using Python

## Usage Examples

```bash
# Analyze dataset distribution
./Distribution.[extension] ./Apple

# Augment a single image
./Augmentation.[extension] ./Apple/apple_healthy/image.JPG

# Transform images in batch
./Transformation.[extension] -src Apple/apple_healthy/ -dst output/ -mask

# Train the model
./train.[extension] ./Apple/

# Make predictions
./predict.[extension] ./Apple/apple_healthy/image.JPG
```

## Validation & Submission

- All code must be submitted via Git repository
- Dataset must NOT be included in the repository
- `signature.txt` file containing SHA1 hash of dataset.zip is mandatory
- Signature verification during evaluation (mismatches result in grade 0)
- Model must achieve >90% accuracy on validation set

## Learning Outcomes

This project provides hands-on experience with the complete machine learning workflow, from raw data to deployed model. It emphasizes the importance of data quality, proper validation techniques, and the practical challenges of real-world computer vision applications in agriculture and plant pathology.

---

*Developed as part of the 42 School curriculum - A project that transforms understanding of both computer vision and plant disease diagnostics.*