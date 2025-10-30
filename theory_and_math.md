# Leaffliction - Teoría del Proyecto

## Tabla de Contenidos

1. [Computer Vision Fundamentals](#1-computer-vision-fundamentals)
2. [Dataset Analysis](#2-dataset-analysis)
3. [Data Augmentation](#3-data-augmentation)
4. [Image Transformation Pipeline](#4-image-transformation-pipeline)
5. [Convolutional Neural Networks (CNN)](#5-convolutional-neural-networks-cnn)
6. [Training Process](#6-training-process)
7. [Model Evaluation](#7-model-evaluation)
8. [Prediction & Inference](#8-prediction--inference)

---

## 1. Computer Vision Fundamentals

### 1.1 Definition

Computer vision is a field of artificial intelligence that enables machines to interpret and understand visual information from digital images. The process involves:

```
Image Acquisition → Preprocessing → Feature Extraction → Classification → Decision
```

### 1.2 Digital Image Representation

Images are represented as multidimensional arrays:

**Grayscale Image**:
```
Shape: (Height, Width)
Values: 0-255 (8-bit)
Example: 256x256 = 65,536 pixels
```

**Color Image (RGB)**:
```
Shape: (Height, Width, Channels)
Channels: 3 (Red, Green, Blue)
Values: Each channel 0-255
Example: 256x256x3 = 196,608 values
```

### 1.3 Color Spaces

**RGB (Red-Green-Blue)**:
- Most common representation
- Each pixel: [R, G, B]
- Example: [255, 0, 0] = Pure Red

**HSV (Hue-Saturation-Value)**:
- Hue: Color type (0-180°)
- Saturation: Color intensity (0-255)
- Value: Brightness (0-255)
- **Advantage**: Better for color-based segmentation

**BGR (Blue-Green-Red)**:
- OpenCV's default format
- Same as RGB but reversed channel order

---

## 2. Dataset Analysis

### 2.1 Class Distribution

A balanced dataset is crucial for training effective models. Imbalanced datasets lead to biased predictions.

**Example of imbalanced dataset**:
```
apple_healthy:        1500 images (43%)
apple_scab:           800 images  (23%)
apple_black_rot:      700 images  (20%)
apple_cedar_rust:     500 images  (14%)
Total:                3500 images
```

**Problems caused**:
- Model biased toward majority class
- Poor performance on minority classes
- Lower overall accuracy

### 2.2 Statistical Analysis

Key metrics to compute:

**Mean images per class**:
```
μ = Σ(images_per_class) / num_classes
```

**Standard deviation**:
```
σ = sqrt(Σ(xi - μ)² / N)
```

**Coefficient of variation**:
```
CV = (σ / μ) × 100
```
- CV < 10%: Well balanced
- CV > 30%: Needs balancing

### 2.3 Visualization Techniques

**Pie Chart**:
- Shows relative proportions
- Good for seeing imbalance at a glance

**Bar Chart**:
- Shows absolute counts
- Easy to compare quantities

---

## 3. Data Augmentation

### 3.1 Definition

Data augmentation artificially increases dataset size by creating modified versions of existing images while preserving their labels.

### 3.2 Augmentation Techniques

#### 3.2.1 Geometric Transformations

**Horizontal Flip**:
```python
flipped = cv2.flip(image, 1)
```
- Mirror image along vertical axis
- Preserves disease characteristics
- Doubles dataset size instantly

**Rotation**:
```python
M = cv2.getRotationMatrix2D(center, angle, scale)
rotated = cv2.warpAffine(image, M, (width, height))
```
- Rotates image by specified angle
- Common angles: 90°, 180°, 270°, or random
- Disease independent of orientation

**Mathematical representation**:
```
[x']   [cos(θ)  -sin(θ)] [x]   [tx]
[y'] = [sin(θ)   cos(θ)] [y] + [ty]
```

**Affine Transformation (Skew)**:
```python
pts_src = np.float32([[0,0], [width,0], [0,height], [width,height]])
pts_dst = np.float32([[offset,0], [width-offset,0], [0,height], [width,height]])
M = cv2.getPerspectiveTransform(pts_src, pts_dst)
skewed = cv2.warpPerspective(image, M, (width, height))
```
- Simulates different viewing angles
- Preserves parallel lines

**Shear Transformation**:
```python
M = np.array([[1, shear_factor, 0],
              [0, 1, 0]], dtype=np.float32)
sheared = cv2.warpAffine(image, M, (width, height))
```

Mathematical form:
```
[x']   [1  k] [x]
[y'] = [0  1] [y]
```
where k is the shear factor.

**Crop**:
```python
cropped = image[y_start:y_end, x_start:x_end]
```
- Extracts sub-region of image
- Teaches model to recognize partial information
- Random crops add variability

#### 3.2.2 Photometric Transformations

**Gaussian Blur**:
```python
blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
```

Mathematical formula:
```
G(x,y) = (1 / 2πσ²) × exp(-(x² + y²) / 2σ²)
```
- Reduces noise and detail
- Simulates out-of-focus images
- Kernel size must be odd: (3,3), (5,5), (7,7)

**Brightness Adjustment**:
```python
adjusted = cv2.convertScaleAbs(image, alpha=1.0, beta=brightness)
```
- `beta > 0`: Increases brightness
- `beta < 0`: Decreases brightness
- Simulates different lighting conditions

**Contrast Adjustment**:
```python
adjusted = cv2.convertScaleAbs(image, alpha=contrast, beta=0)
```
- `alpha > 1`: Increases contrast
- `alpha < 1`: Decreases contrast

**Color Jitter**:
```python
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
hsv[:,:,0] = hsv[:,:,0] * hue_factor
hsv[:,:,1] = hsv[:,:,1] * saturation_factor
hsv[:,:,2] = hsv[:,:,2] * value_factor
augmented = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
```
- Randomly adjusts hue, saturation, brightness
- Robust to color variations

### 3.3 Balancing Strategy

**Algorithm**:
```python
# Pseudocode
max_class_size = max(count for each class)

for each_class:
    current_size = count_images(class)
    deficit = max_class_size - current_size
    
    while deficit > 0:
        original_image = random_choice(class_images)
        augmentation = random_choice(techniques)
        new_image = apply(augmentation, original_image)
        save(new_image)
        deficit -= 1
```

**Result**: All classes have equal representation, preventing model bias.

### 3.4 Augmentation Best Practices

1. **Preserve label validity**: Ensure transformations don't change the disease
2. **Realistic transformations**: Avoid unrealistic distortions
3. **Variety**: Use multiple techniques, not just one
4. **Balance**: Don't over-augment majority classes
5. **Validation set**: Never augment validation/test sets

---

## 4. Image Transformation Pipeline

### 4.1 Preprocessing Operations

#### 4.1.1 Gaussian Blur

**Purpose**: Noise reduction and smoothing.

**Kernel example** (3×3):
```
[1  2  1]
[2  4  2]  × (1/16)
[1  2  1]
```

**Application**:
```python
blurred = cv2.GaussianBlur(image, (5, 5), sigmaX=0)
```

#### 4.1.2 Color Space Conversion

**RGB to HSV**:
```python
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
```

**Advantages of HSV**:
- Hue channel isolates color information
- Better for color-based segmentation
- Less sensitive to lighting changes

#### 4.1.3 Masking & Segmentation

**Objective**: Isolate the leaf from the background.

**Color-based masking**:
```python
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define green range
lower_green = np.array([25, 40, 40])
upper_green = np.array([90, 255, 255])

# Create mask
mask = cv2.inRange(hsv, lower_green, upper_green)
```

**Morphological Operations**:

**Erosion** (removes small noise):
```python
kernel = np.ones((5,5), np.uint8)
eroded = cv2.erode(mask, kernel, iterations=1)
```

**Dilation** (fills small holes):
```python
dilated = cv2.dilate(mask, kernel, iterations=1)
```

**Opening** (erosion then dilation):
```python
opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
```
- Removes small objects while preserving shape

**Closing** (dilation then erosion):
```python
closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
```
- Fills small holes inside objects

### 4.2 Feature Extraction

#### 4.2.1 Edge Detection

**Canny Edge Detector**:
```python
edges = cv2.Canny(image, threshold1=50, threshold2=150)
```

**Algorithm steps**:
1. Gaussian smoothing
2. Gradient calculation (Sobel)
3. Non-maximum suppression
4. Double threshold
5. Edge tracking by hysteresis

**Sobel Operator**:
```python
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)  # Horizontal edges
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)  # Vertical edges
magnitude = np.sqrt(sobelx**2 + sobely**2)
```

#### 4.2.2 Contour Detection

**Finding contours**:
```python
contours, hierarchy = cv2.findContours(mask, 
                                       cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
```

**Retrieval modes**:
- `RETR_EXTERNAL`: Only outermost contours
- `RETR_LIST`: All contours, no hierarchy
- `RETR_TREE`: Full hierarchy

**Approximation methods**:
- `CHAIN_APPROX_NONE`: All boundary points
- `CHAIN_APPROX_SIMPLE`: Compresses contours (saves memory)

#### 4.2.3 Shape Analysis

**Contour properties**:

**Area**:
```python
area = cv2.contourArea(contour)
```

**Perimeter**:
```python
perimeter = cv2.arcLength(contour, closed=True)
```

**Bounding Rectangle**:
```python
x, y, w, h = cv2.boundingRect(contour)
aspect_ratio = w / h
```

**Minimum Enclosing Circle**:
```python
(x, y), radius = cv2.minEnclosingCircle(contour)
```

**Fitted Ellipse**:
```python
if len(contour) >= 5:
    ellipse = cv2.fitEllipse(contour)
    cv2.ellipse(image, ellipse, (0,255,0), 2)
```

**Circularity**:
```python
circularity = (4 * np.pi * area) / (perimeter ** 2)
```
- Circle: circularity ≈ 1.0
- Elongated: circularity < 0.5

#### 4.2.4 Color Histograms

**RGB Histogram**:
```python
for i, color in enumerate(['b', 'g', 'r']):
    hist = cv2.calcHist([image], [i], mask, [256], [0, 256])
    plt.plot(hist, color=color)
```

**HSV Analysis**:
```python
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
h_hist = cv2.calcHist([hsv], [0], mask, [180], [0, 180])  # Hue
s_hist = cv2.calcHist([hsv], [1], mask, [256], [0, 256])  # Saturation
v_hist = cv2.calcHist([hsv], [2], mask, [256], [0, 256])  # Value
```

**Disease indicators**:
- Healthy leaves: Hue peak around 60° (green)
- Black rot: Low saturation, low value (brown/black)
- Rust: Hue shift toward orange (20-40°)

#### 4.2.5 Texture Analysis

**Local Binary Patterns (LBP)**:
- Compares center pixel with neighbors
- Creates binary pattern
- Histogram of patterns = texture descriptor

**Gabor Filters**:
```python
# Create Gabor kernel
kernel = cv2.getGaborKernel(ksize=(21,21), 
                            sigma=5, 
                            theta=0, 
                            lambd=10, 
                            gamma=0.5)
filtered = cv2.filter2D(image, cv2.CV_8UC3, kernel)
```

Parameters:
- `theta`: Orientation (0°, 45°, 90°, 135°)
- `lambda`: Wavelength
- `sigma`: Bandwidth

**Gradient-based texture**:
```python
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
magnitude = np.sqrt(sobelx**2 + sobely**2)
texture_map = cv2.applyColorMap(np.uint8(magnitude), cv2.COLORMAP_JET)
```

---

## 5. Convolutional Neural Networks (CNN)

### 5.1 Architecture Overview

CNNs are specialized neural networks designed for processing grid-structured data (images).

**Basic structure**:
```
Input → [Conv → Activation → Pool] × N → Flatten → [Dense → Activation] × M → Output
```

### 5.2 Convolutional Layer

#### 5.2.1 Operation

A convolutional layer applies learned filters (kernels) to extract features.

**Mathematical definition**:
```
(f * g)[i,j] = ΣΣ f[m,n] × g[i-m, j-n]
               m n
```

**Discrete 2D convolution example**:

```
Input (5×5):                Kernel (3×3):
[1  2  3  0  1]             [1   0  -1]
[4  5  6  1  2]             [2   0  -2]
[7  8  9  2  3]             [1   0  -1]
[1  2  3  4  5]
[6  7  8  9  0]

Output (3×3):
Position (0,0):
(1×1 + 2×0 + 3×-1) + (4×2 + 5×0 + 6×-2) + (7×1 + 8×0 + 9×-1) = -8
```

#### 5.2.2 Key Parameters

**Number of filters**:
```python
Conv2D(filters=32, kernel_size=(3,3))
```
- Each filter learns a different feature
- More filters = more representational capacity
- Common progression: 32 → 64 → 128 → 256

**Kernel size**:
- `(3,3)`: Small, local features (most common)
- `(5,5)`: Medium features
- `(7,7)`: Large features
- Larger kernels = more parameters, slower training

**Stride**:
```python
Conv2D(..., strides=(2,2))
```
- `stride=1`: Move 1 pixel at a time
- `stride=2`: Downsample by 2x
- Larger stride = smaller output

**Output size formula**:
```
output_size = (input_size - kernel_size + 2×padding) / stride + 1
```

Example:
```
Input: 256×256
Kernel: 3×3
Padding: 0 (valid)
Stride: 1

Output: (256 - 3 + 0) / 1 + 1 = 254×254
```

**Padding**:
- `valid`: No padding (output shrinks)
- `same`: Pad to maintain size

```python
# Valid padding
Conv2D(32, (3,3), padding='valid')
Input: 256×256 → Output: 254×254

# Same padding
Conv2D(32, (3,3), padding='same')
Input: 256×256 → Output: 256×256
```

#### 5.2.3 Receptive Field

The receptive field is the region in the input that affects a particular neuron.

**Calculation**:
```
Layer 1: 3×3 kernel → RF = 3×3
Layer 2: 3×3 kernel → RF = 5×5
Layer 3: 3×3 kernel → RF = 7×7

Formula: RF_n = RF_(n-1) + (K-1) × Π(stride_i)
```

**With pooling**:
```
Conv(3×3) → Pool(2×2) → Conv(3×3)
RF: 3×3 → 6×6 → 10×10
```

Deeper networks can see larger context with fewer parameters.

#### 5.2.4 Parameter Count

```
Parameters = (kernel_h × kernel_w × input_channels + 1) × num_filters
                                                      ↑
                                                    bias
```

**Example**:
```python
# First layer
Conv2D(32, (3,3), input_shape=(256, 256, 3))
Params = (3 × 3 × 3 + 1) × 32 = 896

# Second layer
Conv2D(64, (3,3))  # Receives 32 channels from previous layer
Params = (3 × 3 × 32 + 1) × 64 = 18,496
```

### 5.3 Activation Functions

#### 5.3.1 ReLU (Rectified Linear Unit)

**Function**:
```python
def relu(x):
    return max(0, x)
```

**Mathematical form**:
```
f(x) = max(0, x) = {  x,  if x > 0
                   {  0,  if x ≤ 0
```

**Properties**:
- Non-linear
- Computationally efficient
- Sparse activation (many zeros)
- No vanishing gradient for x > 0

**Derivative**:
```
f'(x) = {  1,  if x > 0
        {  0,  if x ≤ 0
```

**Variants**:

**Leaky ReLU**:
```python
def leaky_relu(x, alpha=0.01):
    return max(alpha * x, x)
```
- Prevents "dying ReLU" problem
- `alpha` is small (0.01-0.1)

**ELU (Exponential Linear Unit)**:
```python
def elu(x, alpha=1.0):
    return x if x > 0 else alpha * (np.exp(x) - 1)
```
- Smooth for negative values
- Mean activation closer to zero

#### 5.3.2 Softmax

**Function**:
```
softmax(xi) = exp(xi) / Σ exp(xj)
                        j=1
```

**Properties**:
- Converts logits to probabilities
- Output sums to 1.0
- Used in final layer for classification

**Example**:
```python
logits = [2.0, 1.0, 0.1, 3.0]

# Step 1: Exponentiate
exp_logits = [7.39, 2.72, 1.11, 20.09]

# Step 2: Sum
sum_exp = 31.31

# Step 3: Normalize
probabilities = [0.236, 0.087, 0.035, 0.642]
```

### 5.4 Pooling Layers

#### 5.4.1 Max Pooling

**Operation**: Takes maximum value in each region.

```python
MaxPooling2D(pool_size=(2,2), strides=(2,2))
```

**Example**:
```
Input (4×4):
[1  3  2  4]
[5  6  7  8]
[2  1  0  3]
[4  2  1  2]

Max Pool (2×2):
Region 1: max(1,3,5,6) = 6
Region 2: max(2,4,7,8) = 8
Region 3: max(2,1,4,2) = 4
Region 4: max(0,3,1,2) = 3

Output (2×2):
[6  8]
[4  3]
```

**Purpose**:
- Downsampling: Reduces spatial dimensions
- Translation invariance: Feature detected regardless of exact position
- Reduces parameters and computation

#### 5.4.2 Average Pooling

**Operation**: Takes average of each region.

```python
AveragePooling2D(pool_size=(2,2))
```

**Example**:
```
Input (4×4):
[2  4  1  3]
[6  8  7  5]
[1  3  2  4]
[5  7  6  8]

Avg Pool (2×2):
Region 1: (2+4+6+8)/4 = 5
Region 2: (1+3+7+5)/4 = 4
Region 3: (1+3+5+7)/4 = 4
Region 4: (2+4+6+8)/4 = 5

Output (2×2):
[5  4]
[4  5]
```

#### 5.4.3 Global Average Pooling (GAP)

**Operation**: Averages each feature map to single value.

```python
GlobalAveragePooling2D()
```

**Example**:
```
Input: 7×7×512 (512 feature maps)

For each feature map:
  average all 49 values → single value

Output: 1×1×512 (512 values)
```

**Advantages**:
- Massive parameter reduction
- No overfitting from FC layers
- More interpretable (each value = presence of feature)

**Comparison**:
```
7×7×512 → Flatten → Dense(4)
  = 25,088 × 4 = 100,352 parameters

7×7×512 → GAP → Dense(4)
  = 512 × 4 = 2,048 parameters

50× reduction!
```

### 5.5 Fully Connected (Dense) Layers

#### 5.5.1 Operation

Every neuron connects to every neuron in previous layer.

```python
Dense(units=128, activation='relu')
```

**Mathematical form**:
```
output = activation(W × input + b)

Where:
  W = weight matrix (units × input_dim)
  b = bias vector (units)
```

**Example**:
```python
Input: [1, 2, 3, 4]
Dense(3)

Weights W (3×4):
[[0.5, -0.2,  0.3,  0.1],
 [0.1,  0.4, -0.1,  0.2],
 [-0.3,  0.2,  0.5, -0.1]]

Bias b (3):
[0.1, -0.2, 0.3]

Computation:
neuron_1 = 0.5×1 + (-0.2)×2 + 0.3×3 + 0.1×4 + 0.1 = 1.4
neuron_2 = 0.1×1 + 0.4×2 + (-0.1)×3 + 0.2×4 + (-0.2) = 0.9
neuron_3 = (-0.3)×1 + 0.2×2 + 0.5×3 + (-0.1)×4 + 0.3 = 1.0

Output: [1.4, 0.9, 1.0]
```

#### 5.5.2 Parameter Count

```
parameters = input_size × output_size + output_size
```

**Example**:
```python
Dense(512, input_shape=(6272,))
Params = 6272 × 512 + 512 = 3,211,776
```

**Problem**: Most parameters are in Dense layers!

**Solution**: Use fewer Dense layers, smaller sizes, or GAP.

### 5.6 Dropout

#### 5.6.1 Concept

Randomly "drops" (sets to zero) neurons during training.

```python
Dropout(rate=0.5)  # Drop 50%
```

**Training mode**:
```
Input:  [1.2, 3.4, 2.1, 0.8, 4.5, 1.9, 3.2, 2.7]
Mask:   [ 1,   0,   1,   0,   1,   1,   0,   1]
Output: [1.2,  0,  2.1,  0,  4.5, 1.9,  0,  2.7]
```

**Inference mode**:
```
Dropout is disabled, all neurons active
```

#### 5.6.2 Inverted Dropout

To maintain expected values, scale activations:

```python
# During training
output = input * mask / (1 - dropout_rate)

# Example with rate=0.5
Input:  [2, 4, 6, 8]
Mask:   [1, 0, 1, 1]
Output: [4, 0, 12, 16]  # Scaled by 2

# During inference
Output: [2, 4, 6, 8]  # No scaling needed
```

#### 5.6.3 Why It Works

**Prevents co-adaptation**: Neurons can't rely on specific other neurons.

**Ensemble effect**: Like training multiple models and averaging.

**Mathematical intuition**:
```
E[output with dropout] = E[output without dropout]
```

#### 5.6.4 Best Practices

```python
# Typical placement
Conv2D(64, (3,3), activation='relu')
# NO dropout after Conv

Flatten()
Dense(512, activation='relu')
Dropout(0.5)  # YES after Dense

Dense(256, activation='relu')
Dropout(0.3)  # YES, lower rate

Dense(4, activation='softmax')
# NO dropout in output layer
```

### 5.7 Batch Normalization

#### 5.7.1 Purpose

Normalizes activations to have mean≈0, std≈1.

```python
BatchNormalization()
```

#### 5.7.2 Operation

For each mini-batch:

```
# Step 1: Compute statistics
μ = mean(batch)
σ² = variance(batch)

# Step 2: Normalize
x_norm = (x - μ) / sqrt(σ² + ε)

# Step 3: Scale and shift (learnable)
y = γ × x_norm + β
```

**Example**:
```
Batch activations: [100, 150, 80, 200, 50]

μ = 116
σ² = 3064
σ = 55.35

Normalized: [-0.29, 0.61, -0.65, 1.52, -1.19]

If γ=2, β=1:
Output: [-0.58, 1.22, -1.30, 3.04, -2.38] + 1
      = [0.42, 2.22, -0.30, 4.04, -1.38]
```

#### 5.7.3 Benefits

1. **Faster training**: Can use higher learning rates
2. **Reduces internal covariate shift**
3. **Regularization effect**: Slight noise from batch statistics
4. **Less sensitive to initialization**

#### 5.7.4 Placement

```python
# After Conv, before activation
Conv2D(32, (3,3))
BatchNormalization()
Activation('relu')

# OR after activation (less common)
Conv2D(32, (3,3), activation='relu')
BatchNormalization()
```

### 5.8 Complete Architecture Example

```python
model = Sequential([
    # Block 1
    Conv2D(32, (3,3), padding='same', input_shape=(256,256,3)),
    BatchNormalization(),
    Activation('relu'),
    Conv2D(32, (3,3), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D((2,2)),
    
    # Block 2
    Conv2D(64, (3,3), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    Conv2D(64, (3,3), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D((2,2)),
    
    # Block 3
    Conv2D(128, (3,3), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    Conv2D(128, (3,3), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D((2,2)),
    
    # Block 4
    Conv2D(256, (3,3), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    Conv2D(256, (3,3), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D((2,2)),
    
    # Classifier
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(4, activation='softmax')
])
```

**Data flow**:
```
Input: 256×256×3
  ↓ Conv+BN+ReLU+Conv+BN+ReLU+Pool
128×128×32
  ↓ Conv+BN+ReLU+Conv+BN+ReLU+Pool
64×64×64
  ↓ Conv+BN+ReLU+Conv+BN+ReLU+Pool
32×32×128
  ↓ Conv+BN+ReLU+Conv+BN+ReLU+Pool
16×16×256
  ↓ Flatten
65,536
  ↓ Dense+Dropout
512
  ↓ Dense+Dropout
256
  ↓ Dense+Softmax
4 (probabilities)
```

---

## 6. Training Process

### 6.1 Dataset Split

#### 6.1.1 Train/Validation Split

**Common ratios**:
```
Training:   80% (model learns from this)
Validation: 20% (evaluate performance)
```

**Stratified split**: Maintains class proportions.

```python
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    images, labels,
    test_size=0.2,
    stratify=labels,  # Important!
    random_state=42
)
```

**Example**:
```
Total dataset:
  healthy: 1000 images
  scab: 800 images
  black_rot: 600 images
  cedar_rust: 600 images

After stratified split (80/20):
Training:
  healthy: 800
  scab: 640
  black_rot: 480
  cedar_rust: 480

Validation:
  healthy: 200
  scab: 160
  black_rot: 120
  cedar_rust: 120
```

#### 6.1.2 Why Not Test on Training Data?

**Problem**: Overfitting detection impossible.

```
Training accuracy: 99%
Validation accuracy: ???

Without validation, you don't know if the model generalizes!
```

### 6.2 Loss Functions

#### 6.2.1 Categorical Cross-Entropy

**Formula**:
```
L = -Σ yi × log(ŷi)
    i=1
```

Where:
- `yi`: True label (one-hot encoded)
- `ŷi`: Predicted probability

**Example**:
```python
True label: [0, 1, 0, 0]  # Class 1 (scab)
Prediction: [0.1, 0.7, 0.1, 0.1]

Loss = -(0×log(0.1) + 1×log(0.7) + 0×log(0.1) + 0×log(0.1))
     = -log(0.7)
     = 0.357
```

**Good prediction**:
```
Prediction: [0.05, 0.90, 0.03, 0.02]
Loss = -log(0.90) = 0.105  (low loss ✓)
```

**Bad prediction**:
```
Prediction: [0.40, 0.20, 0.30, 0.10]
Loss = -log(0.20) = 1.609  (high loss ✗)
```

#### 6.2.2 Why Cross-Entropy?

**Penalizes confident wrong predictions**:
```
True: [0, 1, 0, 0]

Prediction A: [0.1, 0.6, 0.2, 0.1]  → Loss = 0.511
Prediction B: [0.8, 0.1, 0.05, 0.05] → Loss = 2.303

Prediction B is confident AND wrong → Higher penalty
```

### 6.3 Optimizers

#### 6.3.1 Gradient Descent

**Basic idea**: Update weights in direction of negative gradient.

```
w_new = w_old - learning_rate × ∂L/∂w
```

**Problem**: Too slow, can get stuck in local minima.

#### 6.3.2 Adam (Adaptive Moment Estimation)

**Most commonly used optimizer**.

```python
optimizer = Adam(learning_rate=0.001)
```

**Key features**:
- Adapts learning rate per parameter
- Combines momentum and RMSprop
- Works well in most cases

**Update rule**:
```
m_t = β₁ × m_{t-1} + (1 - β₁) × g_t        # Momentum
v_t = β₂ × v_{t-1} + (1 - β₂) × g_t²       # RMSprop
m̂_t = m_t / (1 - β₁ᵗ)                      # Bias correction
v̂_t = v_t / (1 - β₂ᵗ)
w_t = w_{t-1} - α × m̂_t / (√v̂_t + ε)

Where:
  β₁ = 0.9 (momentum decay)
  β₂ = 0.999 (RMSprop decay)
  α = learning rate
  ε = 1e-8 (numerical stability)
```

#### 6.3.3 Other Optimizers

**SGD (Stochastic Gradient Descent)**:
```python
optimizer = SGD(learning_rate=0.01, momentum=0.9)
```
- Simple, reliable
- Requires careful learning rate tuning

**RMSprop**:
```python
optimizer = RMSprop(learning_rate=0.001)
```
- Good for RNNs
- Adapts learning rate based on recent gradients

### 6.4 Learning Rate

#### 6.4.1 Impact

**Too high**:
```
Loss oscillates or diverges
Epoch 1: Loss = 2.5
Epoch 2: Loss = 3.1
Epoch 3: Loss = 5.2  ✗
```

**Too low**:
```
Learning extremely slow
Epoch 1: Loss = 2.500
Epoch 2: Loss = 2.498
Epoch 3: Loss = 2.496  (barely moving)
```

**Just right**:
```
Steady decrease
Epoch 1: Loss = 2.50
Epoch 2: Loss = 2.15
Epoch 3: Loss = 1.85  ✓
```

#### 6.4.2 Learning Rate Schedules

**Step Decay**:
```python
# Reduce LR every N epochs
lr_schedule = StepDecay(initial_lr=0.001, drop=0.5, epochs_drop=10)

Epoch 0-9:   lr = 0.001
Epoch 10-19: lr = 0.0005
Epoch 20-29: lr = 0.00025
```

**ReduceLROnPlateau**:
```python
# Reduce when validation loss stops improving
lr_schedule = ReduceLROnPlateau(monitor='val_loss', 
                                 factor=0.5, 
                                 patience=5)

If val_loss doesn't improve for 5 epochs → lr = lr × 0.5
```

**Cosine Annealing**:
```
lr = lr_min + 0.5 × (lr_max - lr_min) × (1 + cos(π × epoch / total_epochs))
```
- Smooth decay
- Popular in modern training

### 6.5 Training Loop

#### 6.5.1 Epoch vs Batch vs Iteration

**Definitions**:
```
Batch: Small group of samples (e.g., 32 images)
Iteration: One forward + backward pass on one batch
Epoch: One complete pass through entire dataset
```

**Example**:
```
Dataset: 3200 images
Batch size: 32
Iterations per epoch: 3200 / 32 = 100
Total epochs: 50
Total iterations: 50 × 100 = 5000
```

#### 6.5.2 Forward Pass

```python
# Pseudocode
for epoch in range(num_epochs):
    for batch in train_loader:
        images, labels = batch
        
        # 1. Forward pass
        predictions = model(images)
        # images: (32, 256, 256, 3)
        # predictions: (32, 4)
```

**What happens**:
```
Input batch (32 images) 
  → Conv layers extract features
  → Pooling reduces dimensions
  → Dense layers combine features
  → Softmax outputs probabilities
Output: (32, 4) tensor of probabilities
```

#### 6.5.3 Loss Calculation

```python
        # 2. Calculate loss
        loss = criterion(predictions, labels)
        # predictions: (32, 4)
        # labels: (32,) or (32, 4) one-hot
        # loss: scalar value
```

**Batch loss**: Average loss over all samples in batch.

```
loss_batch = (loss_sample1 + loss_sample2 + ... + loss_sample32) / 32
```

#### 6.5.4 Backward Pass (Backpropagation)

```python
        # 3. Backward pass
        optimizer.zero_grad()  # Reset gradients
        loss.backward()        # Compute gradients
        optimizer.step()       # Update weights
```

**What backward() does**:
```
1. Compute ∂L/∂w for every parameter w
2. Store gradients in w.grad
```

**Chain rule example**:
```
L = loss(softmax(dense(flatten(pool(conv(x))))))

∂L/∂w_conv = ∂L/∂softmax × ∂softmax/∂dense × ∂dense/∂flatten × 
             ∂flatten/∂pool × ∂pool/∂conv × ∂conv/∂w_conv
```

#### 6.5.5 Weight Update

```python
        optimizer.step()  # Updates all parameters
```

**What happens** (simplified SGD):
```
for each parameter w:
    w = w - learning_rate × w.grad
```

**Example**:
```
Before:
  w = 0.5
  grad = -0.2
  lr = 0.01

After:
  w = 0.5 - 0.01 × (-0.2) = 0.502
```

#### 6.5.6 Complete Training Loop

```python
for epoch in range(epochs):
    # Training phase
    model.train()  # Enable dropout, batch norm training mode
    train_loss = 0
    train_correct = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        # Move to GPU if available
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        predictions = model(images)
        loss = criterion(predictions, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track metrics
        train_loss += loss.item()
        _, predicted = torch.max(predictions, 1)
        train_correct += (predicted == labels).sum().item()
    
    # Calculate epoch metrics
    avg_train_loss = train_loss / len(train_loader)
    train_accuracy = 100 * train_correct / len(train_dataset)
    
    # Validation phase
    model.eval()  # Disable dropout, batch norm eval mode
    val_loss = 0
    val_correct = 0
    
    with torch.no_grad():  # No gradient computation
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            predictions = model(images)
            loss = criterion(predictions, labels)
            
            val_loss += loss.item()
            _, predicted = torch.max(predictions, 1)
            val_correct += (predicted == labels).sum().item()
    
    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = 100 * val_correct / len(val_dataset)
    
    # Print progress
    print(f'Epoch {epoch+1}/{epochs}')
    print(f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%')
    print(f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')
    
    # Learning rate scheduling
    scheduler.step(avg_val_loss)
```

### 6.6 Overfitting Prevention

#### 6.6.1 Detecting Overfitting

**Signs**:
```
Epoch 10: Train 85%, Val 83%  ✓ Good
Epoch 20: Train 92%, Val 90%  ✓ Good
Epoch 30: Train 97%, Val 91%  ⚠ Warning
Epoch 40: Train 99%, Val 89%  ✗ Overfitting!

Train accuracy keeps increasing
Val accuracy plateaus or decreases
```

#### 6.6.2 Techniques

**1. Data Augmentation**
```python
# Already covered in Section 3
transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(0.2, 0.2, 0.2),
])
```

**2. Dropout**
```python
Dense(512, activation='relu')
Dropout(0.5)
```

**3. L2 Regularization (Weight Decay)**
```python
optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
```

**Loss with L2**:
```
Total_Loss = Cross_Entropy_Loss + λ × Σ(w²)

Penalizes large weights
λ = weight_decay parameter (1e-4 to 1e-3)
```

**4. Early Stopping**
```python
best_val_loss = float('inf')
patience = 10
counter = 0

for epoch in range(max_epochs):
    train()
    val_loss = validate()
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_model()
        counter = 0
    else:
        counter += 1
    
    if counter >= patience:
        print("Early stopping triggered")
        load_best_model()
        break
```

**5. Reduce Model Complexity**
```python
# Option 1: Fewer layers
# Option 2: Fewer filters per layer
# Option 3: Smaller dense layers

# Before
Conv2D(256, ...)
Dense(1024)

# After
Conv2D(128, ...)
Dense(512)
```

### 6.7 Batch Size Impact

#### 6.7.1 Small Batch (8-16)

**Advantages**:
- More frequent updates
- Better generalization (noise helps escape local minima)
- Less memory usage

**Disadvantages**:
- Slow training (more iterations)
- Noisy gradients
- Underutilizes GPU

#### 6.7.2 Large Batch (128-256)

**Advantages**:
- Faster training (fewer iterations)
- Stable gradients
- Better GPU utilization

**Disadvantages**:
- More memory usage
- Worse generalization
- Can get stuck in sharp minima

#### 6.7.3 Typical Choice

```python
batch_size = 32  # Good balance for most cases

# Adjust based on:
# - GPU memory (larger = more memory)
# - Dataset size (small dataset → smaller batch)
# - Model size (large model → smaller batch)
```

### 6.8 Training Tips

**1. Start with a baseline**:
```python
# Simple model, train for 10 epochs
# Check if it learns at all (accuracy > random)
# Random baseline for 4 classes: 25%
```

**2. Monitor training curves**:
```python
# Plot loss and accuracy over epochs
# Should see smooth decrease in loss
# Should see increase in accuracy
```

**3. Check for bugs**:
```python
# Sanity check: Overfit on 1 batch
small_batch = next(iter(train_loader))
for i in range(100):
    loss = train_on_batch(small_batch)
    
# Should reach ~0 loss if model can learn
```

**4. Gradual complexity increase**:
```
Start: Simple model, no augmentation
→ Add augmentation
→ Add dropout
→ Add more layers
→ Tune hyperparameters
```

---

## 7. Model Evaluation

### 7.1 Metrics

#### 7.1.1 Accuracy

**Definition**:
```
Accuracy = (Correct Predictions) / (Total Predictions)
```

**Example**:
```
100 validation images
92 predicted correctly
Accuracy = 92 / 100 = 92%
```

**Limitation**: Not good for imbalanced datasets.

```
Dataset: 95 healthy, 5 diseased
Model predicts "healthy" for everything
Accuracy = 95% (misleading!)
```

#### 7.1.2 Confusion Matrix

**Structure**:
```
                    Predicted
                 H    S    B    C
Actual   H     [90   5    3    2]   = 100
         S     [8   75    5    2]   = 90
         B     [3    7   80   10]   = 100
         C     [2    3    5   90]   = 100
```

**Reading**:
- Diagonal: Correct predictions
- Off-diagonal: Misclassifications

**Example**:
```
Row S, Column B = 5
Meaning: 5 scab images predicted as black_rot
```

#### 7.1.3 Precision & Recall

**Per-class metrics**:

**Precision**: Of all predictions for class X, how many were correct?
```
Precision_X = TP_X / (TP_X + FP_X)

Example for "scab":
Predicted 90 as scab
75 actually were scab
Precision = 75 / 90 = 0.833 = 83.3%
```

**Recall** (Sensitivity): Of all actual class X, how many did we find?
```
Recall_X = TP_X / (TP_X + FN_X)

Example for "scab":
90 actual scab images
Correctly identified 75
Recall = 75 / 90 = 0.833 = 83.3%
```

#### 7.1.4 F1-Score

**Harmonic mean of Precision and Recall**:
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)

Example:
Precision = 0.833
Recall = 0.833
F1 = 2 × (0.833 × 0.833) / (0.833 + 0.833) = 0.833
```

**Why harmonic mean?** Penalizes extreme values.

```
Case 1: Precision = 1.0, Recall = 0.1
Arithmetic mean = 0.55 (misleading)
Harmonic (F1) = 0.18 (better reflects poor performance)
```

#### 7.1.5 Macro vs Micro Averaging

**Macro Average**: Average of per-class metrics
```
Macro_Precision = (Prec_H + Prec_S + Prec_B + Prec_C) / 4
Treats all classes equally
```

**Micro Average**: Calculate from total TP, FP, FN
```
Micro_Precision = Σ TP / (Σ TP + Σ FP)
Weighted by class frequency
```

### 7.2 Classification Report

**Example output**:
```
              precision  recall  f1-score  support
     healthy       0.90    0.90      0.90      100
        scab       0.83    0.83      0.83       90
   black_rot       0.80    0.86      0.83      100
 cedar_rust       0.90    0.87      0.88      100

    accuracy                        0.86      390
   macro avg       0.86    0.87      0.86      390
weighted avg       0.86    0.86      0.86      390
```

### 7.3 ROC Curve & AUC

**ROC (Receiver Operating Characteristic)**:
- Plots True Positive Rate vs False Positive Rate
- Different thresholds for classification

**AUC (Area Under Curve)**:
```
AUC = 1.0: Perfect classifier
AUC = 0.5: Random classifier
AUC > 0.9: Excellent
AUC 0.8-0.9: Good
AUC < 0.7: Poor
```

---

## 8. Prediction & Inference

### 8.1 Model Loading

```python
# Load trained model
model = load_model('plant_disease_model.h5')

# Or for PyTorch
model = PlantCNN(num_classes=4)
checkpoint = torch.load('model.pth')
model.load_state_dict(checkpoint['state_dict'])
model.eval()  # Set to evaluation mode
```

### 8.2 Image Preprocessing

**CRITICAL**: Must match training preprocessing exactly.

```python
def preprocess_image(image_path):
    # 1. Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 2. Resize (same as training)
    image = cv2.resize(image, (256, 256))
    
    # 3. Normalize (same as training)
    image = image.astype(np.float32) / 255.0
    
    # 4. Apply same normalization as training
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image = (image - mean) / std
    
    # 5. Add batch dimension
    image = np.expand_dims(image, axis=0)
    # Shape: (1, 256, 256, 3)
    
    return image
```

**Common mistakes**:
```
✗ Different resize dimensions
✗ Forgetting normalization
✗ Wrong normalization values
✗ Missing batch dimension
✗ Wrong color space (BGR vs RGB)
```

### 8.3 Making Predictions

```python
def predict(model, image_path, class_names):
    # Preprocess
    image = preprocess_image(image_path)
    
    # Predict
    predictions = model.predict(image)
    # Shape: (1, 4)
    # Example: [[0.05, 0.82, 0.10, 0.03]]
    
    # Get predicted class
    predicted_idx = np.argmax(predictions[0])
    predicted_class = class_names[predicted_idx]
    confidence = predictions[0][predicted_idx] * 100
    
    return predicted_class, confidence, predictions[0]
```

### 8.4 Interpreting Results

**Confidence thresholds**:
```
confidence >= 90%: Very confident
confidence 70-90%: Confident
confidence 50-70%: Uncertain
confidence < 50%: Very uncertain (near random)
```

**Example**:
```python
predictions = [0.05, 0.82, 0.10, 0.03]
classes = ['healthy', 'scab', 'black_rot', 'cedar_rust']

Output:
Predicted: scab
Confidence: 82%
Interpretation: Model is confident this is scab
```

**Low confidence example**:
```python
predictions = [0.30, 0.28, 0.25, 0.17]

Output:
Predicted: healthy
Confidence: 30%
Interpretation: Model is very uncertain
Action: Manual inspection recommended
```

### 8.5 Batch Prediction

```python
def predict_batch(model, image_paths):
    images = []
    for path in image_paths:
        img = preprocess_image(path)
        images.append(img)
    
    # Stack into single batch
    batch = np.vstack(images)
    # Shape: (N, 256, 256, 3)
    
    # Predict all at once
    predictions = model.predict(batch)
    # Shape: (N, 4)
    
    return predictions
```

**Advantage**: Much faster than one-by-one.

```
10 images one-by-one: 10 × 0.1s = 1.0s
10 images as batch: 0.2s
5× speedup!
```

### 8.6 Error Analysis

**Analyzing mistakes**:

```python
# Get all wrong predictions
errors = []
for image, true_label in validation_set:
    predicted = model.predict(image)
    if predicted != true_label:
        errors.append((image, true_label, predicted))

# Analyze patterns
# - Which classes are confused?
# - What do misclassified images have in common?
# - Low quality images?
# - Ambiguous cases?
```

**Common issues**:
```
1. Similar diseases confused
   (e.g., scab vs black_rot)
   → Need more training data
   → Better feature extraction

2. Low quality images
   → Add data augmentation for blur, noise
   → Improve image quality

3. Edge cases (early stage disease)
   → Collect more diverse data
   → Human expert might also struggle
```

### 8.7 Model Deployment Considerations

**1. Model size**:
```
Full model: 50 MB
Quantized: 12 MB (4× smaller)
Pruned: 8 MB (6× smaller)
```

**2. Inference speed**:
```
CPU: 100-200 ms per image
GPU: 10-20 ms per image
Mobile (quantized): 50-100 ms
```

**3. Memory requirements**:
```
Model weights: ~50 MB
Intermediate activations: ~100-200 MB
Total: ~150-250 MB RAM
```

**4. Production pipeline**:
```
User uploads image
  → Validate (format, size)
  → Preprocess (resize, normalize)
  → Predict
  → Post-process (threshold, format)
  → Return result
  → Log prediction (for monitoring)
```

---

## 9. Key Takeaways

### 9.1 Essential Concepts

1. **CNNs extract hierarchical features**: Early layers detect edges, later layers detect complex patterns

2. **Data quality > Model complexity**: Good, balanced data is more important than sophisticated architecture

3. **Regularization is crucial**: Dropout, data augmentation, and early stopping prevent overfitting

4. **Validation set is sacred**: Never train on validation data, never peek during development

5. **Preprocessing consistency**: Inference must use identical preprocessing as training

### 9.2 Common Pitfalls

```
✗ Training on imbalanced data
✗ Not using data augmentation
✗ Too complex model for small dataset
✗ Not monitoring validation metrics
✗ Inconsistent preprocessing
✗ No error analysis
✗ Ignoring confidence scores
```

### 9.3 Success Checklist

```
✓ Balanced dataset (or weighted loss)
✓ Train/val split (80/20)
✓ Data augmentation
✓ Batch normalization
✓ Dropout (0.3-0.5)
✓ Learning rate schedule
✓ Early stopping
✓ >90% validation accuracy
✓ Confusion matrix analysis
✓ Error cases reviewed
```

---

## 10. Mathematical Reference

### 10.1 Key Formulas

**Convolution**:
```
(f * g)[i,j] = ΣΣ f[m,n] × g[i-m, j-n]
```

**Cross-Entropy Loss**:
```
L = -Σ yi × log(ŷi)
```

**Softmax**:
```
σ(zi) = exp(zi) / Σ exp(zj)
```

**Batch Normalization**:
```
x̂ = (x - μ) / √(σ² + ε)
y = γ × x̂ + β
```

**Accuracy**:
```
Acc = TP / (TP + TN + FP + FN)
```

**Precision**:
```
Prec = TP / (TP + FP)
```

**Recall**:
```
Rec = TP / (TP + FN)
```

**F1-Score**:
```
F1 = 2 × (Prec × Rec) / (Prec + Rec)
```

---

## 11. Resources for Further Study

### 11.1 Papers
- **LeNet-5** (LeCun et al., 1998): First successful CNN
- **AlexNet** (Krizhevsky et al., 2012): Deep learning revolution
- **VGGNet** (Simonyan & Zisserman, 2014): Deeper networks
- **ResNet** (He et al., 2015): Residual connections
- **Batch Normalization** (Ioffe & Szegedy, 2015)

### 11.2 Datasets
- **Plant Village**: 54,000+ images, 38 classes
- **PlantDoc**: 2,500+ images, 13 classes
- **Kaggle Plant Pathology**: Competition datasets

### 11.3 Tools & Libraries
- **TensorFlow/Keras**: High-level deep learning
- **PyTorch**: Research-oriented framework
- **OpenCV**: Image processing
- **PlantCV**: Plant phenotyping
- **scikit-learn**: Machine learning utilities

---

*End of Theory Document*