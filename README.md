# 6th Place Solution - Yandex Cup 2024 ML: Self-Driving Cars 🚗

## Problem Overview 🎯

The challenge focuses on predicting self-driving vehicle trajectories using various input data sources. 

### Input Data
- **Vehicle Metadata**
  - Vehicle ID
  - Model specifications
  - Tire type and conditions
- **Localization Data**
  - Real-time coordinates
  - Vehicle orientation
- **Control Commands**
  - Acceleration inputs
  - Steering parameters

### Task Description
- Given: First 5 seconds of trajectory + 20 seconds of control commands
- Goal: Predict vehicle positions at specific timestamps (5-20 seconds)
- Evaluation: Accuracy of predicted positions vs. actual trajectories

## Solution Architecture 🏗️

The solution employs a two-stage prediction approach:
1. **TSMixer** (Time Series Model)
   - Primary prediction of core vehicle parameters
   - Focus on total velocity and yaw rate estimation

2. **CatBoost**
   - Refinement of initial predictions
   - Enhanced accuracy through gradient boosting

## Model Training Pipeline 🔄

### 1. TSMixer Data Preparation
```bash
pdm run python -m src.prepare_data_before_tsmixer
```
*Preprocesses and structures data for TSMixer training*

### 2. TSMixer Model Training
```bash
pdm run python -m src.train_tsmixer
```
*Trains the TSMixer model on prepared dataset*

### 3. TSMixer Inference
```bash
# Validation set inference
pdm run python -m src.inference_tsmixer -t val

# Test set inference
pdm run python -m src.inference_tsmixer -t test
```
*Generates predictions using trained TSMixer model*

### 4. CatBoost Data Preparation
```bash
pdm run python -m src.prepare_data_before_catboost
```
*Formats and prepares data for CatBoost training phase*

### 5. CatBoost Model Training
```bash
pdm run python -m src.train_catboost
```
*Trains the CatBoost model for prediction refinement*

### 6. Final Inference
```bash
# Validation set inference
pdm run python -m src.inference_catboost -t val

# Test set inference and submission generation
pdm run python -m src.inference_catboost -t test
```
*Produces final predictions and generates submission file*