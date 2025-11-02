<<<<<<< HEAD
# TC5044.10-Equipo-8
Energy Efficiency Analysis - Machine Learning Pipeline

## Overview
This project implements a complete machine learning pipeline for analyzing energy efficiency in buildings. It now leverages scikit-learn Pipelines to automate preprocessing, model training, and evaluation with Linear Regression, Random Forest, and Gradient Boosting regressors.

## Setup Instructions

### 1. Create Virtual Environment
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download Dataset
1. Download the Energy Efficiency dataset from: https://experiencia21.tec.mx/courses/624720/pages/datasets-por-equipos (team 8)
2. Extract the CSV file and rename it to `energy_efficiency_modified.csv`
3. Place it in the `src/data/` folder

### 4. Run the Pipeline
```bash
cd src
python main.py
```

## Project Structure
```
src/
├── main.py                    # Main execution pipeline
├── data/                      # Data directory
│   └── energy_efficiency_modified.csv
├── handlers/
│   ├── data_loader.py         # Data loading and saving
│   ├── data_preprocessor.py   # Data cleaning and preprocessing
│   ├── model_trainer.py       # Pipeline creation, training and validation
│   ├── model_evaluator.py     # Pipeline evaluation on the hold-out set
│   └── visual_eda.py          # Exploratory data analysis
```

## Features
- **Data Loading**: CSV loading with consistent column renaming
- **Data Preprocessing**: Type coercion, robust outlier handling, and automatic imputation/scaling encoded inside scikit-learn Pipelines
- **Visual EDA**: Histograms, boxplots, correlation heatmaps
- **Model Training**: Linear Regression, Random Forest, and Gradient Boosting, all trained through reusable Pipelines
- **Model Evaluation**: Hold-out metrics (R², RMSE, MAE) plus 5-fold cross-validation summaries
- **Version Control**: DVC integration for data versioning

## Models
- **Linear Regression**: Baseline multi-output regressor
- **Random Forest**: Ensemble method with 600 estimators
- **Gradient Boosting**: Gradient-boosted trees with tuned learning rate and depth

## Target Variables
- `heating_load`: Building heating energy requirements
- `cooling_load`: Building cooling energy requirements
=======
# TC5044-MLOps-Equipo-8
TC5044 MLOps-Equipo-8
>>>>>>> origin/main
