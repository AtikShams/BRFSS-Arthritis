# Explainable Machine Learning for Arthritis Prediction

This repository provides the official implementation, configuration files, and evaluation framework for the study titled:

"Explainable Machine Learning Revealing the Impact of Mental and Physical Health on Arthritis"

## Overview

This study presents a comprehensive evaluation of eleven predictive models, combining traditional machine learning methods, such as XGBoost, LightGBM, and stacked ensembles, with deep learning approaches, including attention-based artificial neural networks.

The models are trained and validated on large-scale national survey datasets, specifically BRFSS and NHIS, to predict arthritis risk. To improve data quality and model performance, the framework incorporates generative adversarial networks (GANs) for missing data imputation and resampling techniques, such as random under-sampling (RUS), to address class imbalance.

## Installation and Setup

To ensure reproducibility of the results, install the required dependencies using the provided requirements file:

```bash
# Clone the repository
git clone https://github.com/AtikShams/BRFSS-Arthritis-.git
cd BRFSS-Arthritis-

# Install dependencies
pip install -r requirements.txt
```
