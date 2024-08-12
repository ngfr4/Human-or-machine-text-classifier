# Text Classification: Machine-Generated vs. Human-Written

## Overview

This repository contains a project aimed at classifying text as either machine-generated or human-written. We utilize both traditional machine learning and advanced deep learning techniques to achieve high accuracy. The dataset includes text samples in both English and Spanish.

## Dataset

- **English Dataset**: `subtask1_train_en.tsv`
- **Spanish Dataset**: `subtask1_train_es.tsv`

The dataset features labeled text samples used for both training and evaluation purposes.

## Methodology

### Data Preprocessing

1. **Loading Data**
   - Used `pandas` to load and merge TSV files.

2. **Cleaning and Preparation**
   - Removed extraneous columns.
   - Performed text cleaning including lowercasing and punctuation removal.
   - Applied tokenization, stopword removal, stemming, and lemmatization.

3. **Feature Engineering**
   - Extracted features such as word count, character count, and unique word count.
   - Implemented TF-IDF vectorization and Word2Vec embeddings.

### Model Training

#### Traditional Machine Learning Models

- **Logistic Regression**: Trained using TF-IDF and Word2Vec features.
- **Naive Bayes**: Applied with TF-IDF features.
- **Evaluation**: Metrics include classification report, confusion matrix, and ROC-AUC score.

#### Deep Learning Models

- **BERT-based Model**: Fine-tuned Roberta for sequence classification.
- **Tokenization**: Used `RobertaTokenizer` for text encoding.
- **Training**: Implemented a training loop with performance evaluation.

### Model Interpretation

- **LIME**: Employed LIME to interpret model predictions and visualize feature importance.

## Getting Started

1. **Install Dependencies**

2. **Preprocess Data**

Execute `preprocess_data.py` to clean and prepare the dataset.

3. **Train Models**

- Run `train_ml_models.py` for traditional machine learning models.
- Run `train_dl_models.py` for deep learning models.

4.  **Evaluate Models**

Use `evaluate_models.py` to assess the performance of the trained models.


5.  **Interpret Results**

Run `interpret_models.py` to generate LIME explanations and visualize feature importance.

### Contributing
We welcome contributions to improve the models and methodology. Please fork the repository and submit a pull request with your proposed changes.
**Contact**
For questions or suggestions, please open an issue on the GitHub repository or contact us.

Thank you for your interest in this project!

