# SHL Grammar Scoring Engine (Research-Grade)

## Overview
This project implements an end-to-end Grammar Scoring Engine designed to predict continuous grammar scores (0-5) from spoken audio samples. The solution leverages state-of-the-art self-supervised speech models to capture both acoustic and linguistic cues, providing a robust "research-grade" assessment of spoken grammar.

## Methodology & Approach

Our approach moves beyond traditional signal processing (e.g., MFCCs) by adopting a **Multi-Modal Feature Fusion** strategy. We hypothesize that "grammar" in speech is a function of both *what* is said (syntax/vocabulary) and *how* it is said (fluency/prosody).

### 1. Acoustic Features: Wav2Vec 2.0
We utilize **`facebook/wav2vec2-base`**, a Transformer-based model pretrained on 960 hours of unlabeled speech.
*   **Reasoning**: Wav2Vec2 basic embeddings implicitly capture phonetic structure, hesitation, pauses, and speech rhythm. These are strong indicators of fluency and grammatical control.
*   **Implementation**: We extract the last hidden state and apply **Mean and Standard Deviation Pooling** over the time dimension to create a fixed-size feature vector (1536 dimensions) for each audio file.

### 2. Linguistic Features: OpenAI Whisper
We utilize **`openai/whisper-tiny`** for Automatic Speech Recognition (ASR).
*   **Reasoning**: To judge grammar, we need access to the textual content. Whisper provides robust transcription even for diverse accents.
*   **Derived Metrics**:
    *   **Word Count**: Proxy for sentence complexity and length.
    *   **Speech Rate**: (Words / Duration). Indicators of hesitancy or struggle with language formulation.

### 3. Regression Head: Ensemble Learning
We fuse the acoustic and linguistic features and train ensemble regressors: **Random Forest** and **XGBoost**.
*   **Reasoning**:
    *   **High Dimensionality**: Tree-based models handle high-dimensional embedding spaces well without extensive dimensionality reduction (like PCA).
    *   **Non-Linearity**: They capture complex non-linear relationships between latent speech features and human-assigned scores.
    *   **Robustness**: Ensembling reduces the variance associated with single-model predictions.

## Evaluation
The model is evaluated using:
*   **RMSE (Root Mean Squared Error)**: To measure the average deviation from the ground truth.
*   **Pearson Correlation (r)**: To measure how well the model ranks speakers relative to each other.

## Directory Structure
```
.
├── SHL_Grammar_Scoring_Engine.ipynb  # Main Research Notebook
├── requirements.txt                  # Python Dependencies
├── submission.csv                    # Final Test Predictions
├── dataset/                          # Audio and CSV data
└── README.md                         # Project Documentation
```

## How to Run
1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Run Notebook**:
    Open `SHL_Grammar_Scoring_Engine.ipynb` and execute all cells. This will:
    *   Extract features from the dataset.
    *   Train the models.
    *   Generate the `submission.csv` file.
