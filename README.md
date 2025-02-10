# Spotify Song Genre Classification Using Data Mining Techniques

This project explores the automatic classification of **Spotify songs** by genre using **data mining** and **machine learning** techniques. We utilized a dataset of **114,000 tracks** across **114 genres** and experimented with various models and preprocessing strategies to improve classification accuracy.

---

## Project Overview
Music genre classification is crucial for **music discovery** and **recommendation systems**. Given the challenges of manually assigning genres, this project aims to develop a machine learning pipeline to automate the process.

Key steps include:
- Exploratory Data Analysis (EDA) to understand feature distributions and correlations.
- Data preprocessing with techniques such as **balancing categorical features**, **feature engineering**, and **log transformations**.
- Model training and comparison across **Logistic Regression**, **Random Forests**, **SVM**, **kNN**, and a **neural network**.

---

## Dataset
- **Size:** 114,000 tracks  
- **Classes:** 114 genres (1,000 samples per genre)  
- **Features:** 20 attributes, including:
  - Audio features: `danceability`, `energy`, `valence`, `speechiness`
  - Metadata: `artist`, `explicit flag`, `time signature`

---

## Methodology

### 1. Data Preprocessing
- **Cleaning:** Removed duplicates and missing rows.
- **Balancing:** Applied upsampling to balance minority categories (e.g., explicit tracks, time signatures).
- **Encoding:** Used one-hot encoding for low-cardinality features and label encoding for high-cardinality ones.
- **Feature Engineering:** Created interaction features like `danceability_valence` and `energy_acousticness`.
- **Transformations:** Applied log transformations to skewed features and standardized numeric columns.

### 2. Models Trained
We experimented with a variety of models:
- **Simple Models:** Logistic Regression, Naive Bayes  
- **Tree-based Models:** Decision Trees, Random Forests  
- **Distance and Margin-based Models:** k-Nearest Neighbors (kNN), Support Vector Machines (SVM)  
- **Neural Network:** A fully connected network with ReLU activations and dropout regularization.

### 3. Hyperparameter Tuning
- Used **RandomizedSearchCV** to efficiently explore parameter configurations.
- Final models were retrained on the full dataset after tuning.

---

## Results

| Model               | Accuracy | Training Time |
|---------------------|----------|---------------|
| Logistic Regression | 19.82%   | Slow          |
| Naive Bayes         | 47.95%   | Fast          |
| Decision Tree       | 85.59%   | Fast          |
| Random Forest       | 84.81%   | Moderate      |
| SVM                 | 49.30%   | Slow          |
| kNN                 | 82.72%   | Slow          |
| Neural Network      | 61.64%   | Slow          |

- **Random Forests** and **Decision Trees** performed best, achieving over **80% accuracy**.
- Preprocessing steps such as balancing features and log transformations significantly improved performance.

![Model Performance](model_accuracies.png)

---

## Key Features and Insights
- Top features identified by the Random Forest model include:
  - `valence`
  - `speechiness`
  - `popularity`

These features were crucial for improving genre classification performance.

![Feature Importance](top_10_feature_importances.png)

---

## Challenges
- Distinguishing between **114 genres** proved difficult due to subtle differences in audio characteristics.
- Training complex models required significant computation time, presenting challenges for scalability.

---

## Future Work
Possible next steps include:
- Further feature analysis and engineering.
- More extensive hyperparameter tuning.
- Exploration of deep learning models like **convolutional recurrent neural networks (CRNNs)** for learning from raw audio data.

---

## Contributions
- **Praveen Rangavajhula:** Data preprocessing, feature engineering, model training, parameter tuning, results analysis, and visualizations.
- **Qingchan Zhu:** Literature review, EDA, algorithm selection, hyperparameter tuning, and drafting sections on data description and results interpretation.

