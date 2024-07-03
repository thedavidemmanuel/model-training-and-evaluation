# Fake News Classification Project by David Emmanuel

## Overview

This project aims to develop a machine learning model to classify news articles as either real or fake. We use a dataset of labeled news articles and implement various natural language processing and deep learning techniques to create an effective classifier.

## Dataset

We use the ISOT Fake News Dataset, which contains two types of articles: fake and real news.

### Dataset Details:

- **Source**: The dataset was collected from real-world sources.
- **Content**:
  - True.csv: Contains over 12,600 true articles from Reuters.com
  - Fake.csv: Contains over 12,600 fake articles from various unreliable websites
- **Time Period**: Focused on articles from 2016 to 2017
- **Features**: Each article contains the title, text, type, and publication date

## Methodology

### 1. Data Preprocessing

- URL removal
- Special character and number removal
- Text lowercasing
- Stopword removal
- Stemming

### 2. Feature Engineering

- Text length calculation
- Title length calculation
- Readability score computation (using Flesch-Kincaid Grade Level)
- Sentiment analysis for title and text

### 3. Text Representation

- TF-IDF (Term Frequency-Inverse Document Frequency) vectorization

### 4. Model Architecture

We implement two neural network models:

1. Simple Model
2. Optimized Model

### 5. Optimization Techniques

- L1/L2 Regularization
- Dropout
- Batch Normalization
- Early Stopping
- Learning Rate Adjustment
- Hyperparameter Tuning

## Implementation Details

### Libraries Used

- pandas, numpy: Data manipulation
- matplotlib, seaborn: Data visualization
- nltk: Natural Language Processing
- scikit-learn: Machine Learning utilities
- tensorflow: Deep Learning framework
- textstat: Readability scoring
- textblob: Sentiment analysis

### Model Architectures

#### Simple Model

- Input Layer
- 3 Dense Layers with ReLU activation
- Output Layer with Sigmoid activation

#### Optimized Model

- Input Layer
- 3 Dense Layers with ReLU activation and L1/L2 regularization
- Batch Normalization after each Dense Layer
- Dropout Layers
- Output Layer with Sigmoid activation

### Hyperparameter Tuning

We use a manual random search to tune the following hyperparameters:

- Batch size
- Number of epochs
- Learning rate
- Number of neurons in the first layer

## Results

(Include a summary of your results here, such as accuracy, precision, recall, and F1-score for both models)

## Conclusion

(Summarize the key findings, the performance difference between the simple and optimized models, and any insights gained from the project)

## Future Work

- Experiment with more advanced architectures (e.g., LSTM, transformer-based models)
- Incorporate external knowledge bases for fact-checking
- Implement an ensemble of different models
- Explore more features related to writing style and source credibility

## How to Use

1. Clone this repository
2. Install the required packages: `pip install -r requirements.txt`
3. Run the Jupyter notebook or Python script
4. The trained models will be saved in the 'saved_models' directory

## References

1. Ahmed H, Traore I, Saad S. "Detecting opinion spams and fake news using text classification", Journal of Security and Privacy, Volume 1, Issue 1, Wiley, January/February 2018.
2. Ahmed H, Traore I, Saad S. (2017) "Detection of Online Fake News Using N-Gram Analysis and Machine Learning Techniques. In: Traore I., Woungang I., Awad A. (eds) Intelligent, Secure, and Dependable Systems in Distributed and Cloud Environments. ISDDC 2017. Lecture Notes in Computer Science, vol 10618. Springer, Cham (pp. 127- 138).
