# Twitter_sentiment_analysis

## Sentiment Analysis with LSTM using TensorFlow

## Overview
This repository contains a sentiment analysis project using LSTM (Long Short-Term Memory) neural networks implemented in TensorFlow. The model is trained on a Twitter dataset to classify tweets into positive and negative sentiment categories.

## Dataset
The dataset used (`Twitter_Data.csv`) contains tweets labeled with sentiment categories. Tweets with a sentiment score of 0 were removed, and preprocessing steps included removing special characters, converting text to lowercase, and tokenizing the text data.

## Preprocessing
- Removed special characters and URLs from tweets.
- Converted text to lowercase for uniformity.
- Tokenized text using Keras' `Tokenizer` and padded sequences to a maximum length.

## Model Architecture
The LSTM model architecture consists of:
- An Embedding layer for word embeddings.
- SpatialDropout1D layer for regularization.
- LSTM layer with 128 units and 30% dropout.
- Dense layers with ReLU activation and dropout for classification.
- Output layer with a sigmoid activation function for binary classification.

## Training
The model was trained for 22 epochs with a batch size of 64. Training was optimized using the Adam optimizer with a learning rate of 1e-4 and binary cross-entropy loss. Class weights were adjusted to handle class imbalance.

### Callbacks Used:
- **TensorBoard**: Visualize metrics and model graphs during training.
- **EarlyStopping**: Stop training when validation accuracy plateaus.
- **ModelCheckpoint**: Save the best model based on validation accuracy.
- **ReduceLROnPlateau**: Reduce learning rate when validation accuracy stops improving.

## Results
After training, the model achieved:
- **Training Accuracy**: 97.57%
- **Validation Accuracy**: 95.39%


## Usage
Clone the repository:
   ```
   git clone [https://github.com/your-username/sentiment-analysis-lstm.git](https://github.com/Ahmedmostafa2000/Twitter_sentiment_analysis)
   ```

## Acknowledgments

- Dataset sourced from [HuggingFace](https://huggingface.co/datasets/carblacac/twitter-sentiment-analysis)
