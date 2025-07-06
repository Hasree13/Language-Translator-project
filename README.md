# Language-Translator-project

This project implements a character-level sequence-to-sequence (seq2seq) model using LSTM layers to translate English sentences into French. It uses a dataset of aligned English-French sentence pairs and is built with TensorFlow/Keras.


## Model Overview

The model architecture consists of:

- Encoder: A single LSTM layer that processes the input English sentence.

- Decoder: Another LSTM layer that generates the output French sentence one character at a time.

- One-hot Encoding: Both inputs and outputs are encoded using one-hot vectors at the character level.

- Teacher Forcing: Used during training to accelerate convergence.


## Dataset

You need a tab-separated text file eng-french.txt where:

- Each line contains an English sentence and its French translation, separated by a tab (\t).

- Example:
        go	vas-y
        hi	salut

Place the dataset in: /content/drive/MyDrive/eng-french.txt


## Getting Started

### 1. Clone the Repository
git clone https://github.com/yourusername/english-french-seq2seq.git
cd english-french-seq2seq

### 2. Install Requirements
pip install tensorflow numpy

### 3. Train the Model
You can run the notebook or script to train the model:
- Make sure the dataset path is correct.
- Adjust hyperparameters as needed (epochs, batch size, etc).

#Inside notebook
translator.fit(...)

The trained model will be saved as: eng2french.h5


## Inference

After training:
- You can feed in an English sentence (already preprocessed to match one-hot format)
- The model generates the corresponding French translation character by character


## Sample Output

Source: hello
Translated: bonjour

Source: how are you?
Translated: comment ça va ?
