# Language-Translator-project
**English-French Sequence-to-Sequence Translator**
This project implements a basic character-level Sequence-to-Sequence (Seq2Seq) model using TensorFlow and Keras to translate short English sentences into French. The model is trained on a dataset of English-French sentence pairs and demonstrates the fundamental principles of neural machine translation with Long Short-Term Memory (LSTM) networks.

**Features**
Character-level Translation: Translates text character by character.

Seq2Seq Model: Utilizes an Encoder-Decoder architecture with LSTM layers.

Teacher Forcing: Employs teacher forcing during training for stable learning.

Inference Mode: Supports generating translations for new input sentences after training.

Model Persistence: Saves the trained model for later use.

**Prerequisites**
Before running this project, ensure you have the following installed:

Python 3.x

pip (Python package installer)

Dataset
This project requires a dataset of English-French sentence pairs. The script expects a file named eng-french.txt in the same directory as the Python script.

Each line in eng-french.txt should contain an English sentence and its corresponding French translation, separated by a tab (\t).

The dataset used is taken from Kaggle.


Running the script will perform the following steps:

Load and preprocess the data.

Define and compile the Seq2Seq model.

Train the model for a specified number of epochs.

Save the trained model as fr_translator_model.h5.

Set up the inference (prediction) models.

Demonstrate translation by decoding the first 100 input sentences from the training set.

**Model Architecture**
The translation system is built upon a standard Sequence-to-Sequence architecture consisting of:

Encoder: An LSTM layer that processes the input (English) sequence and compresses its information into a fixed-size "context vector" (represented by its final hidden and cell states).

Decoder: Another LSTM layer that takes the context vector from the encoder as its initial state. It then generates the output (French) sequence one character at a time. During training, it uses "teacher forcing" where the actual target character from the previous time step is fed as input. During inference, the predicted character from the previous time step is fed as input.

Dense Layer with Softmax: A final dense layer with a softmax activation function is used on the decoder's output to predict the probability distribution over the target vocabulary for each time step.

**Customization**
You can modify the following parameters at the beginning of the seq2seq_translation.py script to experiment with the model's behavior:

training_batch_size: Number of samples per gradient update.

num_epochs_train: Number of training iterations over the entire dataset.

embedding_dimension: The dimensionality of the output space and the hidden state for the LSTMs. Larger values allow the model to learn more complex patterns but require more data and computation.

sample_limit: The maximum number of sentence pairs to use from the dataset. Useful for quicker experimentation with smaller subsets.

source_data_file: The name of your dataset file.

trained_model_output_path: The filename for saving the trained Keras model.
