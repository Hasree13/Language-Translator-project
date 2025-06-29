import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense
import os

# --- Model and Training Parameters (Re-named) ---
training_batch_size = 64
num_epochs_train = 100
embedding_dimension = 256  # Equivalent to latent_dim
sample_limit = 10000       # Renamed from num_samples

# --- Data Source (Re-oriented) ---
source_data_file = 'eng-french.txt'
trained_model_output_path = 'fr_translator_model.h5' # New model save name

# --- Data Preparation Stages (Re-oriented/Re-named variables) ---
source_sentences = []
target_phrases = []
input_vocabulary = set()
output_vocabulary = set()

print("Initiating data loading and vocabulary construction...")
try:
    with open(source_data_file, 'r', encoding='utf-8') as f:
        all_lines = f.read().split('\n')
except FileNotFoundError:
    print(f"Error: The dataset file '{source_data_file}' was not found.")
    print("Please ensure 'eng-french.txt' is in the same directory as this script.")
    exit()

# Process lines up to the defined sample limit
for current_line in all_lines[: min(sample_limit, len(all_lines) - 1)]:
    try:
        source_text, target_text_raw = current_line.split('\t')
    except ValueError:
        continue # Skip malformed lines

    # Add start-of-sequence and end-of-sequence tokens to target phrases
    processed_target_phrase = '\t' + target_text_raw + '\n'

    source_sentences.append(source_text)
    target_phrases.append(processed_target_phrase)

    # Populate unique characters for input vocabulary
    for char_in_input in source_text:
        input_vocabulary.add(char_in_input)

    # Populate unique characters for output vocabulary
    for char_in_output in processed_target_phrase:
        output_vocabulary.add(char_in_output)

# Sort vocabularies for consistent indexing
sorted_input_chars = sorted(list(input_vocabulary))
sorted_output_chars = sorted(list(output_vocabulary))

# Calculate vocabulary sizes and max sequence lengths
size_input_vocab = len(sorted_input_chars)
size_output_vocab = len(sorted_output_chars)

max_input_sequence_length = max([len(s) for s in source_sentences])
max_output_sequence_length = max([len(p) for p in target_phrases])

print(f'Total number of processed samples: {len(source_sentences)}')
print(f'Size of unique input character set: {size_input_vocab}')
print(f'Size of unique output character set: {size_output_vocab}')
print(f'Maximum input sequence length: {max_input_sequence_length}')
print(f'Maximum output sequence length: {max_output_sequence_length}')

# Create character-to-index mappings
char_to_input_idx = dict([(char, i) for i, char in enumerate(sorted_input_chars)])
char_to_output_idx = dict([(char, i) for i, char in enumerate(sorted_output_chars)])

# Create numerical input/output data tensors (one-hot encoded)
encoder_data_matrix = np.zeros(
    (len(source_sentences), max_input_sequence_length, size_input_vocab),
    dtype='float32')
decoder_input_tensor = np.zeros(
    (len(source_sentences), max_output_sequence_length, size_output_vocab),
    dtype='float32')
decoder_target_tensor = np.zeros(
    (len(source_sentences), max_output_sequence_length, size_output_vocab),
    dtype='float32')

# Populate the one-hot encoded matrices
for sample_idx, (input_text_data, target_text_data) in enumerate(zip(source_sentences, target_phrases)):
    # Encoder input data
    for char_idx, current_char in enumerate(input_text_data):
        encoder_data_matrix[sample_idx, char_idx, char_to_input_idx[current_char]] = 1.0
    # Pad remaining input sequence with a 'space' token if shorter than max length
    if char_idx + 1 < max_input_sequence_length:
        encoder_data_matrix[sample_idx, char_idx + 1:, char_to_input_idx.get(' ', 0)] = 1.0

    # Decoder input and target data
    for char_idx, current_char in enumerate(target_text_data):
        # Decoder input (shifted by one for teacher forcing)
        decoder_input_tensor[sample_idx, char_idx, char_to_output_idx[current_char]] = 1.0
        # Decoder target (output sequence, shifted by one relative to input)
        if char_idx > 0:
            decoder_target_tensor[sample_idx, char_idx - 1, char_to_output_idx[current_char]] = 1.0
    # Pad remaining decoder sequences with a 'space' token
    if char_idx + 1 < max_output_sequence_length:
        decoder_input_tensor[sample_idx, char_idx + 1:, char_to_output_idx.get(' ', 0)] = 1.0
    if char_idx < max_output_sequence_length: # For target, last valid char is at char_idx-1
        decoder_target_tensor[sample_idx, char_idx:, char_to_output_idx.get(' ', 0)] = 1.0

# --- Neural Network Architecture Definition (Re-named layers/variables) ---
print("\nDefining the sequence-to-sequence model architecture...")

# Encoder network
encoder_input_layer = Input(shape=(None, size_input_vocab), name='encoder_input_layer')
encoder_lstm_unit = LSTM(embedding_dimension, return_state=True, name='encoder_lstm_unit')
encoder_output_states, state_h_vec, state_c_vec = encoder_lstm_unit(encoder_input_layer)
encoder_final_states = [state_h_vec, state_c_vec] # The context vector

# Decoder network
decoder_input_layer = Input(shape=(None, size_output_vocab), name='decoder_input_layer')
decoder_lstm_unit = LSTM(embedding_dimension, return_sequences=True, return_state=True, name='decoder_lstm_unit')
decoder_sequence_output, _, _ = decoder_lstm_unit(decoder_input_layer,
                                                  initial_state=encoder_final_states)
decoder_output_projection = Dense(size_output_vocab, activation='softmax', name='decoder_output_projection')
final_decoder_predictions = decoder_output_projection(decoder_sequence_output)

# Construct the full training model
seq2seq_training_model = Model([encoder_input_layer, decoder_input_layer], final_decoder_predictions, name='Seq2Seq_Translation_Model')

# --- Model Training (Re-oriented comments/print statements) ---
print("\nCommencing model training process...")
seq2seq_training_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
seq2seq_training_model.summary() # Added summary for better visibility

# Fit the model to the prepared data
history = seq2seq_training_model.fit(
    [encoder_data_matrix, decoder_input_tensor],
    decoder_target_tensor,
    batch_size=training_batch_size,
    epochs=num_epochs_train,
    validation_split=0.2, # 20% of data used for validation
    verbose=1
)

# Save the trained model
seq2seq_training_model.save(trained_model_output_path)
print(f"Trained model saved to: {trained_model_output_path}")

# --- Inference Model Construction (Re-named variables, slightly different flow) ---
print("\nPreparing inference models for translation...")

# Encoder inference model: Maps input sequence to state vectors
inference_encoder_model = Model(encoder_input_layer, encoder_final_states, name='Inference_Encoder')

# Decoder inference model: Takes target sequence and states, outputs probabilities and new states
decoder_state_h_input = Input(shape=(embedding_dimension,), name='decoder_state_h_input')
decoder_state_c_input = Input(shape=(embedding_dimension,), name='decoder_state_c_input')
decoder_initial_states_for_inference = [decoder_state_h_input, decoder_state_c_input]

decoder_inference_output, new_state_h, new_state_c = decoder_lstm_unit(
    decoder_input_layer, initial_state=decoder_initial_states_for_inference)
decoder_inference_states_output = [new_state_h, new_state_c]
decoder_inference_predictions = decoder_output_projection(decoder_inference_output)

inference_decoder_model = Model(
    [decoder_input_layer] + decoder_initial_states_for_inference,
    [decoder_inference_predictions] + decoder_inference_states_output,
    name='Inference_Decoder'
)

# --- Reverse Lookup Dictionaries (Re-named) ---
idx_to_input_char = dict([(i, char) for char, i in char_to_input_idx.items()])
idx_to_output_char = dict([(i, char) for char, i in char_to_output_idx.items()])

# --- Translation Function (Re-named variables and comments) ---
def translate_sequence(source_seq_input):
    # Encode the input sequence into context vectors (encoder states)
    current_states_from_encoder = inference_encoder_model.predict(source_seq_input, verbose=0)

    # Prepare the initial target sequence with the start token
    decoder_input_for_prediction = np.zeros((1, 1, size_output_vocab))
    decoder_input_for_prediction[0, 0, char_to_output_idx['\t']] = 1.0

    generated_translation = ''
    translation_completed = False

    # Iterative decoding loop
    while not translation_completed:
        output_probabilities, h_state, c_state = inference_decoder_model.predict(
            [decoder_input_for_prediction] + current_states_from_encoder, verbose=0)

        # Sample the next character with the highest probability
        next_char_index = np.argmax(output_probabilities[0, -1, :])
        predicted_char = idx_to_output_char[next_char_index]
        generated_translation += predicted_char

        # Check termination conditions: newline or max length reached
        if (predicted_char == '\n' or len(generated_translation) > max_output_sequence_length):
            translation_completed = True

        # Update the decoder input for the next time step (with the newly predicted char)
        decoder_input_for_prediction = np.zeros((1, 1, size_output_vocab))
        decoder_input_for_prediction[0, 0, next_char_index] = 1.0

        # Update the states for the next iteration
        current_states_from_encoder = [h_state, c_state]

    return generated_translation

# --- Demonstration of Translation (Re-oriented print statements) ---
print("\n--- Demonstrating Translations (First 100 Samples) ---")
for demonstration_index in range(100):
    # Select an input sequence from the training data for demonstration
    input_sample_for_decoding = encoder_data_matrix[demonstration_index: demonstration_index + 1]
    translated_output = translate_sequence(input_sample_for_decoding)

    print(f'--- Sample {demonstration_index + 1} ---')
    print('Original Input:', source_sentences[demonstration_index])
    print('Generated Translation:', translated_output.strip()) # .strip() removes potential trailing newlinesimport numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense
import os

# --- Model and Training Parameters (Re-named) ---
training_batch_size = 64
num_epochs_train = 100
embedding_dimension = 256  # Equivalent to latent_dim
sample_limit = 10000       # Renamed from num_samples

# --- Data Source (Re-oriented) ---
source_data_file = 'eng-french.txt'
trained_model_output_path = 'fr_translator_model.h5' # New model save name

# --- Data Preparation Stages (Re-oriented/Re-named variables) ---
source_sentences = []
target_phrases = []
input_vocabulary = set()
output_vocabulary = set()

print("Initiating data loading and vocabulary construction...")
try:
    with open(source_data_file, 'r', encoding='utf-8') as f:
        all_lines = f.read().split('\n')
except FileNotFoundError:
    print(f"Error: The dataset file '{source_data_file}' was not found.")
    print("Please ensure 'eng-french.txt' is in the same directory as this script.")
    exit()

# Process lines up to the defined sample limit
for current_line in all_lines[: min(sample_limit, len(all_lines) - 1)]:
    try:
        source_text, target_text_raw = current_line.split('\t')
    except ValueError:
        continue # Skip malformed lines

    # Add start-of-sequence and end-of-sequence tokens to target phrases
    processed_target_phrase = '\t' + target_text_raw + '\n'

    source_sentences.append(source_text)
    target_phrases.append(processed_target_phrase)

    # Populate unique characters for input vocabulary
    for char_in_input in source_text:
        input_vocabulary.add(char_in_input)

    # Populate unique characters for output vocabulary
    for char_in_output in processed_target_phrase:
        output_vocabulary.add(char_in_output)

# Sort vocabularies for consistent indexing
sorted_input_chars = sorted(list(input_vocabulary))
sorted_output_chars = sorted(list(output_vocabulary))

# Calculate vocabulary sizes and max sequence lengths
size_input_vocab = len(sorted_input_chars)
size_output_vocab = len(sorted_output_chars)

max_input_sequence_length = max([len(s) for s in source_sentences])
max_output_sequence_length = max([len(p) for p in target_phrases])

print(f'Total number of processed samples: {len(source_sentences)}')
print(f'Size of unique input character set: {size_input_vocab}')
print(f'Size of unique output character set: {size_output_vocab}')
print(f'Maximum input sequence length: {max_input_sequence_length}')
print(f'Maximum output sequence length: {max_output_sequence_length}')

# Create character-to-index mappings
char_to_input_idx = dict([(char, i) for i, char in enumerate(sorted_input_chars)])
char_to_output_idx = dict([(char, i) for i, char in enumerate(sorted_output_chars)])

# Create numerical input/output data tensors (one-hot encoded)
encoder_data_matrix = np.zeros(
    (len(source_sentences), max_input_sequence_length, size_input_vocab),
    dtype='float32')
decoder_input_tensor = np.zeros(
    (len(source_sentences), max_output_sequence_length, size_output_vocab),
    dtype='float32')
decoder_target_tensor = np.zeros(
    (len(source_sentences), max_output_sequence_length, size_output_vocab),
    dtype='float32')

# Populate the one-hot encoded matrices
for sample_idx, (input_text_data, target_text_data) in enumerate(zip(source_sentences, target_phrases)):
    # Encoder input data
    for char_idx, current_char in enumerate(input_text_data):
        encoder_data_matrix[sample_idx, char_idx, char_to_input_idx[current_char]] = 1.0
    # Pad remaining input sequence with a 'space' token if shorter than max length
    if char_idx + 1 < max_input_sequence_length:
        encoder_data_matrix[sample_idx, char_idx + 1:, char_to_input_idx.get(' ', 0)] = 1.0

    # Decoder input and target data
    for char_idx, current_char in enumerate(target_text_data):
        # Decoder input (shifted by one for teacher forcing)
        decoder_input_tensor[sample_idx, char_idx, char_to_output_idx[current_char]] = 1.0
        # Decoder target (output sequence, shifted by one relative to input)
        if char_idx > 0:
            decoder_target_tensor[sample_idx, char_idx - 1, char_to_output_idx[current_char]] = 1.0
    # Pad remaining decoder sequences with a 'space' token
    if char_idx + 1 < max_output_sequence_length:
        decoder_input_tensor[sample_idx, char_idx + 1:, char_to_output_idx.get(' ', 0)] = 1.0
    if char_idx < max_output_sequence_length: # For target, last valid char is at char_idx-1
        decoder_target_tensor[sample_idx, char_idx:, char_to_output_idx.get(' ', 0)] = 1.0

# --- Neural Network Architecture Definition (Re-named layers/variables) ---
print("\nDefining the sequence-to-sequence model architecture...")

# Encoder network
encoder_input_layer = Input(shape=(None, size_input_vocab), name='encoder_input_layer')
encoder_lstm_unit = LSTM(embedding_dimension, return_state=True, name='encoder_lstm_unit')
encoder_output_states, state_h_vec, state_c_vec = encoder_lstm_unit(encoder_input_layer)
encoder_final_states = [state_h_vec, state_c_vec] # The context vector

# Decoder network
decoder_input_layer = Input(shape=(None, size_output_vocab), name='decoder_input_layer')
decoder_lstm_unit = LSTM(embedding_dimension, return_sequences=True, return_state=True, name='decoder_lstm_unit')
decoder_sequence_output, _, _ = decoder_lstm_unit(decoder_input_layer,
                                                  initial_state=encoder_final_states)
decoder_output_projection = Dense(size_output_vocab, activation='softmax', name='decoder_output_projection')
final_decoder_predictions = decoder_output_projection(decoder_sequence_output)

# Construct the full training model
seq2seq_training_model = Model([encoder_input_layer, decoder_input_layer], final_decoder_predictions, name='Seq2Seq_Translation_Model')

# --- Model Training (Re-oriented comments/print statements) ---
print("\nCommencing model training process...")
seq2seq_training_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
seq2seq_training_model.summary() # Added summary for better visibility

# Fit the model to the prepared data
history = seq2seq_training_model.fit(
    [encoder_data_matrix, decoder_input_tensor],
    decoder_target_tensor,
    batch_size=training_batch_size,
    epochs=num_epochs_train,
    validation_split=0.2, # 20% of data used for validation
    verbose=1
)

# Save the trained model
seq2seq_training_model.save(trained_model_output_path)
print(f"Trained model saved to: {trained_model_output_path}")

# --- Inference Model Construction (Re-named variables, slightly different flow) ---
print("\nPreparing inference models for translation...")

# Encoder inference model: Maps input sequence to state vectors
inference_encoder_model = Model(encoder_input_layer, encoder_final_states, name='Inference_Encoder')

# Decoder inference model: Takes target sequence and states, outputs probabilities and new states
decoder_state_h_input = Input(shape=(embedding_dimension,), name='decoder_state_h_input')
decoder_state_c_input = Input(shape=(embedding_dimension,), name='decoder_state_c_input')
decoder_initial_states_for_inference = [decoder_state_h_input, decoder_state_c_input]

decoder_inference_output, new_state_h, new_state_c = decoder_lstm_unit(
    decoder_input_layer, initial_state=decoder_initial_states_for_inference)
decoder_inference_states_output = [new_state_h, new_state_c]
decoder_inference_predictions = decoder_output_projection(decoder_inference_output)

inference_decoder_model = Model(
    [decoder_input_layer] + decoder_initial_states_for_inference,
    [decoder_inference_predictions] + decoder_inference_states_output,
    name='Inference_Decoder'
)

# --- Reverse Lookup Dictionaries (Re-named) ---
idx_to_input_char = dict([(i, char) for char, i in char_to_input_idx.items()])
idx_to_output_char = dict([(i, char) for char, i in char_to_output_idx.items()])

# --- Translation Function (Re-named variables and comments) ---
def translate_sequence(source_seq_input):
    # Encode the input sequence into context vectors (encoder states)
    current_states_from_encoder = inference_encoder_model.predict(source_seq_input, verbose=0)

    # Prepare the initial target sequence with the start token
    decoder_input_for_prediction = np.zeros((1, 1, size_output_vocab))
    decoder_input_for_prediction[0, 0, char_to_output_idx['\t']] = 1.0

    generated_translation = ''
    translation_completed = False

    # Iterative decoding loop
    while not translation_completed:
        output_probabilities, h_state, c_state = inference_decoder_model.predict(
            [decoder_input_for_prediction] + current_states_from_encoder, verbose=0)

        # Sample the next character with the highest probability
        next_char_index = np.argmax(output_probabilities[0, -1, :])
        predicted_char = idx_to_output_char[next_char_index]
        generated_translation += predicted_char

        # Check termination conditions: newline or max length reached
        if (predicted_char == '\n' or len(generated_translation) > max_output_sequence_length):
            translation_completed = True

        # Update the decoder input for the next time step (with the newly predicted char)
        decoder_input_for_prediction = np.zeros((1, 1, size_output_vocab))
        decoder_input_for_prediction[0, 0, next_char_index] = 1.0

        # Update the states for the next iteration
        current_states_from_encoder = [h_state, c_state]

    return generated_translation

# --- Demonstration of Translation (Re-oriented print statements) ---
print("\n--- Demonstrating Translations (First 100 Samples) ---")
for demonstration_index in range(100):
    # Select an input sequence from the training data for demonstration
    input_sample_for_decoding = encoder_data_matrix[demonstration_index: demonstration_index + 1]
    translated_output = translate_sequence(input_sample_for_decoding)

    print(f'--- Sample {demonstration_index + 1} ---')
    print('Original Input:', source_sentences[demonstration_index])
    print('Generated Translation:', translated_output.strip()) # .strip() removes potential trailing newlines