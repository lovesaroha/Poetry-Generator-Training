# Love Saroha
# lovesaroha1994@gmail.com (email address)
# https://www.lovesaroha.com (website)
# https://github.com/lovesaroha  (github)

# Generate poetry.
from tensorflow import keras
import numpy
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import regularizers
import tensorflow.keras.utils as ku

# Parameters.
epochs = 30
batchSize = 256
embedding_dim = 100

# Download training text from (https://drive.google.com/uc?id=108jAePKK4R3BVYBbYJZ32JWUwxeMg20K).

# Load text data.
text_data = open("sonnets.txt").read()
text_data = text_data.lower().split("\n")

# Create tokenizer.
tokenizer = Tokenizer()
tokenizer.fit_on_texts(text_data)
total_words = len(tokenizer.word_index) + 1

# Create input sequences.
input_sequences = []
for line in text_data:
	token_list = tokenizer.texts_to_sequences([line])[0]
	for i in range(1, len(token_list)):
		n_gram_sequence = token_list[:i+1]
		input_sequences.append(n_gram_sequence)

# Add padding.
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = numpy.array(pad_sequences(
    input_sequences, maxlen=max_sequence_len, padding='pre'))

# Prepare labels and training data,
training_data, training_labels = input_sequences[:,
    :-1], input_sequences[:, -1]

# Set training labels.
training_labels = ku.to_categorical(training_labels, num_classes=total_words)

# Create model with  output units equal to total words.
model = keras.Sequential([
    keras.layers.Embedding(total_words, embedding_dim,
                           input_length=max_sequence_len-1),
    keras.layers.Bidirectional(keras.layers.LSTM(150, return_sequences=True)),
    keras.layers.Dropout(0.3),
    keras.layers.LSTM(100),
    keras.layers.Dense(total_words/2, activation="relu",
                       kernel_regularizer=regularizers.l2(0.01)),
    keras.layers.Dense(total_words, activation="sigmoid")
])

# Set loss function and optimizer.
model.compile(loss="categorical_crossentropy",
              optimizer="adam", metrics=["accuracy"])


class myCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
      # Stop when validation accuracy is more than 98%.
        if logs.get('val_accuracy') is not None and logs.get('val_accuracy') > 0.98:
            print("\nTraining Stopped!")
            self.model.stop_training = True


# Callback function to check accuracy.
checkAccuracy = myCallback()

# Train model.
model.fit(training_data, training_labels, epochs=epochs, callbacks=[
         checkAccuracy], batch_size=batchSize, verbose=1)

# Generate from given text.
text = "i have never"
next_words = 10

for _ in range(next_words):
	token_list = tokenizer.texts_to_sequences([text])[0]
	token_list = pad_sequences(
	    [token_list], maxlen=max_sequence_len-1, padding='pre')
	prediction = numpy.argmax(model.predict(token_list, verbose=0) , axis=1)
	output_word = ""
	for word, index in tokenizer.word_index.items():
		if index == prediction:
			output_word = word
			break
	text += " " + output_word

print(text)         
