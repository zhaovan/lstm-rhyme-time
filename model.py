import numpy as np
import tensorflow as tf
from preprocess import *


class PoemGenerator(tf.keras.Model):
    def __init__(self, vocab_size):
        """
        The ReinforceWithBaseline class that inherits from tf.keras.Model.
        """
        super(PoemGenerator, self).__init__()

        # Numerical Hyperparameters
        self.batch_size = 128
        self.window_size = 150
        self.learning_rate = 0.001
        self.rnn_size = 128
        self.hidden_size = 256
        self.embedding_size = 256
        self.vocab_size = vocab_size
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate)

        self.embedding_layer = tf.keras.layers.Embedding(
            self.vocab_size, self.embedding_size, input_length=self.window_size)
        self.rnn_layer = tf.keras.layers.GRU(
            self.rnn_size, return_sequences=True, return_state=True)
        self.dense_1 = tf.keras.layers.Dense(
            self.hidden_size, activation='relu')
        self.dense_2 = tf.keras.layers.Dense(
            self.vocab_size, activation='softmax')

    def call(self, inputs, initial_state):
        embeddings = self.embedding_layer(tf.convert_to_tensor(inputs))
        (output, state) = self.rnn_layer(embeddings, initial_state)
        logits = self.dense_1(output)
        probs = self.dense_2(logits)
        return probs, state

    def loss(self, logits, labels):
        return tf.math.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, logits))


def train(model, train_inputs, train_labels):
    i = 0
    end = model.batch_size
    length = len(train_labels)
    indices = tf.random.shuffle([l for l in range(length)])
    train_inputs = tf.gather(train_inputs, indices)
    train_labels = tf.gather(train_labels, indices)
    while end <= length:
        batch_inputs = train_inputs[i:end]
        batch_labels = train_labels[i:end]
        with tf.GradientTape() as tape:
            logits, _ = model(batch_inputs, None)
            loss = model.loss(logits, batch_labels)
            print(i / model.batch_size, ": ", loss)
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(
            zip(gradients, model.trainable_variables))
        i += model.batch_size
        end += model.batch_size


def test(model, test_inputs, test_labels):
    i = 0
    end = model.batch_size
    length = len(test_labels)
    avg = 0
    loss = 0
    while end <= length:
        batch_inputs = test_inputs[i:end]
        batch_labels = test_labels[i:end]
        logits, _ = model(batch_inputs, None)
        loss += model.loss(logits, batch_labels)
        avg += 1
        i += model.batch_size
        end += model.batch_size

    return tf.math.exp(loss / avg)


def generate_sentence(word1, length, vocab, model):
    reverse_vocab = {idx: word for word, idx in vocab.items()}
    previous_state = None

    first_string = word1
    first_word_index = vocab[word1]
    next_input = [[first_word_index]]
    text = [first_string]

    for i in range(length):
        logits, previous_state = model.call(next_input, previous_state)
        out_index = np.argmax(np.array(logits[0][0]))

        text.append(reverse_vocab[out_index])
        next_input = [[out_index]]

    print(" ".join(text))


def main():
    (train_ids, syllable_dict, syllable_size) = get_data(
        "./data/limericks.csv")
    train_inputs = train_ids[:-1]
    train_labels = train_ids[1:]

    model = PoemGenerator(syllable_size)

    i = 0
    end = model.window_size
    train_length = len(train_labels)
    train_inputs_setup = []
    train_labels_setup = []
    while end <= train_length:
        train_inputs_setup.append(train_inputs[i:end])
        train_labels_setup.append(train_labels[i:end])
        i += model.window_size
        end += model.window_size
    train(model, train_inputs_setup, train_labels_setup)
    while True:
        query = input("Enter syllable: ")
        generate_sentence(query, 500, syllable_dict, model)


if __name__ == '__main__':
    main()
