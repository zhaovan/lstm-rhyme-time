import numpy as np
import tensorflow as tf
from preprocess import *
import copy


class PoemGenerator(tf.keras.Model):
    def __init__(self, vocab_size):
        super(PoemGenerator, self).__init__()

        # hyper parameters
        self.batch_size = 128
        self.window_size = 150
        self.learning_rate = 0.001
        self.rnn_size = 128
        # both hidden size and embedding size were recommended to be 256 by the paper
        self.hidden_size = 256
        self.embedding_size = 256
        self.vocab_size = vocab_size
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate)

        # instantiate layers for our model
        self.embedding_layer = tf.keras.layers.Embedding(
            self.vocab_size, self.embedding_size, input_length=self.window_size)
        self.rnn_layer = tf.keras.layers.LSTM(
            self.rnn_size, return_sequences=True, return_state=True)
        self.dense_1 = tf.keras.layers.Dense(
            self.hidden_size, activation='relu')
        self.dense_2 = tf.keras.layers.Dense(
            self.vocab_size)

    def call(self, inputs, initial_state):
        # architecture from our paper - embedding layer, LSTM layer, and then 2 dense layers before returning logits
        embeddings = self.embedding_layer(tf.convert_to_tensor(inputs))
        (output, last_output, cell_state) = self.rnn_layer(embeddings, initial_state)
        logits = self.dense_1(output)
        logits = self.dense_2(logits)
        return logits, last_output, cell_state

    def loss(self, logits, labels):
        # sparse categorical crossentropy with logits because we don't softmax in our call function
        return tf.math.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True))

def train(model, train_inputs, train_labels, epoch):
    i = 0
    end = model.batch_size
    length = len(train_labels)
    # shuffle poems for better results
    indices = tf.random.shuffle([l for l in range(length)])
    train_inputs = tf.gather(train_inputs, indices)
    train_labels = tf.gather(train_labels, indices)
    batch = 0
    # loop through all the poems and update gradients
    while end <= length:
        batch_inputs = train_inputs[i:end]
        batch_labels = train_labels[i:end]
        with tf.GradientTape() as tape:
            logits, _, _ = model(batch_inputs, None)
            loss = model.loss(logits, batch_labels)
            if batch % 100 == 0:
                print(loss)
            batch += 1
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
        logits, _, _ = model(batch_inputs, None)
        loss += model.loss(logits, batch_labels)
        avg += 1
        i += model.batch_size
        end += model.batch_size

    # return perplexity
    return tf.math.exp(loss / avg)


def generate_poem(word1, vocab, model):
    reverse_vocab = {idx: word for word, idx in vocab.items()}
    previous_state = None
    first_string = word1
    first_word_index = vocab[word1]
    next_input = [[first_word_index]]
    text = [first_string]

    # for loop to generate next 200 phonemes, or the END token
    for _ in range(200):
        logits, last_output, cell_state = model.call(next_input, previous_state)
        previous_state = (last_output, cell_state)

        logits = tf.squeeze(logits, 0)
        # weighted randomization of the next phoneme
        out_index = tf.random.categorical(logits, num_samples=1)[-1, 0].numpy()
        
        next_phoneme = reverse_vocab[out_index]
        text.append(reverse_vocab[out_index])

        if next_phoneme == "END":
            break

        next_input = [[out_index]]

    print(" ".join(text))


def main():
    (ids, phoneme_dict, phoneme_size, pad_index) = get_data(
        "./data/limericks.csv")

    # first 80% of the poems are for training, the last 20% are for testing
    num_train_ids = int(0.8 * len(ids))

    label_ids = copy.deepcopy(ids)

    # cut off the first word of every poem and add a PAD token to the end of them for labels
    for i in range(len(label_ids)):
        l = label_ids[i]
        new_l = l[1:]
        new_l.append(pad_index)
        label_ids[i] = new_l
    
    # making training inputs and labels
    train_inputs = ids[:num_train_ids]
    train_labels = label_ids[:num_train_ids]

    # making testing inputs and labels
    test_inputs = ids[num_train_ids:]
    test_labels = label_ids[num_train_ids:]

    # instantiate model with phoneme_size
    model = PoemGenerator(phoneme_size)

    # train model for 15 epochs, printing perplexity after each epoch
    for i in range(15):
        print("------------------------------")
        print("EPOCH " + str(i + 1))
        train(model, train_inputs, train_labels, i + 1)
        perplexity = test(model, test_inputs, test_labels)
        print("PERPLEXITY OF EPOCH " + str(i + 1) + ": " + str(perplexity))

    # while loop to generate poems starting with a specific phoneme
    while True:
        query = input("Enter phoneme: ")
        try:
            generate_poem(query, phoneme_dict, model)
        except:
            print("Invalid key. Type again")


if __name__ == '__main__':
    main()
