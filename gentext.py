# -*- coding: utf-8 -*-

import tensorflow as tf

import numpy
import os

class BuildModel(tf.keras.Model):
    def __init__(self, text, embedding_dim=256, rnn_units=1024):
        super(BuildModel, self).__init__()
        self.text = text
        self.vocab = sorted(set(text))
        self.vocab_size = len(tf.keras.layers.StringLookup(
            vocabulary=list(self.vocab), mask_token=None).get_vocabulary())
        # self.vocab_size = len(self.ids_from_chars.get_vocabulary())
        self.embedding = tf.keras.layers.Embedding(self.vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(rnn_units,
                                    return_sequences=True,
                                    return_state=True)
        self.dense = tf.keras.layers.Dense(self.vocab_size)

    def call(self, inputs, states=None, return_state=False, training=False):
        x = inputs
        x = self.embedding(x, training=training)
        if states is None:
            states = self.gru.get_initial_state(x)
        x, states = self.gru(x, initial_state=states, training=training)
        x = self.dense(x, training=training)

        if return_state:
            return x, states
        else:
            return x

class OneStep(tf.keras.Model):
    def __init__(self, text, model, temperature=1.0):
        super().__init__()
        self.vocab = sorted(set(text))
        self.ids_from_chars = tf.keras.layers.StringLookup(
            vocabulary=list(self.vocab), mask_token=None)
        self.chars_from_ids = tf.keras.layers.StringLookup(
            vocabulary= self.ids_from_chars.get_vocabulary(), invert=True, mask_token=None)
        self.temperature = temperature
        self.model = model
        # Create a mask to prevent "[UNK]" from being generated.
        skip_ids = self.ids_from_chars(['[UNK]'])[:, None]
        sparse_mask = tf.SparseTensor(
            # Put a -inf at each bad index.
            values=[-float('inf')]*len(skip_ids),
            indices=skip_ids,
            # Match the shape to the vocabulary
            dense_shape=[len(self.ids_from_chars.get_vocabulary())])
        self.prediction_mask = tf.sparse.to_dense(sparse_mask)

    @tf.function
    def generate_one_step(self, inputs, states=None):
        # Convert strings to token IDs.
        input_chars = tf.strings.unicode_split(inputs, 'UTF-8')
        input_ids = self.ids_from_chars(input_chars).to_tensor()

        # Run the model.
        # predicted_logits.shape is [batch, char, next_char_logits]
        predicted_logits, states = self.model(inputs=input_ids, states=states,
                                            return_state=True)
        # Only use the last prediction.
        predicted_logits = predicted_logits[:, -1, :]
        predicted_logits /= self.temperature
        # Apply the prediction mask: prevent "[UNK]" from being generated.
        predicted_logits += self.prediction_mask

        # Sample the output logits to generate token IDs.
        predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
        predicted_ids = tf.squeeze(predicted_ids, axis=-1)

        # Convert from token ids to characters
        predicted_chars = self.chars_from_ids(predicted_ids)

        # Return the characters and model state.
        return predicted_chars, states

class GenText(BuildModel):
    def __init__ (self, checkpoint_dir = './training_checkpoints'):
        super(GenText, self).__init__(text)
        self.text = text
        self.checkpoint_dir = checkpoint_dir
        self.vocab = sorted(set(text))
        self.ids_from_chars = tf.keras.layers.StringLookup(
            vocabulary=list(self.vocab), mask_token=None)
        self.chars_from_ids = tf.keras.layers.StringLookup(
            vocabulary= self.ids_from_chars.get_vocabulary(), invert=True, mask_token=None)
        self.chars = tf.strings.unicode_split(text, input_encoding='UTF-8')
        self.ids = self.ids_from_chars(self.chars)
        self.all_ids = self.ids_from_chars(tf.strings.unicode_split(text, 'UTF-8'))
        self.ids_dataset = tf.data.Dataset.from_tensor_slices(self.all_ids)
        self.seq_length = 100
        self.sequences = self.ids_dataset.batch(self.seq_length+1, drop_remainder=True)
        def split_input_target(sequence):
            input_text = sequence[:-1]
            target_text = sequence[1:]
            return input_text, target_text
        self.dataset = self.sequences.map(split_input_target)
        BATCH_SIZE = 64
        BUFFER_SIZE = 10000
        self.dataset = (
            self.dataset
            .shuffle(BUFFER_SIZE)
            .batch(BATCH_SIZE, drop_remainder=True)
            .prefetch(tf.data.experimental.AUTOTUNE))
        self.example_batch_predictions = None


    def preprocess_text(self):
        return self.all_ids

    def train_model(self, model, EPOCHS=20):
        loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
        model.compile(optimizer='adam', loss=loss)

        # Name of the checkpoint files
        checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt_{epoch}")

        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_prefix,
            save_weights_only=True)

        EPOCHS = EPOCHS
        return model.fit(self.dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])

    def save_model(self, model):
        model.save_weights(os.path.join(self.checkpoint_dir, "final_model"))

    def load_model(self):
        if os.path.exists(os.path.join(self.checkpoint_dir, "final_model")):
            model.load_weights(os.path.join(self.checkpoint_dir, "final_model"))

    def generate_text(self, gen_model, input_text, length=1000):
        states = None
        next_char = tf.constant([input_text])
        result = [next_char]

        for n in range(length):
            next_char, states = gen_model.generate_one_step(next_char, states=states)
            result.append(next_char)

        result = tf.strings.join(result)
        generated_text = result[0].numpy().decode('utf-8').replace("\r\n", " ")
        return generated_text


