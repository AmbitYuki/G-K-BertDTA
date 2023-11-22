import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling1D, Reshape, Permute, multiply

class SelfAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(SelfAttention, self).__init__()
        self.units = units
        self.W_q = Dense(units)
        self.W_k = Dense(units)
        self.W_v = Dense(units)

    def call(self, inputs):
        Q = self.W_q(inputs)
        K = self.W_k(inputs)
        V = self.W_v(inputs)
        attention_scores = tf.matmul(Q, K, transpose_b=True)
        attention_scores = tf.keras.activations.softmax(attention_scores, axis=-1)

        output = tf.matmul(attention_scores, V)

        return output

input_sequence = tf.keras.layers.Input(shape=(None, 64))  
attention_output = SelfAttention(units=64)(input_sequence)

output = GlobalAveragePooling1D()(attention_output)

output = Dense(32, activation='relu')(output)
output = Dense(1, activation='sigmoid')(output)

model = tf.keras.models.Model(inputs=input_sequence, outputs=output)


import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling1D, Reshape, Permute, Concatenate, MultiHeadAttention

class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, units, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.units = units
        self.num_heads = num_heads

        assert units % num_heads == 0
        self.depth = units // num_heads

        self.attention_heads = [SelfAttention(units=self.depth) for _ in range(num_heads)]
        self.output_dense = Dense(units)

    def call(self, inputs):
        attention_heads_outputs = [attention_head(inputs) for attention_head in self.attention_heads]

        concatenated_attention = Concatenate(axis=-1)(attention_heads_outputs)

        output = self.output_dense(concatenated_attention)

        return output

input_sequence = tf.keras.layers.Input(shape=(None, 64)) 
multi_head_attention_output = MultiHeadSelfAttention(units=64, num_heads=8)(input_sequence)

output = GlobalAveragePooling1D()(multi_head_attention_output)

output = Dense(32, activation='relu')(output)
output = Dense(1, activation='sigmoid')(output)

model = tf.keras.models.Model(inputs=input_sequence, outputs=output)