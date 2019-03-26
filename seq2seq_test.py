import tensorflow as tf
import numpy as np
from tensorflow.nn import embedding_lookup

tf.enable_eager_execution()

sources = ['isso vai funcionar?']

vocab_sources = [word for sentence in sources for word in sentence.split()]

word2idx_vocab = {word:i for i,word in enumerate(vocab_sources)}

print(word2idx_vocab)


encoder_inputs = np.zeros((len(sources[0].split()), 1))

print(encoder_inputs)
embedding = tf.get_variable("embedding_encoder", [len(vocab_sources), 2])
print(embedding)
print(embedding_lookup(embedding, [0, 0, 2, 3]))
