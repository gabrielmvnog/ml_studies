import tensorflow as tf


# Embedding Encoder & Embedding Decoder

embedding_encoder = variable_scope.get_variable(
    "embedding_encoder", [src_vocab_size, embedding_size], ...)
encoder_emb_inp = embedding_ops.embedding_lookup(
    embedding_encoder, encoder_inputs)

embedding_decoder = variable_scope.get_variable(
    "embedding_encoder", [tgt_vocab_size, embedding_size], ...)
decoder_emb_inp = embedding_ops.embedding_lookup(
    embedding_decoder, decoder_inputs)


# Encoder

encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)
encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder_cell, encoder_emb_inp,
                                                   sequence_length=source_sequence_length, time_major=True)

# decoder

decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)

helper = tf.contrib.seq2seq.TrainingHelper(
    decoder_emb_inp, decoder_lengths, time_major=True)

decoder = tf.contrib.seq2seq.BasicDecoder(
    decoder_cell, helper, encoder_state,
    output_layer=projection_layer)

outputs, _ = tf.contrib.seq2seq.dynamic_decode(decoder, ...)
logits = outputs.rnn_output

projection_layer = layers_core.Dense(
    tgt_vocab_size, use_bias=False)

# Loss

crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
    labels=decoder_outputs, logits=logits)
train_loss = (tf.reduce_sum(crossent * target_weights) /
    batch_size)

# Gradient computation 

params = tf.trainable_variables()
gradients = tf.gradients(train_loss, params)
clipped_gradients, _ = tf.clip_by_global_norm(
    gradients, max_gradient_norm)

# Optimiazation

optimizer = tf.train.AdamOptimizer(learning_rate)
update_step = optimizer.apply_gradients(zip(clipped_gradients, params))


# Translation

maximum_iterations = tf.round(tf.reduce_max(source_sequence_length) * 2)

helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
    embedding_decoder,
    tf.fill([batch_size], tgt_sos_id), tgt_eos_id)

decoder = tf.contrib.seq2seq.BasicDecoder(
    decoder_cell, helper, encoder_state,
    output_layer=projection_layer)

outputs, _ = tf.contrib.seq2seq.dynamic_decode(
    decoder, maximum_iterations=maximum_iterations)
translations = outputs.sample_id