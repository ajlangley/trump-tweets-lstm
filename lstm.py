import tensorflow as tf
import numpy as np


class LSTM:
    def __init__(self, embedding_size, layer_sizes, n_classes,
    activation=tf.nn.tanh, learning_rate=1e-3, batch_size=1):
        self.session = tf.Session()

        self.__init_embedding_layer__(embedding_size, n_classes)
        self.__init_multilayer_lstm__(layer_sizes, activation)
        self.__init_softmax_layer__(layer_sizes[-1], n_classes)

        self.x = tf.placeholder(shape=(batch_size, None),
                                dtype=tf.int32,
                                name='x_train')
        self.y = tf.placeholder(shape=(batch_size, None),
                                dtype=tf.int32,
                                name='y_train')
        lr = tf.constant(learning_rate)

        initial_state = self.lstm.zero_state(batch_size, tf.float32)
        net_output = self.feed_forward(initial_state, self.x)
        self.prediction = self.softmax_predict(net_output)

        self.loss = self.cross_ent_loss(net_output, self.y, n_classes)
        self.optimizer = tf.train.AdamOptimizer(lr).minimize(self.loss)

        init_op = tf.global_variables_initializer()
        self.session.run(init_op)

    def __init_embedding_layer__(self, embedding_size, n_classes):
        embeddings = tf.random_uniform(shape=(n_classes, embedding_size),
                                       minval=-1.0,
                                       maxval=1.0,
                                       dtype=tf.float32)

        self.embeddings = tf.Variable(initial_value=embeddings,
                                      name='embeddings')

    def __init_multilayer_lstm__(self, layer_sizes, activation):
        lstm_cells = [self.init_lstm_layer(size, i, activation)
                      for size, i in zip(layer_sizes, range(len(layer_sizes)))]
        self.lstm = tf.nn.rnn_cell.MultiRNNCell(cells=lstm_cells, state_is_tuple=True)

    def init_lstm_layer(self, state_size, layer_num, activation):
        return tf.contrib.rnn.BasicLSTMCell(num_units=state_size,
                                            state_is_tuple=True,
                                            activation=activation,
                                            name=f'lstm_layer_{layer_num}')


    def __init_softmax_layer__(self, input_size, n_classes):
        V = tf.random_normal(shape=(input_size, n_classes),
                                 stddev=0.01,
                                 dtype=tf.float32)
        b = tf.zeros(shape=(n_classes,),
                     dtype=tf.float32)

        self.V = tf.Variable(initial_value=V,
                             name='V_softmax')
        self.b = tf.Variable(initial_value=b,
                             name='b_softmax')


    def embed(self, x):
        return tf.nn.embedding_lookup(params=self.embeddings,
                                      ids=x,
                                      name='E')

    def lstm_feed_forward(self, initial_state, x):
        outputs, state = tf.nn.dynamic_rnn(cell=self.lstm,
                                           inputs=x,
                                           initial_state=initial_state,
                                           dtype=tf.float32)

        return outputs

    def softmax_feed_forward(self, x):
        return tf.matmul(a=tf.reshape(x, (-1, tf.shape(x)[-1])), b=self.V) + self.b

    def softmax_predict(self, x):
        return tf.nn.softmax(x[-1])

    def feed_forward(self, initial_state, x):
        embeddings = self.embed(x)
        lstm_output = self.lstm_feed_forward(initial_state, embeddings)
        net_output = self.softmax_feed_forward(lstm_output)

        return net_output

    def cross_ent_loss(self, net_output, labels, n_classes):
        true_dist = tf.one_hot(indices=tf.reshape(labels, [-1]), depth=n_classes)
        # net_output_flat = 
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=true_dist,
                                                          logits=net_output,
                                                          name='cross_entropy_loss')

        return tf.reduce_mean(loss)

    def sgd_step(self, x, y):
        feed_dict = {self.x: x,
                     self.y: y}

        _, loss = self.session.run([self.optimizer, self.loss], feed_dict)

        return loss

    def generate_sequence(self, min_len, max_len, start, end, index_to_class,
    class_to_index):
        x = [class_to_index[start]]

        while len(x) < max_len and x[-1] != class_to_index[end]:
            feed_dict = {self.x: np.asmatrix(x)}
            softmax_dist = self.session.run(self.prediction, feed_dict)
            # For numerical stability, make sure softmax_dist sums to 1
            softmax_dist = np.float64(softmax_dist)
            softmax_dist /= np.sum(softmax_dist)
            sample_token = np.argmax(np.random.multinomial(1, softmax_dist, 1))

            if sample_token != class_to_index[start] and not \
            (sample_token == class_to_index[end] and len(x) < min_len):
                x.append(sample_token)

        return [index_to_class[token] for token in x]
