import tensorflow as tf 
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError

class LSTMClassifier(BaseEstimator, ClassifierMixin):
  def __init__(self, embedding_weights, state_size=50, 
              input_keep_prob=0.9, output_keep_prob=0.9, 
              state_keep_prob=0.75, optimizer_class=tf.train.AdamOptimizer,
              learning_rate=0.001, batch_size=200,
              max_epochs_without_progess=20,
              random_state=None):
    '''Initialize all hyperparameters for a LSTM Classifier'''
    self.embedding_weights = embedding_weights # store a pretrained word2vec matrix
    self.state_size = state_size
    self.input_keep_prob = input_keep_prob
    self.output_keep_prob = output_keep_prob
    self.state_keep_prob = state_keep_prob
    self.learning_rate = learning_rate
    self.optimizer_class = optimizer_class
    self.batch_size = batch_size
    self.max_epochs_without_progess = max_epochs_without_progess
    self.random_state = random_state
    self._session = None

  def _build_embedding_lstm(self, inputs):
    n_inputs = inputs.shape[1]
    embedding_vectors = tf.nn.embedding_lookup(self.embedding_weights, inputs)
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.state_size)
    lstm_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell,
                                              output_keep_prob=self.output_keep_prob,
                                              input_keep_prob=self.input_keep_prob,
                                              state_keep_prob=self.state_keep_prob)

    (fw_output, bw_output), _ = tf.nn.bidirectional_dynamic_rnn(lstm_cell, lstm_cell, 
                                                                embedding_vectors, dtype=tf.float64)
    outputs = tf.concat([fw_output, bw_output], axis=2) # concatenate forward and backward outputs
    outputs = tf.unstack(outputs, n_inputs, 1) # unstack a list of time steps with length = n_inputs
    return embedding_vectors, outputs

  def _build_graph(self, n_inputs, n_outputs):
    if self.random_state is not None:
      tf.set_random_seed(self.random_state)
      np.random.seed(self.random_state)

    X = tf.placeholder(tf.int64, [None, n_inputs], name='X')
    y = tf.placeholder(tf.int64, [None], name='y')

    # Embedding and bidirectional LSTM layer
    _, lstm_outputs = self._build_embedding_lstm(X)

    # Output layer
    logits = tf.layers.dense(lstm_outputs[-1], n_outputs, name='logits') # select the last time step
    Y_proba = tf.nn.softmax(logits, name='Y_proba')
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(cross_entropy, name='loss')

    # Optimizer
    optimizer = self.optimizer_class(learning_rate=self.learning_rate)
    training_op = optimizer.minimize(loss)

    auc = tf.metrics.auc(y, Y_proba[:, 1])[1]

    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    saver = tf.train.Saver()

    # Make the important operations available easily through instance variables
    self._X, self._y = X, y 
    self._Y_proba, self._loss = Y_proba, loss 
    self._training_op, self._auc = training_op, auc 
    self._init, self._saver = init, saver 

  def close_session(self):
    if self._session:
      self._session.close()

  def _get_model_params(self):
    '''Get all variable values (used for early stopping, faster than saving to disk)'''
    with self._graph.as_default():
        gvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    return {gvar.op.name: value for gvar, value in zip(gvars, self._session.run(gvars))}

  def _restore_model_params(self, model_params):
    '''Set all variables to the given values (for early stopping, faster than loading from disk)'''
    gvar_names = list(model_params.keys())
    assign_ops = {gvar_name: self._graph.get_operation_by_name(gvar_name + "/Assign")
                  for gvar_name in gvar_names}
    init_values = {gvar_name: assign_op.inputs[1] for gvar_name, assign_op in assign_ops.items()}
    feed_dict = {init_values[gvar_name]: model_params[gvar_name] for gvar_name in gvar_names}
    self._session.run(assign_ops, feed_dict=feed_dict)
  
  def _generate_batch(self, X, y, batch_size):
    m = X.shape[0]
    shuffled_indices = np.random.permutation(m)
    X = X[shuffled_indices]
    y = y[shuffled_indices]
    for i in range(0, m, batch_size):
      X_batch = X[i:i+batch_size]
      y_batch = y[i:i+batch_size]
      yield X_batch, y_batch


  def fit(self, X, y, n_epochs=100, X_valid=None, y_valid=None):
    '''Fit the model to the training set. If X_valid and y_valid are provided, use early stopping'''
    self.close_session()

    # infer n_inputs and n_outputs from the training set
    n_inputs = X.shape[1]
    self.classes_ = np.unique(y)
    n_outputs = len(self.classes_)

    self._graph = tf.Graph()
    with self._graph.as_default():
      self._build_graph(n_inputs, n_outputs)

    # hyperparams for early stopping
    epochs_without_progress = 0
    best_loss = np.infty

    # Train the model
    self._session = tf.Session(graph=self._graph)
    with self._session.as_default() as sess:
      self._init.run()
      for epoch in range(n_epochs):
        for i, (X_batch, y_batch) in enumerate(self._generate_batch(X, y, self.batch_size)):
          sess.run(self._training_op, feed_dict={self._X: X_batch, self._y: y_batch})

        if X_valid is not None and y_valid is not None:
          loss_val, auc_val = sess.run([self._loss, self._auc], 
                                        feed_dict={self._X: X_valid, self._y: y_valid})

          if loss_val < best_loss:
            best_params = self._get_model_params()
            best_loss = loss_val 
            epochs_without_progress = 0
          else:
            epochs_without_progress += 1
          print('%s\tValidation loss: %s \tBest loss: %s \tAuc: %s' % \
                (epoch, loss_val, best_loss, auc_val))

          if epochs_without_progress > self.max_epochs_without_progess:
            print('Early stopping!')
            break 

      self._best_params = best_params
      # If we used early stopping then rollback to the best model found
      if best_params:
        self._restore_model_params(best_params)
      
      return self

  def predict_proba(self, X):
    if not self._session:
      raise NotFittedError('This %s instance is not fitted yet' % self.__class__.__name__)
    with self._session.as_default() as sess:
      return self._Y_proba.eval(feed_dict={self._X: X})

  def predict(self, X):
    return np.argmax(self.predict_proba(X), axis = 1)

  def save(self, path):
    self._saver.save(self._session, path)


# if __name__='__main__':
#   lstm_clf = LSTMClassifier(embedding_weights=embedding_matrix, random_state=42)
#   lstm_clf.fit(X_train, y_train, n_epochs=2, X_valid=X_test, y_valid=y_test)