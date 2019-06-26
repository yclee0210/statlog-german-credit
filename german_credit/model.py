import os
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam, SGD

def fn(correct, predicted):
    return tf.nn.softmax_cross_entropy_with_logits_v2(labels=correct,
                                                      logits=predicted)

LEARNING_RATE = 1e-4
LAYERS = 3
UNITS_PER_LAYER = 256
DROPOUT = 0.7
BATCH_SIZE = 100
NUM_EPOCHS = 500

dir_path = os.path.dirname(os.path.realpath(__file__))

class CreditNN:
    def __init__(self, input_dim, label_dim, layers=LAYERS,
                 units=UNITS_PER_LAYER, dropout=DROPOUT,
                 loss_fn=fn, optimizer=Adam(LEARNING_RATE)):
        self.input_dim = input_dim
        self.label_dim = label_dim
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.layers = layers
        self.units = units
        self.dropout = dropout

    def get_model(self):
        model = Sequential()

        for i in range(self.layers):
            input_dim = self.units if i > 0 else self.input_dim
            model.add(Dense(self.units, input_dim=input_dim, activation='relu'))
            model.add(Dropout(self.dropout))
        model.add(Dense(self.label_dim))

        return model

    def restore(self, weights_file='weights'):
        model = self.get_model()
        model.load_weights("%s/%s" % (dir_path, weights_file))
        model.compile(loss=self.loss_fn, optimizer=self.optimizer,
                      metrics=['accuracy'])

        return model

    def save(self, model, file_name='weights'):
        model.save("%s/%s" % (dir_path, file_name))

    def train(self, model, train_data, train_labels, epochs=NUM_EPOCHS,
              batch_size=BATCH_SIZE):
        model.compile(loss=self.loss_fn, optimizer=self.optimizer,
                      metrics=['accuracy'])
        model.fit(train_data, train_labels, epochs=NUM_EPOCHS, shuffle=True,
                  batch_size=50)

        return model
