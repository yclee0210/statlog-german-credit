import sys
import numpy as np
import sklearn.model_selection as sk
from sklearn.model_selection import KFold
from keras.optimizers import Adam, SGD

from german_credit.loader import Loader
from german_credit.model import CreditNN

DEFAULT_FILENAME = "german.data"
LEARNING_RATE = 1e-4
LAYERS = 3
UNITS = 256
DROPOUT = 0.7
BATCH_SIZE = 100
EPOCHS = 500

def train(dataset):
    print(dataset.shape)
    pass

def train_kfold(data, labels, fold=10, learning_rate=1e-4, layers=3, units=256,
                dropout=0.7, batch_size=100, epochs=500):

    kf = KFold(n_splits=fold, shuffle=True)
    folds = kf.split(data)
    history = []
    for train_idx, test_idx in folds:
        train_data = data[train_idx]
        train_labels = labels[train_idx]
        test_data = data[test_idx]
        test_labels = labels[test_idx]
        nn = CreditNN(train_data.shape[1], train_labels.shape[1], layers=layers,
                      units=units, dropout=dropout, optimizer=Adam(learning_rate))
        model = nn.train(nn.get_model(), train_data, train_labels,
                         epochs=epochs, batch_size=batch_size)
        scores = model.evaluate(test_data, test_labels)
        history.append(scores)
    return np.mean(history, axis=0)

if __name__ == "__main__":
    filename = sys.argv[1] if 1 < len(sys.argv) else DEFAULT_FILENAME
    loader = Loader()
    data, labels = loader.load(filename)

    # 10-fold cross validation
    # print(train_kfold(data, labels))

    # Alternatively, generate nn trained on full dataset
    nn = CreditNN(data.shape[1], labels.shape[1], layers=LAYERS,
                  units=UNITS, dropout=DROPOUT, optimizer=Adam(LEARNING_RATE))
    model = nn.train(nn.get_model(), data, labels,
                     epochs=EPOCHS, batch_size=BATCH_SIZE)
    nn.save(model, 'weights')
