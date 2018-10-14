import sys
import sklearn.model_selection as sk

from german_credit.loader import Loader
from german_credit.model import CreditNN

DEFAULT_FILENAME = "german.data"

def train(dataset):
    print(dataset.shape)
    pass

if __name__ == "__main__":
    filename = sys.argv[1] if 1 < len(sys.argv) else DEFAULT_FILENAME
    loader = Loader()

    data, labels = loader.load(filename)
    nn = CreditNN(data.shape[1], labels.shape[1])
    model = nn.get_model()

    train_data, test_data, train_labels, test_labels = sk.train_test_split(data,
                                                                           labels,
                                                                           test_size=0.1)
    model = nn.train(model, train_data, train_labels)
    scores = model.evaluate(test_data, test_labels)
    print("Training Complete.")
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
