import csv
import numpy as np
from sklearn import preprocessing

FEATURE_META = [
    ('ordinal', (14, 11, 12, 13)),
    ('numeric', None),
    ('nominal', (30, 31, 32, 33, 34)),
    ('nominal', (40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 410)),
    ('numeric', None),
    ('ordinal', (65, 61, 62, 63, 64)),
    ('ordinal', (71, 72, 73, 74, 75)),
    ('numeric', None),
    ('nominal', (91, 92, 93, 94, 95)),
    ('nominal', (101, 102, 103)),
    ('numeric', None),
    ('ordinal', (124, 123, 122, 121)),
    ('numeric', None),
    ('nominal', (141, 142, 143)),
    ('nominal', (151, 152, 153)),
    ('numeric', None),
    ('ordinal', (171, 172, 173, 174)),
    ('numeric', None),
    ('nominal', (191, 192)),
    ('nominal', (201, 202)),
    ('nominal', (1, 2))
]
NUM_LABELS = 2
ORDINAL_TO_NOMINAL = True

class Loader:
    def __init__(self):
        self.FEATURE_META = FEATURE_META
        pass

    def _read_file_to_np_array(self, filename):
        with open(filename) as file:
            reader = csv.reader(file, delimiter=' ')
            dataset = []
            for raw_row in reader:
                row = [int(cell[1:]) if cell.startswith('A') else float(cell) for cell in raw_row]
                dataset.append(row)

        self._set_normalization_values(dataset)
        dataset = self._encode_features(dataset)
        return np.array(dataset, dtype=np.float32)

    def _set_normalization_values(self, dataset):
        col_size = len(dataset[0])
        for i in range(col_size):
            if self.FEATURE_META[i][0] == 'numeric':
                col = [row[i] for row in dataset]
                self.FEATURE_META[i] = ('numeric', (np.mean(col), np.std(col)))
        return dataset

    def _encode_features(self, dataset):
        encoded_dataset = []
        encoded_labels = []
        for row in dataset:
            encoded_instance = []
            for i, col in enumerate(row):
                feature_meta = self.FEATURE_META[i]
                if feature_meta[0] == 'numeric':
                    mean = feature_meta[1][0]
                    std = feature_meta[1][1]
                    new_col = (col - mean) / std
                    encoded_instance.append(new_col)
                    # encoded_instance.append(col)
                else:
                    value_idx = feature_meta[1].index(int(col))
                    if feature_meta[0] == 'ordinal' and not ORDINAL_TO_NOMINAL:
                        encoding = [1 if j <= value_idx else 0 for j in range(len(feature_meta[1]))]
                    else:
                        # if len(feature_meta[1]) == 2:
                        #     encoding = [1 if value_idx == 1 else -1]
                        # else:
                        encoding = [1 if j == value_idx else 0 for j in range(len(feature_meta[1]))]
                    encoded_instance.extend(encoding)
            encoded_dataset.append(encoded_instance)
        return encoded_dataset

    def load(self, filename):
        dataset = self._read_file_to_np_array(filename)
        [data, labels] = np.split(dataset, [dataset.shape[1] - NUM_LABELS], axis=1)

        return data, labels
