>> predict
>>> predict
>>>> app.py:
from flask import Flask, request, render_template_string
from predict.predict.run import TextPredictionModel

app = Flask(__name__)

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>stackoverflow tags Prediction Service</title>
    <!-- Bootstrap CSS -->
    <link href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            background-color: #f7f7f7;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .container {
            margin-top: 50px;
        }
        .card {
            box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
            transition: 0.3s;
            border-radius: 5px; /* 5px rounded corners */
        }
        .card:hover {
            box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2);
        }
        .card-body {
            padding: 20px;
        }
        .card-title {
            margin-bottom: 20px;
            text-align: center;
            color: #007bff;
        }
        #predictions {
            margin-top: 20px;
            color: #333;
        }
        .footer {
            margin-top: 20px;
            text-align: center;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="card">
            <div class="card-body">
                <h3 class="card-title">stackoverflow tag Prediction</h3>
                <form action="/" method="post">
                    <div class="form-group">
                        <textarea class="form-control" name="text" rows="4" placeholder="Type your stackoverflow title here..."></textarea>
                    </div>
                    <button type="submit" class="btn btn-primary btn-block">Predict</button>
                </form>
                {% if predictions %}
                <div id="predictions" class="alert alert-success" role="alert">
                    <h4 class="alert-heading">Predictions</h4>
                    <p>{{ predictions }}</p>
                </div>
                {% endif %}
            </div>
        </div>
        <div class="footer">
            <p>stackoverflow tags Prediction Service</p>
        </div>
    </div>
    <!-- Bootstrap JS, Popper.js, and jQuery -->
    <script src="https://code.jquery.com/jquery-3.5.1.slim.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/@popperjs/core@2.9.3/dist/umd/popper.min.js"></script>
    <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js"></script>
</body>
</html>
'''


@app.route('/', methods=['GET', 'POST'])
def index():
    predictions = None
    if request.method == 'POST':
        text_list = [request.form['text']]
        model = TextPredictionModel.from_artefacts('train/data/artefacts/2024-01-06-12-46-21')
        predictions = model.predict(text_list)
    return render_template_string(HTML_TEMPLATE, predictions=predictions)

if __name__ == '__main__':
    app.run(debug=True)

run.py:

import json
import argparse
import os
import time
from collections import OrderedDict

from tensorflow.keras.models import load_model
from numpy import argsort

from preprocessing.preprocessing.embeddings import embed

import logging

logger = logging.getLogger(__name__)

class TextPredictionModel:
    def __init__(self, model, params, labels_to_index):
        self.model = model
        self.params = params
        self.labels_to_index = labels_to_index
        self.labels_index_inv = {ind: lab for lab, ind in self.labels_to_index.items()}

    @classmethod
    def from_artefacts(cls, artefacts_path: str):
        """
            from training artefacts, returns a TextPredictionModel object
            :param artefacts_path: path to training artefacts
        """
        # TODO: CODE HERE
        # load model
        #model = load_model(os.path.join(artefacts_path, "model.h5"))
        model = load_model(os.path.join(artefacts_path, "model"))
        # TODO: CODE HERE
        # load params
        with open(os.path.join(artefacts_path, "params.json"), 'r') as f:
            params = json.load(f)

        # TODO: CODE HERE
        # load labels_to_index
        with open(os.path.join(artefacts_path, "labels_index.json"), 'r') as f:
            labels_to_index = json.load(f)

        return cls(model, params, labels_to_index)

    def predict(self, text_list, top_k=5):
        """
            predict top_k tags for a list of texts
            :param text_list: list of text (questions from stackoverflow)
            :param top_k: number of top tags to predict
        """
        tic = time.time()

        logger.info(f"Predicting text_list=`{text_list}`")

        # TODO: CODE HERE
        # embed text_list
        embeddings = embed(text_list)

        # TODO: CODE HERE
        # predict tags indexes from embeddings
        tag_probabilities = self.model.predict(embeddings)

        # TODO: CODE HERE
        # from tags indexes compute top_k tags for each text
        predictions = []
        for probs in tag_probabilities:
            top_indices = argsort(probs)[-top_k:][::-1]
            top_tags = [self.labels_index_inv[idx] for idx in top_indices]
            predictions.append(top_tags)

        logger.info("Prediction done in {:2f}s".format(time.time() - tic))

        return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("artefacts_path", help="path to trained model artefacts")
    parser.add_argument("text", type=str, default=None, help="text to predict")
    args = parser.parse_args()

    logging.basicConfig(format="%(name)s - %(levelname)s - %(message)s", level=logging.INFO)

    model = TextPredictionModel.from_artefacts(args.artefacts_path)

    if args.text is None:
        while True:
            txt = input("Type the text you would like to tag: ")
            predictions = model.predict([txt])
            print(predictions)
    else:
        print(f'Predictions for `{args.text}`')
        print(model.predict([args.text]))

>>> tests
>>>>test_predict.py:
from predict.predict.run import TextPredictionModel
import unittest
from unittest.mock import MagicMock
import tempfile
import os
import pandas as pd
import json
import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')
from train.train import run
from preprocessing.preprocessing import utils

def load_dataset_mock():
    titles = [
        "Is it possible to execute the procedure of a function in the scope of the caller?",
        "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
        "Is it possible to execute the procedure of a function in the scope of the caller?",
        "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
        "Is it possible to execute the procedure of a function in the scope of the caller?",
        "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
        "Is it possible to execute the procedure of a function in the scope of the caller?",
        "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
        "Is it possible to execute the procedure of a function in the scope of the caller?",
        "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
    ]
    tags = ["php", "ruby-on-rails", "php", "ruby-on-rails", "php", "ruby-on-rails", "php", "ruby-on-rails",
            "php", "ruby-on-rails"]

    return pd.DataFrame({
        'title': titles,
        'tag_name': tags
    })


class TestPredict(unittest.TestCase):
    def test_predict(self):
        utils.LocalTextCategorizationDataset.load_dataset = MagicMock(return_value=load_dataset_mock())
        params = {
            'batch_size': 2,
            'epochs': 5,
            'dense_dim': 64,
            'min_samples_per_label': 5,
            'verbose': 1
        }

        with tempfile.TemporaryDirectory() as model_dir:
            accuracy, artefact = run.train('chemin_fictif_pour_dataset', params, model_dir, False)

            model = TextPredictionModel.from_artefacts(artefact)
            test_text = "Is it possible to execute the procedure of a function in the scope of the caller?"
            predicted_label = model.predict([test_text])[0]
            print(predicted_label)
            self.assertIn("php", predicted_label)

            """
            predicted_labels = model.predict([test_text])
            print(predicted_labels)
            with open(os.path.join(artefact, 'labels_index.json'), 'r') as f:
                label_to_index_map = json.load(f)
            print(label_to_index_map)
            predicted_label_names = [label_to_index_map[str(label)] for label in predicted_labels[0]]
            print(predicted_label_names)
            # Test if the expected label is in the predictions
            self.assertIn("php", predicted_label_names)
            #self.assertEqual(expected_label, "php")
            self.assertEqual(accuracy, 1.0)
            """
>> preprocessing
>>> preprocessing
>>>> embeddings.py:
from functools import lru_cache
import numpy as np

from transformers import TFBertModel, BertTokenizer
import tensorflow as tf


@lru_cache(maxsize=1)
def get_embedding_model():
    # First time this is executed, the model is downloaded. Actually, the download process should be done
    # a side, from a controlled version of the model, included, for instance in the docker build process.
    model = TFBertModel.from_pretrained("bert-base-uncased", output_hidden_states=True)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    return model, tokenizer


def embed(texts):
    model, tokenizer = get_embedding_model()

    embeddings = []
    for text in texts:
        tokens = tokenizer.encode(text, add_special_tokens=True)
        tokens = tf.constant(tokens)[None, :]
        outputs = model(tokens)
        embeddings.append(outputs[1][0])

    return np.array(embeddings)
>>>> utils.py:

import numpy as np
import pandas as pd

from keras.utils import Sequence
from keras.utils import to_categorical

from sklearn.model_selection import train_test_split


def integer_floor(float_value: float):
    """
    link to doc for numpy.floor https://numpy.org/doc/stable/reference/generated/numpy.floor.html
    """
    return int(np.floor(float_value))


class _SimpleSequence(Sequence):
    """
    Base object for fitting to a sequence of data, such as a dataset.
    link to doc : https://www.tensorflow.org/api_docs/python/tf/keras/utils/Sequence
    """

    def __init__(self, get_batch_method, num_batches_method):
        self.get_batch_method = get_batch_method
        self.num_batches_method = num_batches_method

    def __len__(self):
        return self.num_batches_method()

    def __getitem__(self, idx):
        return self.get_batch_method()


class BaseTextCategorizationDataset:
    """
    Generic class for text categorization
    data sequence generation
    """

    def __init__(self, batch_size, train_ratio=0.8):
        assert train_ratio < 1.0
        self.train_ratio = train_ratio
        self.batch_size = batch_size

    def _get_label_list(self):
        """
        returns list of labels
        should not be implemented in this class (we can assume its a given)
        """
        raise NotImplementedError

    def get_num_labels(self):
        """
        returns the number of labels
        """
        # TODO: CODE HERE
        return len(self._get_label_list())

    def _get_num_samples(self):
        """
        returns number of samples (dataset size)
        should not be implemented in this class (we can assume its a given)
        """
        raise NotImplementedError

    def _get_num_train_samples(self):
        """
        returns number of train samples
        (training set size)
        """
        # TODO: CODE HERE
        return integer_floor(self._get_num_samples() * self.train_ratio)

    def _get_num_test_samples(self):
        """
        returns number of test samples
        (test set size)
        """
        # TODO: CODE HERE
        return self._get_num_samples() - self._get_num_train_samples()

    def _get_num_train_batches(self):
        """
        returns number of train batches
        """
        # TODO: CODE HERE
        return integer_floor(self._get_num_train_samples() / self.batch_size)

    def _get_num_test_batches(self):
        """
        returns number of test batches
        """
        # TODO: CODE HERE
        return integer_floor(self._get_num_test_samples() / self.batch_size)

    def get_train_batch(self):
        """
        returns next train batch
        should not be implemented in this class (we can assume its a given)
        """
        raise NotImplementedError

    def get_test_batch(self):
        """
        returns next test batch
        should not be implemented in this class (we can assume its a given)
        """
        raise NotImplementedError

    def get_index_to_label_map(self):
        """
        from label list, returns a map index -> label
        (dictionary index: label)
        """
        # TODO: CODE HERE
        return {i: label for i, label in enumerate(self._get_label_list())}

    def get_label_to_index_map(self):
        """
        from index -> label map, returns label -> index map
        (reverse the previous dictionary)
        """
        # TODO: CODE HERE
        return {label: i for i, label in self.get_index_to_label_map().items()}

    def to_indexes(self, labels):
        """
        from a list of labels, returns a list of indexes
        """
        # TODO: CODE HERE
        return [self.get_label_to_index_map()[label] for label in labels]

    def get_train_sequence(self):
        """
        returns a train sequence of type _SimpleSequence
        """
        return _SimpleSequence(self.get_train_batch, self._get_num_train_batches)

    def get_test_sequence(self):
        """
        returns a test sequence of type _SimpleSequence
        """
        # TODO: CODE HERE
        return _SimpleSequence(self.get_test_batch, self._get_num_test_batches)

    def __repr__(self):
        return self.__class__.__name__ + \
            f"(n_train_samples: {self._get_num_train_samples()}, " \
            f"n_test_samples: {self._get_num_test_samples()}, " \
            f"n_labels: {self.get_num_labels()})"


class LocalTextCategorizationDataset(BaseTextCategorizationDataset):
    """
    A TextCategorizationDataset read from a file residing in the local filesystem
    """

    def __init__(self, filename, batch_size,
                 train_ratio=0.8, min_samples_per_label=100, preprocess_text=lambda x: x):
        """
        :param filename: a CSV file containing the text samples in the format
            (post_id 	tag_name 	tag_id 	tag_position 	title)
        :param batch_size: number of samples per batch
        :param train_ratio: ratio of samples dedicated to training set between (0, 1)
        :param preprocess_text: function taking an array of text and returning a numpy array, default identity
        """
        super().__init__(batch_size, train_ratio)
        self.filename = filename
        self.preprocess_text = preprocess_text

        self._dataset = self.load_dataset(self.filename, min_samples_per_label)

        assert self._get_num_train_batches() > 0
        assert self._get_num_test_batches() > 0

        # TODO: CODE HERE
        # from self._dataset, compute the label list
        self._label_list = list(set(self._dataset['tag_name']))

        y = self.to_indexes(self._dataset['tag_name'])
        y = to_categorical(y, num_classes=len(self._label_list))

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self._dataset['title'],
            y,
            train_size=self._get_num_train_samples(),
            stratify=y)

        self.train_batch_index = 0
        self.test_batch_index = 0

    @staticmethod
    def load_dataset(filename, min_samples_per_label):
        """
        loads dataset from filename apply pre-processing steps (keeps only tag_position = 0 & removes tags that were
        seen less than `min_samples_per_label` times)
        """
        # reading dataset from filename path, dataset is csv
        # TODO: CODE HERE
        df = pd.read_csv(filename)

        # assert that columns are the ones expected
        # TODO: CODE HERE
        expected_columns = ['post_id', 'tag_name', 'tag_id', 'tag_position', 'title']
        assert all(col in df.columns for col in expected_columns), "Columns do not match expected columns."

        def filter_tag_position(position):
            def filter_function(df):
                """
                keep only tag_position = position
                """
                # TODO: CODE HERE
                return df[df['tag_position'] == position]

            return filter_function

        def filter_tags_with_less_than_x_samples(x):
            def filter_function(df):
                """
                removes tags that are seen less than x times
                """
                # TODO: CODE HERE
                tag_counts = df['tag_name'].value_counts()
                tags_to_keep = tag_counts[tag_counts >= x].index
                return df[df['tag_name'].isin(tags_to_keep)]

            return filter_function

        # use pandas.DataFrame.pipe to chain preprocessing steps
        # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.pipe.html
        # return pre-processed dataset
        # TODO: CODE HERE
        pre_processed_df = (
            df.pipe(filter_tag_position(0))
            .pipe(filter_tags_with_less_than_x_samples(min_samples_per_label))
        )

        return pre_processed_df

    # we need to implement the methods that are not implemented in the super class BaseTextCategorizationDataset

    def _get_label_list(self):
        """
        returns label list
        """
        # TODO: CODE HERE
        return self._label_list

    def _get_num_samples(self):
        """
        returns number of samples in dataset
        """
        # TODO: CODE HERE
        return len(self._dataset)

    def get_train_batch(self):
        i = self.train_batch_index
        # TODO: CODE HERE
        # takes x_train between i * batch_size to (i + 1) * batch_size, and apply preprocess_text
        next_x = self.preprocess_text(self.x_train[i * self.batch_size: (i + 1) * self.batch_size])
        # TODO: CODE HERE
        # takes y_train between i * batch_size to (i + 1) * batch_size
        next_y = self.y_train[i * self.batch_size: (i + 1) * self.batch_size]
        # When we reach the max num batches, we start anew
        self.train_batch_index = (self.train_batch_index + 1) % self._get_num_train_batches()
        return next_x, next_y

    def get_test_batch(self):
        """
        it does the same as get_train_batch for the test set
        """
        # TODO: CODE HERE
        i = self.test_batch_index
        next_x = self.preprocess_text(self.x_test[i * self.batch_size: (i + 1) * self.batch_size])
        next_y = self.y_test[i * self.batch_size: (i + 1) * self.batch_size]
        # When we reach the max num batches, we start anew
        self.test_batch_index = (self.test_batch_index + 1) % self._get_num_test_batches()
        return next_x, next_y

>>> tests
>>>>tests_embeddings.py:
import unittest

from preprocessing.preprocessing.embeddings import embed


class EmbeddingsTest(unittest.TestCase):
    def test_embed(self):
        embeddings = embed(['hello world'])
        self.assertEqual(embeddings.shape, (1, 768))

>>>> test_utils.py:

import unittest
import pandas as pd
from unittest.mock import MagicMock

import sys
import os

# Ajoutez le répertoire parent à sys.path pour permettre l'importation de preprocessing

from preprocessing.preprocessing import utils


class TestBaseTextCategorizationDataset(unittest.TestCase):
    def test__get_num_train_samples(self):
        """
        we want to test the class BaseTextCategorizationDataset
        we use a mock which will return a value for the not implemented methods
        then with this mocked value, we can test other methods
        """
        # we instantiate a BaseTextCategorizationDataset object with batch_size = 20 and train_ratio = 0.8
        base = utils.BaseTextCategorizationDataset(20, 0.8)
        # we mock _get_num_samples to return the value 100
        base._get_num_samples = MagicMock(return_value=100)
        # we assert that _get_num_train_samples will return 100 * train_ratio = 80
        self.assertEqual(base._get_num_train_samples(), 80)

    def test__get_num_train_batches(self):
        """
        Tests the _get_num_train_batches method.
        """
        base = utils.BaseTextCategorizationDataset(20, 0.8)
        base._get_num_samples = MagicMock(return_value=100)
        self.assertEqual(base._get_num_train_batches(), 4)

    def test__get_num_test_batches(self):
        """
        Tests the _get_num_test_batches method.
        """
        base = utils.BaseTextCategorizationDataset(20, 0.8)
        base._get_num_samples = MagicMock(return_value=100)
        self.assertEqual(base._get_num_test_batches(), 1)

    def test_get_index_to_label_map(self):
        """
        Tests the get_index_to_label_map method.
        """
        base = utils.BaseTextCategorizationDataset(20, 0.8)
        base._get_label_list = MagicMock(return_value=['php', 'java', '.net'])
        expected_mapping = {0: 'php', 1: 'java', 2: '.net'}
        self.assertDictEqual(base.get_index_to_label_map(), expected_mapping)

    def test_index_to_label_and_label_to_index_are_identity(self):
        """
        Tests if index_to_label_map and label_to_index_map are inverse of each other.
        """
        base = utils.BaseTextCategorizationDataset(20, 0.8)
        base._get_label_list = MagicMock(return_value=['php', 'java', '.net'])
        index_to_label_map = base.get_index_to_label_map()
        label_to_index_map = base.get_label_to_index_map()
        self.assertDictEqual(index_to_label_map, {v: k for k, v in label_to_index_map.items()})

    def test_to_indexes(self):
        """
        Tests the to_indexes method.
        """
        base = utils.BaseTextCategorizationDataset(20, 0.8)
        base._get_label_list = MagicMock(return_value=['php', 'java', '.net'])
        base.get_label_to_index_map = MagicMock(return_value={'php': 0, 'java': 1, '.net': 2})
        labels = ['php', 'java', '.net', 'php']
        expected_indexes = [0, 1, 2, 0]
        self.assertEqual(base.to_indexes(labels), expected_indexes)


class TestLocalTextCategorizationDataset(unittest.TestCase):
    def test_load_dataset_returns_expected_data(self):
        # we mock pandas read_csv to return a fixed dataframe
        pd.read_csv = MagicMock(return_value=pd.DataFrame({
            'post_id': ['id_1', 'id_2'],
            'tag_name': ['tag_a', 'tag_b'],
            'tag_id': [1, 2],
            'tag_position': [0, 1],
            'title': ['title_1', 'title_2']
        }))
        # we instantiate a LocalTextCategorizationDataset (it'll use the mocked read_csv), and we load dataset
        dataset = utils.LocalTextCategorizationDataset.load_dataset("fake_path", 1)
        # we expect the data after loading to be like this
        expected = pd.DataFrame({
            'post_id': ['id_1'],
            'tag_name': ['tag_a'],
            'tag_id': [1],
            'tag_position': [0],
            'title': ['title_1']
        })
        # we confirm that the dataset and what we expected to be are the same thing
        pd.testing.assert_frame_equal(dataset, expected)

    def test__get_num_samples_is_correct(self):
        """
        Tests if _get_num_samples returns the correct number of samples.
        """
        pd.read_csv = MagicMock(return_value=pd.DataFrame({
            'post_id': ['id_1', 'id_2', 'id_3', 'id_4'],
            'tag_name': ['tag_a', 'tag_a', 'tag_c', 'tag_d'],
            'tag_id': [1, 2, 3, 4],
            'tag_position': [0, 0, 2, 3],
            'title': ['title_1', 'title_2', 'title_3', 'title_ 4']
        }))

        local_dataset = utils.LocalTextCategorizationDataset("fake_path", 1, min_samples_per_label=1, train_ratio=0.5)

        self.assertEqual(local_dataset._get_num_samples(), 2)

    def test_get_train_batch_returns_expected_shape(self):
        """
        Tests if get_train_batch returns the expected shape.
        """
        pd.read_csv = MagicMock(return_value=pd.DataFrame({
            'post_id': ['id_1', 'id_2', 'id_3', 'id_4'],
            'tag_name': ['tag_a', 'tag_a', 'tag_c', 'tag_d'],
            'tag_id': [1, 2, 3, 4],
            'tag_position': [0, 0, 2, 3],
            'title': ['title_1', 'title_2', 'title_3', 'title_ 4']
        }))

        local_dataset = utils.LocalTextCategorizationDataset("fake_path", 1, min_samples_per_label=1, train_ratio=0.5)
        batch_x, batch_y = local_dataset.get_train_batch()
        self.assertEqual(len(batch_x), 1)
        self.assertEqual(len(batch_y), 1)

    def test_get_test_batch_returns_expected_shape(self):
        """
        Tests if get_test_batch returns the expected shape.
        """
        pd.read_csv = MagicMock(return_value=pd.DataFrame({
            'post_id': ['id_1', 'id_2', 'id_3', 'id_4'],
            'tag_name': ['tag_a', 'tag_a', 'tag_c', 'tag_d'],
            'tag_id': [1, 2, 3, 4],
            'tag_position': [0, 0, 2, 3],
            'title': ['title_1', 'title_2', 'title_3', 'title_ 4']
        }))

        local_dataset = utils.LocalTextCategorizationDataset("fake_path", 1, min_samples_per_label=1, train_ratio=0.5)
        batch_x, batch_y = local_dataset.get_test_batch()
        self.assertEqual(len(batch_x), 1)
        self.assertEqual(len(batch_y), 1)

    def test_get_train_batch_raises_assertion_error(self):
        """
        Tests if get_train_batch raises an assertion error when train_batch_index exceeds num_train_batches.
        """

        pd.read_csv = MagicMock(return_value=pd.DataFrame({
            'post_id': ['id_1', 'id_2', 'id_3', 'id_4'],
            'tag_name': ['tag_a', 'tag_b', 'tag_c', 'tag_d'],
            'tag_id': [1, 2, 3, 4],
            'tag_position': [0, 1, 2, 3],
            'title': ['title_1', 'title_2', 'title_3', 'title_ 4']
        }))
        with self.assertRaises(AssertionError):
            utils.LocalTextCategorizationDataset("fake_path", 20, min_samples_per_label=1, train_ratio=0.5)


if __name__ == '__main__':
    unittest.main()

>>train
>>> conf
>>>> train-conf.yml
batch_size: 32
epochs: 1
dense_dim: 64
min_samples_per_label: 10
verbose: 1

>>> data
>>>>traing-data
>>>> artefacts

>>>tests
>>>> test_model_train.py:

import unittest
from unittest.mock import MagicMock
import tempfile

import pandas as pd

from train.train import run
from preprocessing.preprocessing import utils


def load_dataset_mock():
    titles = [
        "Is it possible to execute the procedure of a function in the scope of the caller?",
        "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
        "Is it possible to execute the procedure of a function in the scope of the caller?",
        "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
        "Is it possible to execute the procedure of a function in the scope of the caller?",
        "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
        "Is it possible to execute the procedure of a function in the scope of the caller?",
        "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
        "Is it possible to execute the procedure of a function in the scope of the caller?",
        "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
    ]
    tags = ["php", "ruby-on-rails", "php", "ruby-on-rails", "php", "ruby-on-rails", "php", "ruby-on-rails",
            "php", "ruby-on-rails"]

    return pd.DataFrame({
        'title': titles,
        'tag_name': tags
    })


class TestTrain(unittest.TestCase):
    # TODO: CODE HERE
    # use the function defined above as a mock for utils.LocalTextCategorizationDataset.load_dataset
    utils.LocalTextCategorizationDataset.load_dataset = MagicMock(return_value=load_dataset_mock())

    def test_train(self):
        # TODO: CODE HERE
        # create a dictionary params for train conf
        params = {
            'batch_size': 2,
            'epochs': 5,
            'dense_dim': 64,
            'min_samples_per_label': 5,
            'verbose': 1
        }

        # we create a temporary file to store artefacts
        with tempfile.TemporaryDirectory() as model_dir:
            # run a training
            accuracy, _ = run.train('chemin_fictif_pour_dataset', params, model_dir, False)

        # TODO: CODE HERE
        # assert that accuracy is equal to 1.0
        self.assertEqual(accuracy, 1.0)
>>> train

>>>> test_model_train.py:
import unittest
from unittest.mock import MagicMock
import tempfile

import pandas as pd

from train.train import run
from preprocessing.preprocessing import utils


def load_dataset_mock():
    titles = [
        "Is it possible to execute the procedure of a function in the scope of the caller?",
        "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
        "Is it possible to execute the procedure of a function in the scope of the caller?",
        "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
        "Is it possible to execute the procedure of a function in the scope of the caller?",
        "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
        "Is it possible to execute the procedure of a function in the scope of the caller?",
        "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
        "Is it possible to execute the procedure of a function in the scope of the caller?",
        "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
    ]
    tags = ["php", "ruby-on-rails", "php", "ruby-on-rails", "php", "ruby-on-rails", "php", "ruby-on-rails",
            "php", "ruby-on-rails"]

    return pd.DataFrame({
        'title': titles,
        'tag_name': tags
    })


class TestTrain(unittest.TestCase):
    # TODO: CODE HERE
    # use the function defined above as a mock for utils.LocalTextCategorizationDataset.load_dataset
    utils.LocalTextCategorizationDataset.load_dataset = MagicMock(return_value=load_dataset_mock())

    def test_train(self):
        # TODO: CODE HERE
        # create a dictionary params for train conf
        params = {
            'batch_size': 2,
            'epochs': 5,
            'dense_dim': 64,
            'min_samples_per_label': 5,
            'verbose': 1
        }

        # we create a temporary file to store artefacts
        with tempfile.TemporaryDirectory() as model_dir:
            # run a training
            accuracy, _ = run.train('chemin_fictif_pour_dataset', params, model_dir, False)

        # TODO: CODE HERE
        # assert that accuracy is equal to 1.0
        self.assertEqual(accuracy, 1.0)
>>> train
>>>> run.py:

import os
import json
import argparse
import time
import logging
from keras.models import Sequential
from keras.layers import Dense
from preprocessing.preprocessing.embeddings import embed
from preprocessing.preprocessing.utils import LocalTextCategorizationDataset

logger = logging.getLogger(__name__)


def train(dataset_path, train_conf, model_path, add_timestamp):
    """
    :param dataset_path: path to a CSV file containing the text samples in the format
            (post_id 	tag_name 	tag_id 	tag_position 	title)
    :param train_conf: dictionary containing training parameters, example :
            {
                batch_size: 32
                epochs: 1
                dense_dim: 64
                min_samples_per_label: 10
                verbose: 1
            }
    :param model_path: path to folder where training artefacts will be persisted
    :param add_timestamp: boolean to create artefacts in a sub folder with name equal to execution timestamp
    """

    # if add_timestamp then add sub folder with name equal to execution timestamp '%Y-%m-%d-%H-%M-%S'
    if add_timestamp:
        artefacts_path = os.path.join(model_path, time.strftime('%Y-%m-%d-%H-%M-%S'))
    else:
        artefacts_path = model_path

    # TODO: CODE HERE
    # instantiate a LocalTextCategorizationDataset, use embed method from preprocessing module for preprocess_text param
    # use train_conf for other needed params
    dataset = LocalTextCategorizationDataset(
        filename=dataset_path,
        batch_size=train_conf['batch_size'],
        train_ratio=0.8,
        min_samples_per_label=train_conf['min_samples_per_label'],
        preprocess_text=embed
    )

    logger.info(dataset)

    # TODO: CODE HERE
    # instantiate a sequential keras model
    # add a dense layer with relu activation
    # add an output layer (multiclass classification problem)
    model = Sequential()
    model.add(
        Dense(train_conf['dense_dim'], activation='relu', input_shape=(768,)))
    model.add(
        Dense(dataset.get_num_labels(), activation='softmax'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # TODO: CODE HERE
    # model fit using data sequences
    train_history = model.fit(
        dataset.get_train_sequence(),
        epochs=train_conf['epochs'],
        verbose=train_conf['verbose']
    )

    # scores
    #scores = model.evaluate_generator(dataset.get_test_sequence(), verbose=0)
    scores = model.evaluate(dataset.get_test_sequence(), verbose=0)
    logger.info("Test Accuracy: {:.2f}".format(scores[1] * 100))

    # TODO: CODE HERE
    # create folder artefacts_path
    os.makedirs(artefacts_path, exist_ok=True)

    # TODO: CODE HERE
    # save model in artefacts folder, name model.h5
    model.save(os.path.join(artefacts_path, "model"))
    print(f"model saved at {artefacts_path}")
    #model.save(os.path.join(artefacts_path, "model.h5"))
    # TODO: CODE HERE
    # save train_conf used in artefacts_path/params.json
    with open(os.path.join(artefacts_path, "params.json"), "w") as f:
        json.dump(train_conf, f)
    # TODO: CODE HERE
    # save labels index in artefacts_path/labels_index.json
    with open(os.path.join(artefacts_path, "labels_index.json"), "w") as f:
        json.dump(dataset.get_label_to_index_map(), f)

    # train_history.history is not JSON-serializable because it contains numpy arrays
    serializable_hist = {k: [float(e) for e in v] for k, v in train_history.history.items()}
    with open(os.path.join(artefacts_path, "train_output.json"), "w") as f:
        json.dump(serializable_hist, f)

    return scores[1], artefacts_path


if __name__ == "__main__":
    import yaml

    parser = argparse.ArgumentParser()

    parser.add_argument("dataset_path", help="Path to training dataset")
    parser.add_argument("config_path", help="Path to Yaml file specifying training parameters")
    parser.add_argument("artefacts_path", help="Folder where training artefacts will be persisted")
    #parser.add_argument("add_timestamp", action='store_true',
    #                   help="Create artefacts in a sub folder with name equal to execution timestamp")
    parser.add_argument("--add_timestamp", action='store_true',
                        help="Create artefacts in a sub folder with name equal to execution timestamp")

    args = parser.parse_args()

    with open(args.config_path, 'r') as config_f:
        train_params = yaml.safe_load(config_f.read())

    logger.info(f"Training model with parameters: {train_params}")

    train(args.dataset_path, train_params, args.artefacts_path, args.add_timestamp)



