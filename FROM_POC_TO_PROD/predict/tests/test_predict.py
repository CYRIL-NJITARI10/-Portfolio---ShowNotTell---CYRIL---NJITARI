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







