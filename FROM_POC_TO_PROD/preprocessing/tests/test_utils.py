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

