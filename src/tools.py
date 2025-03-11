import pandas as pd
import numpy as np

class Dataset:
    """
    A class to handle dataset loading and preprocessing.

    Attributes:
        df (pd.DataFrame): DataFrame containing the input data.
        target (pd.DataFrame or None): DataFrame containing target labels for training mode, None for test mode.
    """
    def __init__(self, train_idx=0, mode='train'):
        """
        Initializes the Dataset class by loading the corresponding CSV files.

        Args:
            train_idx (int, optional): Index of the dataset partition (default is 0).
            mode (str, optional): Mode of operation ('train' or 'test', default is 'train').
        """
        if mode == 'train':
            self.df = pd.read_csv(f'data/Xtr{train_idx}.csv', index_col=0)
            self.target = pd.read_csv(f'data/Ytr{train_idx}.csv', index_col=0)
        else:
            self.df = pd.read_csv(f'data/Xte{train_idx}.csv', index_col=0)
            self.target = None

    def train_test_split(self, test_size=0.2, shuffle=True):
        """
        Splits the dataset into training and test sets.

        Args:
            test_size (float, optional): Proportion of the dataset to include in the test split (default is 0.2).
            shuffle (bool, optional): Whether to shuffle the dataset before splitting (default is True).

        Returns:
            tuple: (train_df, test_df, train_target, test_target)
                - train_df (np.ndarray): Training feature data.
                - test_df (np.ndarray): Test feature data.
                - train_target (np.ndarray or None): Training target labels, transformed to {-1, 1} if applicable.
                - test_target (np.ndarray or None): Test target labels, transformed to {-1, 1} if applicable.
        """
        n = len(self.df)
        indices = np.arange(n)
        if shuffle:
            np.random.shuffle(indices)
        n_test = int(n * test_size)
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]
        train_df = self.df.iloc[train_indices].seq.to_numpy()
        test_df = self.df.iloc[test_indices].seq.to_numpy()
        if self.target is not None:
            train_target = 2 * self.target.iloc[train_indices].Bound.to_numpy() - 1
            test_target = 2 * self.target.iloc[test_indices].Bound.to_numpy() - 1
        else:
            train_target = None
            test_target = None
        return train_df, test_df, train_target, test_target
    
    def get(self, shuffle=True):
        """
        Returns the entire dataset as numpy arrays, optionally shuffled.

        Args:
            shuffle (bool, optional): Whether to shuffle the dataset before returning (default is True).

        Returns:
            tuple: (X, y)
                - X (np.ndarray): Feature data.
                - y (np.ndarray or None): Target labels transformed to {-1, 1} if applicable.
        """
        n = len(self.df)
        indices = np.arange(n)
        if shuffle:
            np.random.shuffle(indices)
        X = self.df.iloc[indices].seq.to_numpy()
        if self.target is not None:
            y = 2 * self.target.iloc[indices].Bound.to_numpy() - 1
        else:
            y = None
        return X, y

class Score:
    """
    Class to compute the accuracy, precision and recall of a classifier.

    Parameters
    ----------
    pred : np.array
        Predictions.
    labels : np.array
        True labels.
    """
    def __init__(self, pred, labels):
        self.n = len(labels)
        self.tp = np.sum(np.logical_and(pred == 1., labels == 1.))
        self.fn = np.sum(np.logical_and(pred == -1., labels == 1.))
        self.fp = np.sum(np.logical_and(pred == 1., labels == -1.))
        self.recall = self.tp / (self.fn + self.tp) if self.fn + self.tp > 0 else 0
        self.precision = self.tp / (self.fp + self.tp) if self.fp + self.tp > 0 else 0
        self.accuracy = np.sum(labels == pred) / self.n if self.n > 0 else 0

    def __str__(self):
        acc = "Accuracy: " + str(self.accuracy)
        pre = "Precision: " + str(self.precision)
        rec = "Recall: " + str(self.recall)
        return ", ".join([acc, pre, rec])
    
    def __repr__(self):
        return self.__str__()