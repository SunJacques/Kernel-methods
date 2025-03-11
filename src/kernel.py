from itertools import product
import numpy as np
from scipy.sparse import lil_matrix
from tqdm import tqdm

class KmerKernel:
    """
    A kernel method based on k-mers for sequence comparison.

    This class extracts k-mers from sequences, constructs a feature representation 
    (phi matrix), and computes a similarity kernel between sequences.

    Attributes:
        kmin (int): Minimum k-mer length.
        kmax (int): Maximum k-mer length.
        vocab (dict): Dictionary mapping k-mers to indices.
        phi (scipy.sparse.csr_matrix): Feature matrix representing sequences.

    Methods:
        _generate_kmers(seq): Extracts k-mers from a given sequence.
        fit(sequences): Builds the k-mer vocabulary and computes the phi matrix.
        get_phi(sequences): Computes the feature representation of given sequences.
        get_kernel(): Computes the normalized kernel matrix.
        predict(sequences): Computes similarity scores between input sequences and training sequences.
    """

    def __init__(self, kmin=7, kmax=20):
        """
        Initializes the KmerKernel with the given range of k-mer lengths.

        Args:
            kmin (int, optional): Minimum k-mer length (default: 7).
            kmax (int, optional): Maximum k-mer length (default: 20).
        """
        self.kmin = kmin
        self.kmax = kmax
        self.vocab = {}

    def _generate_kmers(self, seq):
        """
        Extracts all k-mers from a sequence within the specified range of k.

        Args:
            seq (str): Input sequence.

        Returns:
            list: List of extracted k-mers.
        """
        kmers = []
        for k in range(self.kmin, self.kmax + 1):
            kmers.extend(seq[i:i + k] for i in range(len(seq) - k + 1))
        return kmers

    def fit(self, sequences):
        """
        Builds the k-mer vocabulary and computes the feature matrix (phi).

        Args:
            sequences (list of str): List of sequences.
        """
        kmer_set = set()
        for seq in sequences:
            kmer_set.update(self._generate_kmers(seq))
        self.vocab = {kmer: idx for idx, kmer in enumerate(sorted(kmer_set))}
        self.phi = self.get_phi(sequences)

    def get_phi(self, sequences):
        """
        Computes the feature representation of sequences using the k-mer vocabulary.

        Args:
            sequences (list of str): List of sequences.

        Returns:
            scipy.sparse.csr_matrix: Sparse feature matrix.
        """
        from scipy.sparse import lil_matrix

        phi = lil_matrix((len(sequences), len(self.vocab)), dtype=int)
        for i, seq in enumerate(sequences):
            kmers = self._generate_kmers(seq)
            for kmer in kmers:
                if kmer in self.vocab:
                    phi[i, self.vocab[kmer]] += 1
        return phi.tocsr()

    def get_kernel(self):
        """
        Computes the normalized kernel matrix using the phi matrix.

        Returns:
            np.ndarray: Normalized kernel matrix.
        """
        return (self.phi @ self.phi.T / np.sqrt(self.phi.sum(axis=1) @ self.phi.sum(axis=1).T)).todense()

    def predict(self, sequences):
        """
        Computes similarity scores between input sequences and training sequences.

        Args:
            sequences (list of str): List of sequences to compare.

        Returns:
            np.ndarray: Similarity scores.
        """
        phis = self.get_phi(sequences)
        k_train = self.get_kernel()
        div = np.sqrt(k_train).sum(axis=0)

        k_test = (phis @ phis.T).todense()
        pred = []
        for i in tqdm(range(phis.shape[0])):
            phi_x = phis[i, :]
            k_xx = k_test[i, i]
            pred.append((phi_x @ self.phi.T / (div * np.sqrt(k_xx))).todense().flatten())
        return np.array(pred).squeeze()
    


class KmerMismatchKernel:
    """
    A kernel method based on k-mers with mismatches for sequence comparison.

    This class extracts k-mers with mismatches from sequences, constructs a feature representation 
    (phi matrix), and computes a similarity kernel between sequences.

    Attributes:
        kmin (int): Minimum k-mer length.
        kmax (int): Maximum k-mer length.
        m (int): Number of allowed mismatches.
        vocab (dict): Dictionary mapping k-mers to indices.
        phi (scipy.sparse.csr_matrix): Feature matrix representing sequences.

    Methods:
        _generate_kmers(seq): Extracts k-mers with mismatches from a given sequence.
        fit(sequences): Builds the k-mer vocabulary and computes the phi matrix.
        get_phi(sequences): Computes the feature representation of given sequences.
        get_kernel(): Computes the normalized kernel matrix.
        predict(sequences): Computes similarity scores between input sequences and training sequences.
    """

    def __init__(self, kmin=7, kmax=20, m=3):
        """
        Initializes the KmerMismatchKernel with the given range of k-mer lengths and mismatches.

        Args:
            kmin (int, optional): Minimum k-mer length (default: 7).
            kmax (int, optional): Maximum k-mer length (default: 20).
            m (int, optional): Number of allowed mismatches (default: 1).
        """
        self.kmin = kmin
        self.kmax = kmax
        self.m = m
        self.vocab = {}

    def _generate_kmers(self, seq):
        """
        Extracts all k-mers with mismatches from a sequence within the specified range of k.

        Args:
            seq (str): Input sequence.

        Returns:
            list: List of extracted k-mers with mismatches.
        """
        kmers = []
        for k in range(self.kmin, self.kmax + 1):
            for i in range(len(seq) - k + 1):
                kmer = seq[i:i + k]
                kmers.extend(self._generate_mismatches(kmer))
        return kmers

    def _generate_mismatches(self, kmer):
        """
        Generates all k-mers with up to m mismatches from the given k-mer.

        Args:
            kmer (str): Input k-mer.

        Returns:
            list: List of k-mers with mismatches.
        """
        mismatches = set([kmer])
        for dist in range(1, self.m + 1):
            for positions in product(range(len(kmer)), repeat=dist):
                for replacements in product('ACGT', repeat=dist):
                    kmer_list = list(kmer)
                    for pos, repl in zip(positions, replacements):
                        kmer_list[pos] = repl
                    mismatches.add(''.join(kmer_list))
        return list(mismatches)

    def fit(self, sequences):
        """
        Builds the k-mer vocabulary and computes the feature matrix (phi).

        Args:
            sequences (list of str): List of sequences.
        """
        kmer_set = set()
        for seq in sequences:
            kmer_set.update(self._generate_kmers(seq))
        self.vocab = {kmer: idx for idx, kmer in enumerate(sorted(kmer_set))}
        self.phi = self.get_phi(sequences)

    def get_phi(self, sequences):
        """
        Computes the feature representation of sequences using the k-mer vocabulary.

        Args:
            sequences (list of str): List of sequences.

        Returns:
            scipy.sparse.csr_matrix: Sparse feature matrix.
        """
        phi = lil_matrix((len(sequences), len(self.vocab)), dtype=int)
        for i, seq in enumerate(sequences):
            kmers = self._generate_kmers(seq)
            for kmer in kmers:
                if kmer in self.vocab:
                    phi[i, self.vocab[kmer]] += 1
        return phi.tocsr()

    def get_kernel(self):
        """
        Computes the normalized kernel matrix using the phi matrix.

        Returns:
            np.ndarray: Normalized kernel matrix.
        """
        return (self.phi @ self.phi.T / np.sqrt(self.phi.sum(axis=1) @ self.phi.sum(axis=1).T)).todense()

    def predict(self, sequences):
        """
        Computes similarity scores between input sequences and training sequences.

        Args:
            sequences (list of str): List of sequences to compare.

        Returns:
            np.ndarray: Similarity scores.
        """
        phis = self.get_phi(sequences)
        k_train = self.get_kernel()
        div = np.sqrt(k_train).sum(axis=0)

        k_test = (phis @ phis.T).todense()
        pred = []
        for i in tqdm(range(phis.shape[0])):
            phi_x = phis[i, :]
            k_xx = k_test[i, i]
            pred.append((phi_x @ self.phi.T / (div * np.sqrt(k_xx))).todense().flatten())
        return np.array(pred).squeeze()