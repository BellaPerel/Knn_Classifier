import os
import sys
import argparse
import time
import itertools
import numpy as np
import pandas as pd

class KnnClassifier:
    def __init__(self, k: int, p: float):
        """
        Constructor for the KnnClassifier.

        :param k: Number of nearest neighbors to use.
        :param p: p parameter for Minkowski distance calculation.
        """
        self.k = k
        self.p = p

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        This method trains a k-NN classifier on a given training set X with label set y.

        :param X: A 2-dimensional numpy array of m rows and d columns. It is guaranteed that m >= 1 and d >= 1.
            Array datatype is guaranteed to be np.float32.
        :param y: A 1-dimensional numpy array of m rows. it is guaranteed to match X's rows in length (|m_x| == |m_y|).
            Array datatype is guaranteed to be np.uint8.
        """
        self.data = X
        self.labels = y
        pass

    def distance_by_norm(self, test_point):
        p = self.p
        train_points = self.data
        distances_labels_vec = []
        for i in range(len(train_points)):
            minkowski_distance = pow(sum(pow(abs(test_point-train_points[i]),p)), 1/p)
            distances_labels_vec.append([minkowski_distance, self.labels[i]])
        return distances_labels_vec

    def sort_sub(self, sub_li):
        # reverse = None (Sorts in Ascending order)
        # key is set to sort using second element of
        # sublist lambda has been used
        sub_li.sort(key = lambda x: x[0])
        return sub_li


    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        This method predicts the y labels of a given dataset X, based on a previous training of the model.
        It is mandatory to call KnnClassifier.fit before calling this method.

        :param X: A 2-dimensional numpy array of m rows and d columns. It is guaranteed that m >= 1 and d >= 1.
            Array datatype is guaranteed to be np.float32.
        :return: A 1-dimensional numpy array of m rows. Should be of datatype np.uint8.
        """
        k = self.k
        prediction = []
        for point in X:
            hist = dict.fromkeys(self.labels, 0)
            vector_of_distances_from_train_points = self.distance_by_norm(point)
            vector_of_distances_from_train_points = sorted(vector_of_distances_from_train_points, key=lambda x: (x[0], x[1]))
            k_nearest_by_distance = vector_of_distances_from_train_points[:k]
            for neigh in k_nearest_by_distance:
                hist[neigh[1]] = hist[neigh[1]] + 1
            predicted_label_amount = hist[max(hist, key=hist.get)]
            for neigh in k_nearest_by_distance:
                if hist[neigh[1]] == predicted_label_amount:
                    prediction.append(neigh[1])
                    break
        return np.array(prediction)


def main():

    print("*" * 20)
    print("Started HW1_ID1_ID2.py")
    # Parsing script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('csv', type=str, help='Input csv file path')
    parser.add_argument('k', type=int, help='k parameter')
    parser.add_argument('p', type=float, help='p parameter')
    args = parser.parse_args()

    print("Processed input arguments:")
    print(f"csv = {args.csv}, k = {args.k}, p = {args.p}")

    print("Initiating KnnClassifier")
    model = KnnClassifier(k=args.k, p=args.p)
    print(f"Student IDs: {model.ids}")
    print(f"Loading data from {args.csv}...")
    data = pd.read_csv(args.csv, header=None)
    print(f"Loaded {data.shape[0]} rows and {data.shape[1]} columns")
    X = data[data.columns[:-1]].values.astype(np.float32)
    y = pd.factorize(data[data.columns[-1]])[0].astype(np.uint8)

    print("Fitting...")
    model.fit(X, y)
    print("Done")
    print("Predicting...")
    y_pred = model.predict(X)
    print("Done")
    accuracy = np.sum(y_pred == y) / len(y)
    print(f"Train accuracy: {accuracy * 100 :.2f}%")
    print("*" * 20)


if __name__ == '__main__':
    main()