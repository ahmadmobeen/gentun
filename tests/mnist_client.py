#!/usr/bin/env python
"""
Create a client which loads MNIST data and waits for jobs
to evaluate models. The rabbitmq service should be running
in 'localhost'.
"""

import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

parser = argparse.ArgumentParser("Neural Architecture Search Client")
parser.add_argument('-d', '--dataset', type=str, default="cifar10", help="Name of dataset (cifar10/mnist)")
parser.add_argument('-a', '--algorithm', type=str, default="csa", help="Name of algorithm (csa/ga)")
args = parser.parse_args()


if __name__ == '__main__':
    import random

    from sklearn.preprocessing import LabelBinarizer
    from gentun import GentunClient, GeneticCnnIndividual, CrowIndividual

    if args.dataset=="mnist":
        import mnist
        train_images = mnist.train_images()
        train_labels = mnist.train_labels()

    elif args.dataset=="cifar10":
        from keras.datasets import cifar10
        (train_images,train_labels),_ = cifar10.load_data()

    else:
        raise Exception("Currently only mnist and cifar10 datasets are supported")

    n = train_images.shape[0]
    input_shape=train_images[0].shape
    lb = LabelBinarizer()
    lb.fit(range(10))
    selection = random.sample(range(n), 10000)  # Use only a subsample
    y_train = lb.transform(train_labels[selection])  # One-hot encodings

    if len(train_images.shape) < 4:
        new_shape = (*train_images.shape, 1)
    else:
        new_shape = train_images.shape

    x_train = train_images.reshape(new_shape)[selection]
    x_train = x_train / 255  # Normalize train data

    (unique_labels, nb_classes) = np.unique(train_labels, return_counts=True)

    if args.algorithm=="csa":
        individual=CrowIndividual
    elif args.algorithm=="ga":
        individual=GeneticCnnIndividual
    else:
        raise Exception("Only Genetic Algorithm and Crow Search Algorithm are supported")

    gc = GentunClient(individual, args.algorithm,x_train, y_train, host='223.195.37.85', user='test', password='test')
    gc.work()
