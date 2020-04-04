#!/usr/bin/env python
"""
Implementation of a distributed version of the Genetic CNN
algorithm on MNIST data. The rabbitmq service should be
running in 'localhost'.
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
parser = argparse.ArgumentParser("Neural Architecture Search Server")
parser.add_argument('-d', '--dataset', type=str, default="mnist", help="Name of dataset (cifar10/mnist)")
parser.add_argument('-a', '--algorithm', type=str, default="csa", help="Name of algorithm (csa/ga)")
args = parser.parse_args()


if __name__ == '__main__':
    from gentun import RussianRouletteGA, DistributedPopulation, GeneticCnnIndividual, DistributedFlock,CrowIndividual,CrowSearchAlgorithm

    if args.dataset=="mnist":
        input_shape = (28,28,1)
        nb_classes = 10

    elif args.dataset=="cifar10":
        input_shape = (32, 32, 3)
        nb_classes = 10

    else:
        raise Exception("Only cifar10 and mnist is supported")

    if args.algorithm=="ga":

        pop = DistributedPopulation(
            GeneticCnnIndividual, input_shape=input_shape,nb_classes=nb_classes,size=5, crossover_rate=0.3, mutation_rate=0.1,
            additional_parameters={
                'kfold': 5, 'epochs': (20, 4, 1), 'learning_rate': (1e-3, 1e-4, 1e-5), 'batch_size': 32
            }, maximize=True, host='localhost', user='test', password='test'
        )
        ga = RussianRouletteGA(pop, crossover_probability=0.2, mutation_probability=0.8)
        ga.run(50)

    elif args.algorithm=="csa":
        flock = DistributedFlock(
            CrowIndividual, input_shape=input_shape,nb_classes=nb_classes,size=3, flight_length=13, awareness_probability=0.15,
            additional_parameters={
                'kfold': 3, 'epochs': (20, 4, 1), 'learning_rate': (1e-3, 1e-4, 1e-5), 'batch_size': 32
            }, maximize=True, host='localhost', user='test', password='test'
        )
        csa = CrowSearchAlgorithm(flock,3)
        csa.run(2)
    else:
        raise Exception("Only GA and CSA are supported")