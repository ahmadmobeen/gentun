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
parser.add_argument('-d', '--dataset', type=str, default="cifar10", help="Name of dataset (cifar10/mnist)")
parser.add_argument('-a', '--algorithm', type=str, default="csa", help="Name of algorithm (csa/ga)")
args = parser.parse_args()

def load_individuals(file):
    import json

    data = {
        "flock_size": 0,
        "total_iterations": 0,
        "initial_flock": [],
        "iterations": []
    }
    init = False
    id = 0
    with open(file, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if "Initializing a random flock." in line:
                data["flock_size"] = int(line.split(":")[1])
            if "S_1" in line and not init:
                id = line.split(" ")[0]
                data["initial_flock"].append(json.loads(line.replace(id + " ", "").replace("\'", "\"")))
            if "Starting Crow Search Algorithm..." in line:
                init = True
    return data["initial_flock"]

if __name__ == '__main__':
    from gentun import RussianRouletteGA, DistributedPopulation, GeneticCnnIndividual, DistributedFlock,CrowIndividual,CrowSearchAlgorithm
    #Todo: Timestamp
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

        individuals_list=load_individuals("../200407_csa_20i_20c_fl13_ap15.txt")

        # print(individuals_list)
        # exit()

        flock = DistributedFlock(
            CrowIndividual, input_shape=input_shape,nb_classes=nb_classes,size=20, flight_length=13, awareness_probability=0.15, individual_list=individuals_list,
            additional_parameters={
                'kfold': 3, 'epochs': (20, 4, 1), 'learning_rate': (1e-3, 1e-4, 1e-5), 'batch_size': 32
            }, maximize=True, host='localhost', user='test', password='test'
        )
        # exit()
        csa = CrowSearchAlgorithm(flock,5)
        csa.run(20)
    else:
        raise Exception("Only GA and CSA are supported")