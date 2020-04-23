#!/usr/bin/env python
"""
Implementation of a distributed version of the Genetic CNN
algorithm on MNIST data. The rabbitmq service should be
running in 'localhost'.
"""

import os
import sys
import argparse
import pymongo
import time
from pprint import pprint
from _datetime import timedelta,datetime
db_client=pymongo.MongoClient("223.195.37.85",27017)
db=db_client["binaryCSA"]
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
parser = argparse.ArgumentParser("Neural Architecture Search Server")
parser.add_argument('-d', '--dataset', type=str, default="mnist", help="Name of dataset (cifar10/mnist)")
parser.add_argument('-a', '--algorithm', type=str, default="csa", help="Name of algorithm (csa/ga)")
args = parser.parse_args()

def load_individuals(file,fromdb=True):
    import json

    if fromdb:
        initial_flock=[]
        exp_col = db["experiments"]
        flock=exp_col.find_one({"no":file})["iterations"][0]
        for crow in flock:
            initial_flock.append(crow["location"])
        return initial_flock
    else:
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

        individuals_list=None
        seed_file=None
        # seed_file="../200407_csa_20i_20c_fl13_ap15.txt"
        # individuals_list=load_individuals(3,fromdb=True)

        iterations = 50
        flock_size = 20
        flight_length = 13
        awareness_probability = 0.15
        tournament_size = 5

        nodes = (3, 4, 5)
        kernels_per_layer = (64, 128, 256)
        kernel_sizes = ((5, 5), (5,5), (5,5))
        dense_units = 1024


        kfold = 3
        batch_size = 32
        epochs = (120, 60, 40, 20)
        learning_rates = (1e-2, 1e-3, 1e-4, 1e-5)
        dropout_probability = 0.5
        maximize = True


        host = 'localhost'
        port = 5672
        user = 'test'
        password = 'test'

        start_time=time.time()
        exp_col=db["experiments"]
        experiment_no=exp_col.estimated_document_count()+1
        experiment_doc={
            "no": experiment_no,
            "algo":args.algorithm,
            "start": start_time,
            "seed_file" : seed_file,
            "flock_size" : flock_size,
            "flight_length" : flight_length,
            "awareness_probability" : awareness_probability,
            "tournament_size" : tournament_size,
            "maximize" : maximize,
            'kfold': kfold, 'epochs': epochs, 'learning_rate': learning_rates, 'batch_size': batch_size,
            "nodes": nodes, "kernels_per_layer": kernels_per_layer, "kernel_sizes": kernel_sizes,
            "dense_units": dense_units,
            "dropout_probability": dropout_probability,
            "iterations":[]
        }

        exp_col.insert_one(experiment_doc)
        print("Running Experiment Number",experiment_no,"at",datetime.fromtimestamp(start_time))

        flock = DistributedFlock(args.algorithm,args.dataset,
            CrowIndividual, input_shape=input_shape,nb_classes=nb_classes,size=flock_size, flight_length=flight_length, awareness_probability=awareness_probability, individual_list=individuals_list,
            additional_parameters={
                'kfold': kfold, 'epochs': epochs, 'learning_rate': learning_rates, 'batch_size': batch_size,
                "nodes": nodes, "kernels_per_layer": kernels_per_layer, "kernel_sizes": kernel_sizes, "dense_units": dense_units,
                "dropout_probability": dropout_probability
            }, maximize=maximize, host=host, port=port, user=user, password=password,exp_no=experiment_no
        )

        csa = CrowSearchAlgorithm(flock,tournament_size)
        csa.run(iterations,experiment_no)
        running_time=time.time()-start_time
        print(timedelta(seconds=running_time))
        exp_col.update_one({"no":experiment_no},{"$set":{"exec_time":str(timedelta(seconds=running_time))}})
        print("Total Running Time is",running_time)
    else:
        raise Exception("Only GA and CSA are supported")