#!/usr/bin/env python
"""
Define client class which loads a train set and receives
job orders from a master via a RabbitMQ message broker.
"""

import json
import pika
import threading
import time
from sklearn.preprocessing import LabelBinarizer
from .individuals import GeneticCnnIndividual, CrowIndividual
import numpy as np
import random

class GentunClient(object):

    def __init__(self, gpu="0", host='localhost', port=5672, user='guest', password='guest', rabbit_queue='rpc_queue'):
        self.gpu = gpu
        self.credentials = pika.PlainCredentials(user, password)
        self.parameters = pika.ConnectionParameters(host, port, '/', self.credentials)
        self.connection = pika.BlockingConnection(self.parameters)
        self.channel = self.connection.channel()
        self.rabbit_queue = rabbit_queue
        self.channel.queue_declare(queue=self.rabbit_queue)
        # Report to the RabbitMQ server
        heartbeat_thread = threading.Thread(target=self.heartbeat)
        heartbeat_thread.daemon = True
        heartbeat_thread.start()

    def heartbeat(self):
        """Send heartbeat messages to RabbitMQ server."""
        while True:
            time.sleep(10)
            try:
                self.connection.process_data_events()
            except:
                pass

    def on_request(self, channel, method, properties, body):
        i, genes, fitness, last_location, best_fitness, memory, new_location, additional_parameters, exp_no, algo, dataset = json.loads(
            body)
        print(additional_parameters)
        # If an additional parameter is received as a list, convert to tuple
        for param in additional_parameters.keys():
            if isinstance(additional_parameters[param], list):
                additional_parameters[param] = tuple(additional_parameters[param])

        self.algorithm = algo
        self.dataset = dataset

        if self.dataset == "mnist":
            from keras.datasets import mnist
            (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

        elif self.dataset == "cifar10":
            from keras.datasets import cifar10
            (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

        else:
            raise Exception("Currently only mnist and cifar10 datasets are supported")
        print(self.dataset)
        n = train_images.shape[0]
        test_n=test_images.shape[0]
        self.input_shape = train_images[0].shape
        lb = LabelBinarizer()
        lb.fit(range(10))
        print("Here")
        selection = random.sample(range(n), 10000)  # Use only a subsample
        test_selection = random.sample(range(test_n), 1000)
        print("Here")
        self.y_train = lb.transform(train_labels[selection])  # One-hot encodings
        print("Here")
        test_sel_labels=test_labels[test_selection]
        self.y_test = lb.transform(test_labels[test_selection])
        print("Here")
        if len(train_images.shape) < 4:
            new_shape = (*train_images.shape, 1)
            new_shape_test = (*test_images.shape, 1)
        else:
            new_shape = train_images.shape
            new_shape_test = test_images.shape

        x_train = train_images.reshape(new_shape)[selection]
        x_test = test_images.reshape(new_shape_test)[test_selection]
        self.x_train = x_train / 255  # Normalize train data
        self.x_test = x_test / 255

        (unique_labels, nb_classes) = np.unique(train_labels, return_counts=True)

        print(" [.] Evaluating individual {}".format(i))
        # print("     ... Genes: {}".format(str(genes)))
        # print("     ... Other: {}".format(str(additional_parameters)))
        # Run model and return fitness metric
        if self.algorithm == "ga":
            self.individual = GeneticCnnIndividual
            individual = self.individual(self.x_train, self.y_train, genes=genes, **additional_parameters)
            fitness = individual.get_fitness()
            # Prepare response for master and send it
            response = json.dumps([i, fitness])
        elif self.algorithm == "csa":
            self.individual = CrowIndividual
            individual = self.individual(self.gpu, self.x_train, self.y_train, self.x_test, self.y_test, id=i,
                                         space=genes, location=new_location, memory=memory, best_fitness=best_fitness,
                                         fitness=fitness, last_location=last_location, **additional_parameters)
            import time
            start_time = time.time()
            fitness = individual.evaluate_fitness()
            training_time = time.time() - start_time

            best_fitness = individual.get_best_fitness()
            memory = individual.get_memory()
            location = individual.get_location()
            last_location = individual.get_last_location()
            # Prepare response for master and send it
            response = json.dumps(
                [i, last_location, fitness, memory, best_fitness, location, training_time, individual.loss,
                 individual.mae, individual.mse, individual.msle,individual.training_history,individual.epochs_history,individual.model_json])
        else:
            raise Exception("Only Genetic Algorithm and Crow Search Algorithm are supported")

        channel.basic_publish(
            exchange='', routing_key=properties.reply_to,
            properties=pika.BasicProperties(correlation_id=properties.correlation_id), body=response
        )
        channel.basic_ack(delivery_tag=method.delivery_tag)
        print("hello")

    def work(self):
        try:
            self.channel.basic_qos(prefetch_count=1)
            self.channel.basic_consume(queue=self.rabbit_queue, on_message_callback=self.on_request)
            print(" [x] Awaiting master's requests")
            print(" [-] Press Ctrl+C to interrupt")
            while True:
                try:
                    self.channel.start_consuming()
                except:
                    pass
        except KeyboardInterrupt:
            print()
            print("Good bye!")
