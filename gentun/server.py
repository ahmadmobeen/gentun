#!/usr/bin/env python
"""
Client to communicate with RabbitMQ and extension of
Population which add parallel computing capabilities.
"""

import json
import pika
import queue
import threading
import time
import uuid
import ast
from .populations import Population, Flock,GridPopulation
import pymongo
db_client=pymongo.MongoClient("223.195.37.85",27017)
db=db_client["binaryCSA"]
exp_col=db["experiments"]

class RpcClient(object):
    """Define a client which sends work orders to a
    RabbitMQ message broker with a unique identifier
    and awaits for a response.
    """

    def __init__(self, jobs, responses, host='localhost', port=5672,
                 user='test', password='test', rabbit_queue='rpc_queue'):
        # Set connection and channel
        self.credentials = pika.PlainCredentials(user, password)
        self.parameters = pika.ConnectionParameters(host, port, '/', self.credentials)
        self.connection = pika.BlockingConnection(self.parameters)
        self.channel = self.connection.channel()
        # Set queue for jobs and callback queue for responses
        result = self.channel.queue_declare(queue='', exclusive=True)
        self.callback_queue = result.method.queue
        self.channel.basic_consume(queue=self.callback_queue, on_message_callback=self.on_response, auto_ack=True)
        self.rabbit_queue = rabbit_queue
        self.channel.queue_declare(queue=self.rabbit_queue)
        self.response = None
        self.id = None
        # Local queues shared between threads
        self.jobs = jobs
        self.responses = responses
        # Report to the RabbitMQ server
        heartbeat_thread = threading.Thread(target=self.heartbeat)
        heartbeat_thread.daemon = True
        heartbeat_thread.start()

    def purge(self):
        self.channel.queue_purge(queue=self.rabbit_queue)

    def heartbeat(self):
        """Send heartbeat messages to RabbitMQ server."""
        while True:
            time.sleep(10)
            try:
                self.connection.process_data_events()
            except:
                # Connection was closed, stop sending heartbeat messages
                break

    def on_response(self, channel, method, properties, body):
        if self.id == properties.correlation_id:
            self.response = body

    def call(self, parameters):
        assert type(parameters) == str
        self.id = str(uuid.uuid4())
        properties = pika.BasicProperties(reply_to=self.callback_queue, correlation_id=self.id)
        self.channel.basic_publish(
            exchange='', routing_key=self.rabbit_queue, properties=properties, body=parameters
        )
        while self.response is None:
            time.sleep(3)
        id, _, fitness,last_location,best_fitness,memory,location,_,exp_no,algo,dataset=json.loads(parameters)

        print("\n [*] Evaluating individual {}".format(id), "on ", location, ".")
        print(" [*] Fitness of Crow {}".format(id), "on location", last_location," was {:.8f}".format(fitness), ".")
        print(" [*] Best known performance of Crow {}".format(id), " is", "{:.8f}".format(best_fitness),"on location", memory)


        client_id,client_last_location,client_acc,client_memory,client_best_acc,client_location,client_train_time,loss,mae,mse,msle=json.loads(self.response)
        # assert(id==client_id)
        # assert(location==client_location)
        # assert(last_location==client_last_location)
        print(" [*] Performance of Crow {}".format(id)," is", "{:.8f}".format(client_acc), "on location", location)
        if best_fitness == None or best_fitness < client_acc:
            print(" [*] Updating best known performance for Crow {}".format(id), " to","{:.8f}".format(client_best_acc), "on location", client_memory)
        else:
            print(" [*] Best known performance for Crow {}".format(id), " remains the same ","{:.8f}".format(client_best_acc), "on location", client_memory)
        print(" [*] Training time of Crow {}".format(id), " is", client_train_time)

        crow_doc={
            "id":client_id,
            "acc":client_acc,
            "memory":client_memory,
            "best":client_best_acc,
            "location":client_location,
            "cross_entropy_loss":loss,
            "mean_absolute_error":mae,
            "mean_squared_erro":mse,
            "mean_squared_log_error":msle,
            "train_time":client_train_time
        }

        iteration=len(exp_col.find({"no":exp_no})[0]["iterations"])-1
        exp_col.update_one({"no":exp_no},{"$push":{"iterations."+str(iteration):crow_doc}})

        # id=int(self.response.decode().split(",")[0].split("[")[1])
        # acc=float(self.response.decode().split(",")[1])
        # best_acc = float(self.response.decode().split(",")[3].split(']')[0])
        # memory=ast.literal_eval("{"+self.response.decode().split("{")[1].split("}")[0]+"}")
        # if individual_attr[3] is None:
        #     individual_attr[3] = 0.0000
        # print(" [*] Fitness for individual {}".format(id)," is {:.8f}".format(acc),"on location",last_location,". Individual's best known performance is","{:.8f}".format(best_acc),"on location",memory,". The new position is ",new_location)
        self.responses.put(self.response)
        # Job is completed, remove job order from queue
        self.jobs.get()
        self.jobs.task_done()
        # Close RabbitMQ connection to prevent file descriptors limit from blocking server
        self.connection.close()


class DistributedPopulation(Population):
    """Override Population class by making x_train and
    y_train optional parameters set to None and sending
    evaluation requests to the workers before computing
    the fittest individual.
    """

    def __init__(self, species, x_train=None, y_train=None, input_shape=(28,28,1),nb_classes=10,individual_list=None, size=None,
                 crossover_rate=0.5, mutation_rate=0.015, maximize=True, additional_parameters=None,
                 host='localhost', port=5672, user='test', password='test', rabbit_queue='rpc_queue'):
        super(DistributedPopulation, self).__init__(
            species, x_train, y_train, input_shape,nb_classes,individual_list, size,
            crossover_rate, mutation_rate, maximize, additional_parameters
        )
        self.credentials = {
            'host': host,
            'port': port,
            'user': user,
            'password': password,
            'rabbit_queue': rabbit_queue
        }

    def get_fittest(self):
        """Evaluate necessary individuals in parallel before getting fittest."""
        self.evaluate_in_parallel()
        return super(DistributedPopulation, self).get_fittest()

    def evaluate_in_parallel(self):
        """Send job requests to RabbitMQ pool so that
        workers evaluate individuals with unknown fitness.
        """
        # Purge job queue if necessary
        RpcClient(None, None, **self.credentials).purge()
        jobs = queue.Queue()  # "Counter" of pending jobs, shared between threads
        responses = queue.Queue()  # Collect fitness values from workers
        for i, individual in enumerate(self.individuals):
            if not individual.get_fitness_status():
                job_order = json.dumps([i, individual.get_genes(), individual.get_additional_parameters()])
                jobs.put(True)
                client = RpcClient(jobs, responses, **self.credentials)
                communication_thread = threading.Thread(target=client.call, args=[job_order])
                communication_thread.daemon = True
                communication_thread.start()
        jobs.join()  # Block here until all jobs are completed
        # Collect results and assign them to their respective individuals
        while not responses.empty():
            response = responses.get(False)
            i, value = json.loads(response)
            self.individuals[i].set_fitness(value)


class DistributedFlock(Flock):
    """Override Population class by making x_train and
    y_train optional parameters set to None and sending
    evaluation requests to the workers before computing
    the fittest individual.
    """

    def __init__(self, algo,dataset,species, x_train=None, y_train=None, x_test=None, y_test=None,input_shape=(28,28,1),nb_classes=10,individual_list=None, size=None,
                 flight_length=13,awareness_probability=0.15, maximize=True, additional_parameters=None,
                 host='localhost', port=5672, user='test', password='test', rabbit_queue='rpc_queue',exp_no=0):
        super(DistributedFlock, self).__init__(
            species, x_train, y_train, x_test,y_test,input_shape,nb_classes,individual_list, size,
            flight_length,awareness_probability, maximize, additional_parameters
        )
        self.algo=algo
        self.dataset=dataset
        self.credentials = {
            'host': host,
            'port': port,
            'user': user,
            'password': password,
            'rabbit_queue': rabbit_queue
        }
        self.exp_no=exp_no

    def get_fittest(self):
        """Evaluate necessary individuals in parallel before getting fittest."""
        self.evaluate_in_parallel()
        return super(DistributedFlock, self).get_fittest()

    def evaluate_in_parallel(self):
        """Send job requests to RabbitMQ pool so that
        workers evaluate individuals with unknown fitness.
        """
        # Purge job queue if necessary
        explored=[]
        explored_fitness=[]
        RpcClient(None, None, **self.credentials).purge()
        jobs = queue.Queue()  # "Counter" of pending jobs, shared between threads
        responses = queue.Queue()  # Collect fitness values from workers
        for i, individual in enumerate(self.individuals):
            # if not individual.get_fitness_status():
            if individual.get_location() not in explored:
                job_order = json.dumps([i, individual.get_space(), individual.get_fitness(),individual.get_last_location(),individual.get_best_fitness(),individual.get_memory(),individual.get_location(),individual.get_additional_parameters(),self.exp_no,self.algo,self.dataset])
                jobs.put(True)
                client = RpcClient(jobs, responses, **self.credentials)
                communication_thread = threading.Thread(target=client.call, args=[job_order])
                communication_thread.daemon = True
                communication_thread.start()
            else:
                print("Performance on location",individual.get_location(), "has already been measured to be",explored_fitness[explored.index(individual.get_location())])
                individual.set_fitness(explored_fitness[explored.index(individual.get_location())])
        jobs.join()  # Block here until all jobs are completed
        # Collect results and assign them to their respective individuals
        while not responses.empty():
            response = responses.get(False)
            # id, last_location, acc, memory, best_acc, new_location =
            client_id, client_last_location, client_acc, client_memory, client_best_acc, client_location,exec_time,loss,mae,mse,msle=json.loads(response)
            individual=self.individuals[client_id]
            assert (individual.get_id() == client_id)
            assert (individual.get_location() == client_location)
            assert (individual.get_last_location() == client_last_location)

            individual.set_fitness(client_acc)
            # self.individuals[id].set_location(new_location)
            individual.set_best_fitness(client_best_acc)
            individual.set_memory(client_memory)
            # self.individuals[id].set_last_location(last_location)
            if client_location not in explored:
                explored.append(client_location)
                explored_fitness.append(client_acc)


class DistributedGridPopulation(DistributedPopulation, GridPopulation):
    """Same as a DistributedPopulation but creates a
    GridPopulation instead of a random one.
    """

    def __init__(self, species, x_train=None, y_train=None, individual_list=None, genes_grid=None,
                 crossover_rate=0.5, mutation_rate=0.015, maximize=True, additional_parameters=None,
                 host='localhost', port=5672, user='test', password='test', rabbit_queue='rpc_queue'):
        # size parameter of DistributedPopulation is replaced with genes_grid
        super(DistributedGridPopulation, self).__init__(
            species, x_train, y_train, individual_list, genes_grid,
            crossover_rate, mutation_rate, maximize, additional_parameters,
            host, port, user, password, rabbit_queue
        )