#!/usr/bin/env python
"""
Classes which define the individuals of a population with
its characteristic genes, generation, crossover and
mutation processes.
"""

import math
import pprint
import random

try:
    from .models.xgboost_models import XgboostModel
except ImportError:
    pass

try:
    from .models.keras_models import GeneticCnnModel
except ImportError:
    pass


def random_log_uniform(minimum, maximum, base, eps=1e-12):
    """Generate a random number which is uniform in a
    logarithmic scale. If base > 0 scale goes from minimum
    to maximum, if base < 0 vice versa, and if base is 0,
    use a uniform scale.
    """
    if base == 0:
        return random.uniform(minimum, maximum)
    minimum += eps  # Avoid math domain error when minimum is zero
    if base > 0:
        return base ** random.uniform(math.log(minimum, base), math.log(maximum, base))
    base = abs(base)
    return maximum - base ** random.uniform(math.log(eps, base), math.log(maximum - minimum, base))


class Individual(object):
    """Basic definition of an individual containing
    reproduction and mutation methods. Do not instantiate,
    use a subclass which extends this object by defining a
    genome and a random individual generator.
    """

    def __init__(self, x_train, y_train, genome, genes, crossover_rate, mutation_rate, additional_parameters=None):
        self.x_train = x_train
        self.y_train = y_train
        self.genome = genome
        self.validate_genome()
        self.genes = genes
        self.validate_genes()
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.fitness = None  # Until evaluated an individual fitness is unknown
        assert additional_parameters is None

    def validate_genome(self):
        """Check genome structure."""
        if type(self.genome) != dict:
            raise TypeError("Genome must be a dictionary.")
        for gene, properties in self.genome.items():
            if type(gene) != str:
                raise TypeError("Gene names must be strings.")

    def validate_genes(self):
        """Check that genes are compatible with genome."""
        if set(self.genome.keys()) != set(self.genes.keys()):
            raise ValueError("Genes passed don't correspond to individual's genome.")

    def get_genes(self):
        """Return individual's genes."""
        return self.genes

    def get_genome(self):
        """Return individual's genome."""
        return self.genome

    @staticmethod
    def generate_random_genes(genome):
        raise NotImplementedError("Use a subclass with genes definition.")

    def evaluate_fitness(self):
        raise NotImplementedError("Use a subclass with genes definition.")

    def get_additional_parameters(self):
        raise NotImplementedError("Use a subclass with genes definition.")

    def get_fitness(self):
        """Compute individual's fitness if necessary and return it."""
        if self.fitness is None:
            self.evaluate_fitness()
        return self.fitness

    def reproduce(self, partner):
        """Mix genes from self and partner randomly and
        return a new instance of an individual. Do not
        mutate parents.
        """
        assert self.__class__ == partner.__class__  # Can only reproduce if they're the same species
        child_genes = {}
        for name, value in self.get_genes().items():
            if random.random() < self.crossover_rate:
                child_genes[name] = partner.get_genes()[name]
            else:
                child_genes[name] = value
        return self.__class__(
            self.x_train, self.y_train, self.genome, child_genes, self.crossover_rate, self.mutation_rate,
            **self.get_additional_parameters()
        )

    def crossover(self, partner):
        """Mix genes from self and partner randomly.
        Mutates each parent instead of producing a
        new instance (child).
        """
        assert self.__class__ == partner.__class__  # Can only cross if they're the same species
        for name in self.get_genes().keys():
            if random.random() < self.crossover_rate:
                self.get_genes()[name], partner.get_genes()[name] = partner.get_genes()[name], self.get_genes()[name]
                self.set_fitness(None)
                partner.set_fitness(None)

    def mutate(self):
        """Mutate instance's genes with a certain probability."""
        for name, value in self.get_genes().items():
            if random.random() < self.mutation_rate:
                default, minimum, maximum, log_scale = self.get_genome()[name]
                if type(default) == int:
                    self.get_genes()[name] = random.randint(minimum, maximum)
                else:
                    self.get_genes()[name] = round(random_log_uniform(minimum, maximum, log_scale), 4)
                self.set_fitness(None)  # The mutation produces a new individual

    def get_fitness_status(self):
        """Return True if individual's fitness in known."""
        return self.fitness is not None

    def set_fitness(self, value):
        """Assign fitness."""
        self.fitness = value

    def copy(self):
        """Copy instance."""
        individual_copy = self.__class__(
            self.x_train, self.y_train, self.genome, self.genes.copy(), self.crossover_rate,
            self.mutation_rate, **self.get_additional_parameters()
        )
        individual_copy.set_fitness(self.fitness)
        return individual_copy

    def __str__(self):
        """Return genes which identify the individual."""
        return pprint.pformat(self.genes)


class XgboostIndividual(Individual):

    def __init__(self, x_train, y_train, genome=None, genes=None, crossover_rate=0.5, mutation_rate=0.015,
                 booster='gbtree', objective='reg:linear', eval_metric='rmse', kfold=5,
                 num_boost_round=5000, early_stopping_rounds=100):
        if genome is None:
            genome = {
                # name: (default, min, max, logarithmic-scale-base)
                'eta': (0.3, 0.001, 1.0, 10),
                'min_child_weight': (1, 0, 10, None),
                'max_depth': (6, 3, 10, None),
                'gamma': (0.0, 0.0, 10.0, 10),
                'max_delta_step': (0, 0, 10, None),
                'subsample': (1.0, 0.0, 1.0, -10),
                'colsample_bytree': (1.0, 0.0, 1.0, -10),
                'colsample_bylevel': (1.0, 0.0, 1.0, -10),
                'lambda': (1.0, 0.1, 10.0, 10),
                'alpha': (0.0, 0.0, 10.0, 10),
                'scale_pos_weight': (1.0, 0.0, 10.0, 0)
            }
        if genes is None:
            genes = self.generate_random_genes(genome)
        # Set individual's attributes
        super(XgboostIndividual, self).__init__(x_train, y_train, genome, genes, crossover_rate, mutation_rate)
        # Set additional parameters which are not tuned
        self.booster = booster
        self.objective = objective
        self.eval_metric = eval_metric
        self.kfold = kfold
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds

    @staticmethod
    def generate_random_genes(genome):
        """Create and return random genes."""
        genes = {}
        for name, (default, minimum, maximum, log_scale) in genome.items():
            if type(default) == int:
                genes[name] = random.randint(minimum, maximum)
            else:
                genes[name] = round(random_log_uniform(minimum, maximum, log_scale), 4)
        return genes

    def evaluate_fitness(self):
        """Create model and perform cross-validation."""
        model = XgboostModel(
            self.x_train, self.y_train, self.genes, booster=self.booster, objective=self.objective,
            eval_metric=self.eval_metric, kfold=self.kfold, num_boost_round=self.num_boost_round,
            early_stopping_rounds=self.early_stopping_rounds
        )
        self.fitness = model.cross_validate()

    def get_additional_parameters(self):
        return {
            'booster': self.booster,
            'objective': self.objective,
            'eval_metric': self.eval_metric,
            'kfold': self.kfold,
            'num_boost_round': self.num_boost_round,
            'early_stopping_rounds': self.early_stopping_rounds
        }


class GeneticCnnIndividual(Individual):

    def __init__(self, x_train, y_train, genome=None, genes=None, crossover_rate=0.3, mutation_rate=0.1, nodes=(3, 5),
                 input_shape=(28, 28, 1), kernels_per_layer=(20, 50), kernel_sizes=((5, 5), (5, 5)), dense_units=500,
                 dropout_probability=0.5, classes=10, kfold=5, epochs=(3,), learning_rate=(1e-3,), batch_size=32):
        if genome is None:
            genome = {'S_{}'.format(i + 1): int(K_s * (K_s - 1) / 2) for i, K_s in enumerate(nodes)}
        if genes is None:
            genes = self.generate_random_genes(genome)
        # Set individual's attributes
        super(GeneticCnnIndividual, self).__init__(x_train, y_train, genome, genes, crossover_rate, mutation_rate)
        # Set additional parameters which are not tuned
        assert len(nodes) == len(kernels_per_layer) and len(kernels_per_layer) == len(kernel_sizes)
        self.nodes = nodes
        self.input_shape = input_shape
        self.kernels_per_layer = kernels_per_layer
        self.kernel_sizes = kernel_sizes
        self.dense_units = dense_units
        self.dropout_probability = dropout_probability
        self.classes = classes
        self.kfold = kfold
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size

    @staticmethod
    def generate_random_genes(genome):
        """Create and return random genes."""
        genes = {}
        for name, connections in genome.items():
            genes[name] = ''.join([random.choice(['0', '1']) for _ in range(connections)])
        return genes

    def evaluate_fitness(self):
        """Create model and perform cross-validation."""
        model = GeneticCnnModel(
            self.x_train, self.y_train, self.genes, self.nodes, self.input_shape, self.kernels_per_layer,
            self.kernel_sizes, self.dense_units, self.dropout_probability, self.classes,
            self.kfold, self.epochs, self.learning_rate, self.batch_size
        )
        self.fitness = model.cross_validate()

    def get_additional_parameters(self):
        return {
            'nodes': self.nodes,
            'input_shape': self.input_shape,
            'kernels_per_layer': self.kernels_per_layer,
            'kernel_sizes': self.kernel_sizes,
            'dense_units': self.dense_units,
            'dropout_probability': self.dropout_probability,
            'classes': self.classes,
            'kfold': self.kfold,
            'epochs': self.epochs,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size
        }

    def mutate(self):
        """Mutate instance's genes with a certain probability."""
        for name, connections in self.get_genes().items():
            new_connections = ''.join([
                str(int(int(byte) != (random.random() < self.mutation_rate))) for byte in connections
            ])
            if new_connections != connections:
                self.set_fitness(None)  # A mutation means the individual has to be re-evaluated
                self.get_genes()[name] = new_connections

class CrowIndividual(object):
    """Basic definition of an individual containing
    reproduction and mutation methods. Do not instantiate,
    use a subclass which extends this object by defining a
    genome and a random individual generator.
    """

    def __init__(self,gpu, x_train, y_train, x_test,y_test, flight_length=13,awareness_probability=0.15,space=None, location=None, fitness=0,memory=None,best_fitness=0,last_location=None,id=None, nodes=(3,4,5),
                 input_shape=(28, 28, 1), kernels_per_layer=(64,128,256), kernel_sizes=((3,3), (3,3), (3,3)), dense_units=1024,
                 dropout_probability=0.5, classes=10, kfold=5, epochs=(3,), learning_rate=(1e-3,), batch_size=32):

        self.gpu=gpu
        if space is None:
            space = {'S_{}'.format(i + 1): int(K_s * (K_s - 1) / 2) for i, K_s in enumerate(nodes)}
        if location is None:
            location = self.fly_random_location(space)
        if memory is None:
            memory=location
        if last_location is None:
            last_location=location

        self.id=id
        #TODO: Add id in all initializations of crows
        self.flight_length=flight_length
        self.awareness_probability=awareness_probability
        # Set individual's attributes
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.space = space
        self.validate_space()
        self.location = location
        self.memory=memory
        self.last_location = last_location
        self.validate_location()
        self.validate_memory()
        self.fitness = fitness  # Until evaluated an individual fitness is unknown
        self.best_fitness = best_fitness
        # assert additional_parameters is None

        # Set additional parameters which are not tuned
        assert len(nodes) == len(kernels_per_layer) and len(kernels_per_layer) == len(kernel_sizes)
        self.nodes = nodes
        self.input_shape = input_shape
        self.kernels_per_layer = kernels_per_layer
        self.kernel_sizes = kernel_sizes
        self.dense_units = dense_units
        self.dropout_probability = dropout_probability
        self.classes = classes
        self.kfold = kfold
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        for name, connections in self.space.items():
            bit_string = self.location[name]
            if len(bit_string) < connections:
                for bit in range(connections - len(bit_string)):
                    bit_string = "0" + bit_string
            self.location[name] = bit_string

    def validate_space(self):
        """Check genome structure."""
        if type(self.space) != dict:
            raise TypeError("Space Coordinates must be a dictionary.")
        for gene, properties in self.space.items():
            if type(gene) != str:
                raise TypeError("Space Coordinate names must be strings.")

    def validate_location(self):
        """Check that genes are compatible with genome."""
        if set(self.space.keys()) != set(self.location.keys()):
            print(self.get_id())
            raise ValueError("Location passed don't correspond to individual's space.")

    def validate_memory(self):
        """Check that genes are compatible with genome."""
        if set(self.space.keys()) != set(self.memory.keys()):
            raise ValueError("Memory passed don't correspond to individual's space.")

    def get_location(self):
        """Return individual's genes."""
        return self.location

    def set_location(self,value):
        """Return individual's genes."""
        self.location=value

    def get_last_location(self):
        """Return individual's genes."""
        return self.last_location

    def get_memory(self):
        """Return individual's genes."""
        return self.memory

    def set_memory(self,value):
        """Return individual's genes."""
        self.memory=value

    def get_space(self):
        """Return individual's genome."""
        return self.space

    @staticmethod
    def fly_random_location(genome):
        """Create and return random genes."""
        location = {}
        for name, connections in genome.items():
            location[name] = ''.join([random.choice(['0', '1']) for _ in range(connections)])
        return location

    def evaluate_fitness(self):
        """Create model and perform cross-validation."""
        print(" [*] Evaluating Crow {}".format(self.id),"on ",self.get_location(),".")
        print(" [*] Fitness of Crow {}".format(self.id),"on location",self.get_last_location()," was {:.8f}".format(self.get_fitness()),".")
        print(" [*] Best known performance of Crow {}".format(self.id)," is","{:.8f}".format(self.get_best_fitness()),"on location",self.get_memory())

        #Todo: Print story line of memory update. Remove unnecessary prints on server side responce.
        model = GeneticCnnModel(self.gpu,
            self.x_train, self.y_train, self.x_test,self.y_test,self.location, self.nodes, self.input_shape, self.kernels_per_layer,
            self.kernel_sizes, self.dense_units, self.dropout_probability, self.classes,
            self.kfold, self.epochs, self.learning_rate, self.batch_size
        )
        self.loss,self.accuracy,self.mae,self.mse,self.msle= model.validate()#cross_validate()
        self.fitness = self.accuracy
        print(" [*] Performance of Crow {}".format(self.id)," is", "{:.8f}".format(self.get_fitness()), "on location", self.get_location())
        if self.best_fitness==None or self.best_fitness < self.fitness:
            print(" [*] Updating best known performance for Crow {}".format(self.id)," to", "{:.8f}".format(self.get_fitness()), "on location", self.get_location())
            self.memory=self.location
            self.best_fitness=self.fitness
        else:
            print(" [*] Best known performance for Crow {}".format(self.id)," remains the same ", "{:.8f}".format(self.get_best_fitness()), "on location", self.get_memory())


        return self.fitness

    def get_additional_parameters(self):
        return {
            'nodes': self.nodes,
            'input_shape': self.input_shape,
            'kernels_per_layer': self.kernels_per_layer,
            'kernel_sizes': self.kernel_sizes,
            'dense_units': self.dense_units,
            'dropout_probability': self.dropout_probability,
            'classes': self.classes,
            'kfold': self.kfold,
            'epochs': self.epochs,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size
        }

    def get_id(self):
        """Compute individual's fitness if necessary and return it."""
        # if self.fitness is None:
        #Hello
        # self.evaluate_fitness()
        return self.id

    def get_fitness(self):
        """Compute individual's fitness if necessary and return it."""
        # if self.fitness is None:
        #Hello
        # self.evaluate_fitness()
        return self.fitness

    def get_best_fitness(self):
        """Compute individual's fitness if necessary and return it."""
        # if self.fitness is None:
        #Hello
        # self.evaluate_fitness()
        return self.best_fitness

    def old_follow(self,crow):
        assert self.__class__ == crow.__class__  # Can only reproduce if they're the same species
        print("\n [*] The Crow {}".format(self.id),"on location",self.get_location(),"is following the Crow {}".format(crow.id),"on location",crow.get_location())

        self.last_location=self.location
        if (random.randint(1, 100)/100.00) > self.awareness_probability:
            print(" [*] The Crow {}".format(crow.id), " is not aware of being followed by the Crow {}".format(self.id))
            print(" [*] So the Crow {}".format(crow.id), " leads the Crow {}".format(self.id),"in direction of it's best known location", crow.get_memory())

            bin_xi="".join([self.get_location()[stage] for stage in self.get_location().keys()])
            bin_mj = "".join([crow.get_memory()[stage] for stage in crow.get_memory().keys()])
            diff = int(bin_mj, 2) - int(bin_xi, 2)
            if diff<0:
                diff=diff*-1
            bin_diff = str(bin(diff >> 1) + str(diff & 1)).replace("b", "")

            if len(bin_diff)<len(bin_xi):
                for i in range(len(bin_xi)-len(bin_diff)):
                    bin_diff="0"+bin_diff

            fl = random.randrange(0, self.flight_length, 1)
            print(" [*] The flight length of Crow {}".format(self.id), " is ",fl)

            index = random.sample(range(0, 14), fl)

            bin_distance = ""
            for i, c in enumerate(bin_xi):
                if i in index:
                    bin_distance += bin_diff[i]
                else:
                    bin_distance += "0"

            int_xiplus1 = int(bin_xi, 2) + int(bin_distance, 2)
            bin_xiplus1 = str(bin(int_xiplus1 >> 1) + str(int_xiplus1 & 1)).replace("b", "")
            if len(bin_xiplus1) > len(bin_xi):
                len_diff = len(bin_xiplus1) - len(bin_xi)
                bin_xiplus1 = bin_xiplus1[len_diff:]

            last=0
            for name, connections in self.space.items():
                end=last+connections
                bit_string=bin_xiplus1[last:end]
                if len(bit_string)<connections:
                    for bit in range(connections-len(bit_string)):
                        bit_string="0"+bit_string
                self.location[name] = bit_string
                last=end
        else:
            print(" [*] The Crow {}".format(crow.id), " is aware of being followed by the Crow {}".format(self.id))
            print(" [*] So the Crow {}".format(crow.id), " leads the Crow {}".format(self.id), "in direction of a random location")
            self.location = self.fly_random_location(self.space)

        print(" [*] The Crow {}".format(self.id), "reaches a new location ",self.get_location(),".")

    def follow(self,crow):
        assert self.__class__ == crow.__class__  # Can only reproduce if they're the same species
        print("\n [*] The Crow {}".format(self.id),"on location",self.get_location(),"is following the Crow {}".format(crow.id),"on location",crow.get_location())

        self.last_location=self.location
        if (random.randint(1, 100)/100.00) > self.awareness_probability:
            print(" [*] The Crow {}".format(crow.id), " is not aware of being followed by the Crow {}".format(self.id))
            print(" [*] So the Crow {}".format(crow.id), " leads the Crow {}".format(self.id),"in direction of it's best known location", crow.get_memory())

            bin_xi="".join([self.get_location()[stage] for stage in self.get_location().keys()])
            bin_mj = "".join([crow.get_memory()[stage] for stage in crow.get_memory().keys()])

            diff_pos = []
            for bit_pos, bit_xi in enumerate(bin_xi):
                if bit_xi != bin_mj[bit_pos]:
                    diff_pos.append(bit_pos)
            # print(diff_pos)

            # fl = random.randrange(0, self.flight_length, 1)
            if len(diff_pos)>0:
                from numpy import round,sqrt
                fl = random.randrange(0, int(round(sqrt(len(bin_xi)*len(diff_pos)-9))), 1)

                print(" [*] The flight length of Crow {}".format(self.id), " is ", fl)

                if fl > len(diff_pos):
                    index = random.sample([x for x in range(0, len(bin_xi)) if x not in diff_pos], fl - len(diff_pos))
                    diff_pos.extend(index)
                else:
                    diff_pos = random.sample(diff_pos, fl)
                # print(diff_pos)

                bin_xiplus1 = list(bin_xi)
                for i in diff_pos:
                    if bin_xiplus1[i] == '1':
                        bin_xiplus1[i] = '0'
                    else:
                        bin_xiplus1[i] = '1'

                bin_xiplus1 = "".join(bin_xiplus1)

                last=0
                for name, connections in self.space.items():
                    end=last+connections
                    bit_string=bin_xiplus1[last:end]
                    if len(bit_string)<connections:
                        for bit in range(connections-len(bit_string)):
                            bit_string="0"+bit_string
                    self.location[name] = bit_string
                    last=end
            else:
                print(" [*] The Crow {}".format(crow.id), " and the Crow {}".format(self.id), "are same location.")
                print(" [*] So the Crow {}".format(self.id), "stays at same location ", self.get_location(), ".")

        else:
            print(" [*] The Crow {}".format(crow.id), " is aware of being followed by the Crow {}".format(self.id))
            print(" [*] So the Crow {}".format(crow.id), " leads the Crow {}".format(self.id), "in direction of a random location")
            self.location = self.fly_random_location(self.space)

        print(" [*] The Crow {}".format(self.id), "reaches a new location ",self.get_location(),".")


    def get_fitness_status(self):
        """Return True if individual's fitness in known."""
        return self.fitness is not None

    def set_fitness(self, value):
        """Assign fitness."""
        self.fitness = value

    def set_best_fitness(self, value):
        """Assign fitness."""
        self.best_fitness = value

    def copy(self):
        """Copy instance."""
        individual_copy = self.__class__(
            self.x_train, self.y_train, self.x_test,self.y_test, self.flight_length,self.awareness_probability, self.space.copy(),self.location.copy(), self.memory.copy(), **self.get_additional_parameters()
        )
        individual_copy.set_fitness(self.fitness)
        return individual_copy

    def __str__(self):
        """Return genes which identify the individual."""
        return pprint.pformat(self.location)


if __name__=="__main__":

    gene1 = {"S_1":"010","S_2":"1001001001"}
    gene2 = {"S_1":"101","S_2":"1011011011"}


    string1="".join([gene1[stage] for stage in gene1.keys()])
    string2="".join([gene2[stage] for stage in gene2.keys()])
    print(string1)
    print(string1[0:3])
    exit()
    diff=int(string2,2)-int(string1,2)
    bindiff=str(bin(diff >> 1) + str(diff & 1)).replace("b","")
    fl=random.randint(0,13)
    index=random.sample(range(0,14),fl)
    print(index)
    print(string2)
    print(string1)
    print(bindiff)
    print(len(bindiff),len(string1))

    string3=""
    for i,c in enumerate(string1):
        if i in index:
            string3+=bindiff[i]
        else:
            string3+="0"
    print(string3)

    new_int=int(string1,2)+int(string3,2)
    new_bin = str(bin(new_int >> 1) + str(new_int & 1)).replace("b", "")
    if len(new_bin)>len(string1):
        len_diff=len(new_bin)-len(string1)
        new_bin=new_bin[len_diff:]
    print(new_bin)