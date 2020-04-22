#!/usr/bin/env python
"""
Population class
"""

import itertools
import operator


class  Population(object):
    """Group of individuals of the same species, that is,
    with the same genome. Can be initialized either with a
    list of individuals or a population size so that
    random individuals are created. The get_fittest method
    returns the strongest individual.
    """

    def __init__(self, species, x_train, y_train, input_shape, nb_classes,individual_list=None, size=None,
                 crossover_rate=0.5, mutation_rate=0.015, maximize=True,
                 additional_parameters=None):
        self.x_train = x_train
        self.y_train = y_train
        self.input_shape=input_shape
        self.nb_classes=nb_classes
        self.species = species
        self.maximize = maximize
        if individual_list is None and size is None:
            raise ValueError("Either pass a list of individuals or a population size for a random population.")
        elif individual_list is None:
            if additional_parameters is None:
                additional_parameters = {}
            self.population_size = size
            self.individuals = [
                self.species(
                    self.x_train, self.y_train, input_shape=self.input_shape, classes=self.nb_classes,crossover_rate=crossover_rate,
                    mutation_rate=mutation_rate, **additional_parameters
                )
                for _ in range(size)
            ]
            print("Initializing a random population. Size: {}".format(size))
        else:
            assert all([type(individual) is self.species for individual in individual_list])
            self.population_size = len(individual_list)
            self.individuals = individual_list

    def add_individual(self, individual):
        assert type(individual) is self.species
        self.individuals.append(individual)
        self.population_size += 1

    def get_species(self):
        return self.species

    def get_size(self):
        return self.population_size

    def get_fittest(self):
        if self.maximize:
            return max(self.individuals, key=operator.methodcaller('get_fitness'))
        return min(self.individuals, key=operator.methodcaller('get_fitness'))

    def get_data(self):
        return self.x_train, self.y_train

    def get_fitness_criteria(self):
        return self.maximize

    def __getitem__(self, item):
        return self.individuals[item]


class  Flock(object):
    """Group of individuals of the same species, that is,
    with the same genome. Can be initialized either with a
    list of individuals or a population size so that
    random individuals are created. The get_fittest method
    returns the strongest individual.
    """

    def __init__(self, species, x_train, y_train, input_shape, nb_classes,individual_list=None, size=None,
                 flight_length=13, awareness_probability=0.15, maximize=True,
                 additional_parameters=None):
        self.x_train = x_train
        self.y_train = y_train
        self.input_shape=input_shape
        self.nb_classes=nb_classes
        self.species = species
        self.maximize = maximize
        self.explored=[]
        self.individuals=[]
        if individual_list is None and size is None:
            raise ValueError("Either pass a list of crows or a flock size for a random flock.\n")
        elif individual_list is None:
            if additional_parameters is None:
                additional_parameters = {}
            self.flock_size = size
            print(self.explored)
            for i in range(size):
                crow=self.species(self.x_train, self.y_train, flight_length,awareness_probability,id=i,input_shape=self.input_shape, classes=self.nb_classes,**additional_parameters)
                while crow.get_location() in self.explored:
                    print (crow.get_location(), "already created")
                    crow.fly_random_location(crow.get_space())

                self.explored.append(crow.get_location())
                self.individuals.append(crow)
            print(self.explored)
            # self.individuals = [
            #     self.species(
            #         self.x_train, self.y_train, flight_length,awareness_probability,id=i,input_shape=self.input_shape, classes=self.nb_classes,**additional_parameters
            #     )
            #     for i in range(size)
            # ]
            print("Initializing a random flock. Size: {}\n".format(size))
            for individual in self.individuals:
                print(individual.get_id(),individual.get_location())
        else:
            self.individuals = [
                self.species(
                    self.x_train, self.y_train, flight_length, awareness_probability, id=i, location=location,
                    input_shape=self.input_shape, classes=self.nb_classes, **additional_parameters
                )
                for i,location in enumerate(individual_list)
            ]
            assert all([type(individual) is self.species for individual in self.individuals])
            self.flock_size = len(individual_list)
            # self.individuals = individual_list

    def add_individual(self, individual):
        assert type(individual) is self.species
        self.individuals.append(individual)
        self.flock_size += 1

    def get_species(self):
        return self.species

    def get_size(self):
        return self.flock_size

    def get_fittest(self):
        if self.maximize:
            return max(self.individuals, key=operator.methodcaller('get_best_fitness'))
        return min(self.individuals, key=operator.methodcaller('get_best_fitness'))

    def get_data(self):
        return self.x_train, self.y_train

    def get_fitness_criteria(self):
        return self.maximize

    def __getitem__(self, item):
        return self.individuals[item]


class GridPopulation(Population):
    """Population whose individuals are created based on a
     grid search approach instead of randomly. Can be
     initialized either with a list of individuals (in
     which case it behaves like a Population) or with a
     dictionary of genes and grid values pairs.
     """

    def __init__(self, species, x_train, y_train, individual_list=None, genes_grid=None,
                 crossover_rate=0.5, mutation_rate=0.015, maximize=True,
                 additional_parameters=None):
        if individual_list is None and genes_grid is None:
            raise ValueError("Either pass a list of individuals or a grid definition.")
        elif genes_grid is not None:
            genome = species(None, None).get_genome()  # Get species' genome
            if not set(genes_grid.keys()).issubset(set(genome.keys())):
                raise ValueError("Some grid parameters do not belong to the species' genome")
            # Fill genes_grid with default parameters
            for gene, properties in genome.items():
                if gene not in genes_grid:
                    genes_grid[gene] = [properties[0]]  # Use default value
            individual_list = [
                species(
                    x_train, y_train, genes=genes, crossover_rate=crossover_rate,
                    mutation_rate=mutation_rate, **additional_parameters
                )
                for genes in (
                    dict(zip(genes_grid, x))
                    for x in itertools.product(*genes_grid.values())
                )
            ]
            print("Initializing a grid population. Size: {}".format(len(individual_list)))
        super(GridPopulation, self).__init__(
            species, x_train, y_train, individual_list, None, crossover_rate, mutation_rate,
            maximize, additional_parameters
        )
