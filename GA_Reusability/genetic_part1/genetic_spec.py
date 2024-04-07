# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 11:24:15 2022

@author: agademer & tdrumond

Template for exercise 1
(genetic algorithm module specification)
"""
import mastermind as mm
import random

def new_indidual_from_2_parents(first_parent, second_parent):
    x_point = random.randrange(0, len(first_parent.chromosome))
    new_chromose = first_parent.chromosome[0:x_point] + second_parent.chromosome[x_point:]
    return Individual(new_chromose, MATCH.rate_guess(new_chromose))

def mutation(parent):
    possibles_colors = mm.get_possible_colors()
    new_gene = random.choice(possibles_colors)
    pos = random.randint(0, len(parent.chromosome)-1)
    new_chrom = parent.chromosome[0:pos] + [new_gene] + parent.chromosome[pos+1:]
    return Individual(new_chrom, MATCH.rate_guess(new_chrom))

class Individual:
    """Represents an Individual for a genetic algorithm"""

    def __init__(self, chromosome: list, fitness: float):
        """Initializes an Individual for a genetic algorithm 

        Args:
            chromosome (list[]): a list representing the individual's chromosome
            fitness (float): the individual's fitness (the higher, the better the fitness)
        """
        self.chromosome = chromosome
        self.fitness = fitness

    def __lt__(self, other):
        """Implementation of the less_than comparator operator"""
        return self.fitness < other.fitness

    def __repr__(self):
        """Representation of the object for print calls"""
        return f'Indiv({self.fitness:.1f},{self.chromosome})'


class GASolver:
    def __init__(self, selection_rate=0.5, mutation_rate=0.1):
        """Initializes an instance of a ga_solver for a given GAProblem

        Args:
            selection_rate (float, optional): Selection rate between 0 and 1.0. Defaults to 0.5.
            mutation_rate (float, optional): mutation_rate between 0 and 1.0. Defaults to 0.1.
        """
        self._selection_rate = selection_rate
        self._mutation_rate = mutation_rate
        self._population = []

    def reset_population(self, pop_size=50):
        """ Initialize the population with pop_size random Individuals """
        chromosomes = [MATCH.generate_random_guess()  for _ in range(pop_size)]
        self._population = [Individual(chromosome, MATCH.rate_guess(chromosome)) for chromosome in chromosomes]

    def evolve_for_one_generation(self):
        """ Apply the process for one generation : 
            -	Sort the population (Descending order)
            -	Selection: Remove x% of population (less adapted)
            -   Reproduction: Recreate the same quantity by crossing the 
                surviving ones 
            -	Mutation: For each new Individual, mutate with probability 
                mutation_rate i.e., mutate it if a random value is below   
                mutation_rate
        """
        # remove weakest individuals
        self._population.sort(reverse=True)
        proportion_to_delete = self._selection_rate * len(self._population)
        del self._population[int(proportion_to_delete):]

        # Reproduction // Add the same number of deleted parents
        self._population = self._population + [new_indidual_from_2_parents(random.choice(self._population), random.choice(self._population))
                                               for _ in range(int(proportion_to_delete))]

        # Mutation

        self._population = [mutation(child_added) if random.random() < self._mutation_rate else child_added
                            for child_added in self._population[int(proportion_to_delete):]] \
                            + self._population[:int(proportion_to_delete)]

        self._population.sort()



    def show_generation_summary(self):
        """ Print some debug information on the current state of the population """
        print("Genzration summary:")
        print(f"current population: {self._population}")
        print(f"Population size: {len(self._population)}")
        print(f"Best individual: {self.get_best_individual()}")
        print(f"Best fitness score: {self.get_best_individual().fitness}")

    def get_best_individual(self):
        """ Return the best Individual of the population """
        self._population.sort()
        return self._population[-1]

    def evolve_until(self, max_nb_of_generations=500, threshold_fitness=None):
        """ Launch the evolve_for_one_generation function until one of the two condition is achieved : 
            - Max nb of generation is achieved
            - The fitness of the best Individual is greater than or equal to
              threshold_fitness
        """
        for _ in range(max_nb_of_generations):
            self.evolve_for_one_generation()
            if self._population[-1].fitness >= threshold_fitness:
                break


MATCH = mm.MastermindMatch(secret_size=4)
solver = GASolver()
solver.reset_population()
solver.evolve_until(threshold_fitness=MATCH.max_score())

best = solver.get_best_individual()
print(f"Best guess {best.chromosome}")
print(f"Problem solved? {MATCH.is_correct(best.chromosome)}")