import numpy as np
from .genetic_algorithm_model import GeneticAlgorithmModel
from .auxilary_functions import number_of_sat_clauses
from typing import List

class GeneticAlgorithm:

    def __init__(self, config: GeneticAlgorithmModel):
        self.config = config

    # -----------------------------
    # Fitness
    # -----------------------------
    def evaluate_fitness(self) -> None:
        self.config.total_fitness = 0
        # Ensure arrays are correctly sized
        self.config.fitness = np.zeros(self.config.population_size, dtype=float)
        self.config.normalized_fitness = np.zeros(self.config.population_size, dtype=float)
        self.config.cumulative_fitness = np.zeros(self.config.population_size, dtype=float)

        for i, individual in enumerate(self.config.population):
            # Compute fitness for each individual
            self.config.fitness[i] = number_of_sat_clauses(
                clauses=self.config.clauses, individual=individual) / self.config.num_clauses

    # -----------------------------
    # Selection (tournament)
    # -----------------------------
    def tournament_selection(self, k=3):
        selected = []
        for _ in range(self.config.population_size):
            competitors = np.random.choice(
                range(self.config.population_size), size=k, replace=False)
            winner = max(competitors, key=lambda i: self.config.fitness[i])
            selected.append(self.config.population[winner])
        return selected

    # -----------------------------
    # Crossover
    # -----------------------------
    def crossover(self, parents: np.ndarray) -> np.ndarray:
        """
        Single-point crossover between pairs of parents.

        Args:
            parents (np.ndarray): Array of parent individuals.

        Returns:
            np.ndarray: Array of offspring.
        """
        num_offspring = self.config.population_size
        offspring = parents 
        for i in range(0, num_offspring, 2):
            rng = np.random.default_rng()
            if rng.random() < self.config.crossover_rate:
                parent1 = parents[i]
                parent2 = parents[i + 1 if i + 1 < num_offspring else 0]
                point = np.random.randint(1, self.config.genome_length)
                offspring[i] = np.concatenate([parent1[:point], parent2[point:]])
                if i + 1 < num_offspring:
                    offspring[i + 1] = np.concatenate([parent2[:point], parent1[point:]])
        return offspring

    # -----------------------------
    # Mutation
    # -----------------------------
    def mutate(self, offspring: np.ndarray) -> np.ndarray:
        """
        Mutate offspring individuals based on mutation rate.

        Args:
            offspring (np.ndarray): Array of offspring individuals.

        Returns:
            np.ndarray: Mutated offspring.
        """
        for i in range(self.config.population_size):
            for j in range(self.config.genome_length):
                if np.random.random() < self.config.mutation_rate:
                    offspring[i][j] = not offspring[i][j]
        return offspring 

    # -----------------------------
    # Evolution loop
    # -----------------------------
    def run(self):
        """
        Run the GA evolution for a number of generations.
        """
        num_generations = self.config.max_generations
        if num_generations == 0:
            num_generations = 100  # default if max_generations=0

        for gen in range(num_generations):
            # Evaluate fitness
            self.evaluate_fitness()

            # write the if in such a way that the best individual is stored
            # if np.any(self.config.fitness == 1.0):
            if np.max(self.config.fitness) == 1.0:
                best_idx = np.argmax(self.config.fitness)
                self.config.best_individual = self.config.population[best_idx]
                self.config.generations = gen + 1
                return self.config  # Solution found

            # Select parents by tournament
            parents = self.tournament_selection()

            # Generate offspring via crossover
            offspring = self.crossover(parents)

            # Mutate offspring
            self.population = self.mutate(offspring)

        # Find the best individual after all generations
        best_idx = np.argmax(self.config.fitness)
        self.config.best_individual = self.config.population[best_idx]

        # Set the number of generations completed
        self.config.generations = num_generations
        
        return self.config