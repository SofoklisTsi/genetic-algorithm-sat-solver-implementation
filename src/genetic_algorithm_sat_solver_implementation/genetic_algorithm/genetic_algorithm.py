"""
genetic_algorithm.py
====================

Implements the main Genetic Algorithm logic for Boolean Satisfiability (SAT) solving.

This module defines the `GeneticAlgorithm` class, which operates on a
`GeneticAlgorithmModel` instance to evolve populations toward satisfying
a set of CNF clauses.

See Also:
    genetic_algorithm_model.GeneticAlgorithmModel
        The data model defining configuration and runtime state for the GA.
"""

import numpy as np
from .genetic_algorithm_model import GeneticAlgorithmModel
from .auxilary_functions import number_of_sat_clauses
from typing import List

class GeneticAlgorithm:
    """
    Implements the core Genetic Algorithm (GA) for SAT solving.

    This class operates on a `GeneticAlgorithmModel` instance, which holds
    all configuration parameters and runtime state (e.g., population, fitness,
    and hyperparameters).

    Attributes:
        config (GeneticAlgorithmModel): The underlying model containing
            configuration and internal GA state. Refer to
            `genetic_algorithm_model.GeneticAlgorithmModel` for detailed
            parameter documentation.
    """

    def __init__(self, config: GeneticAlgorithmModel):
        """
        Initialize the Genetic Algorithm with a given configuration.

        Args:
            config (GeneticAlgorithmModel): The GA configuration and runtime state.
        """
        self.config = config

    # -----------------------------
    # Configuration Summary
    # -----------------------------
    def summary(self) -> None:
        """
        Print a formatted summary of the GA configuration and current state.

        This includes hyperparameters, population statistics, and the best
        individual found so far. It should be called at any point to inspect
        the current status of the algorithm.

        Example:
            >>> ga = GeneticAlgorithm(config)
            >>> ga.evaluate_fitness()
            >>> ga.update_best()
            >>> ga.summary()
        """
        cfg = self.config
        print("Genetic Algorithm Summary")
        print("=" * 40)

        # -----------------------------
        # Configuration section
        # -----------------------------

        print("Configuration:")
        print(f"  Population size : {cfg.population_size}")
        print(f"  Genome length   : {cfg.genome_length}")
        print(f"  Max generations : {cfg.max_generations}")
        print(f"  Crossover rate  : {cfg.crossover_rate}")
        print(f"  Mutation rate   : {cfg.mutation_rate}")
        print(f"  Number of clauses: {cfg.num_clauses}")
        print()

        # -----------------------------
        # Runtime state section
        # -----------------------------

        print("Runtime State:")
        # No fitness yet → GA hasn’t started
        if len(cfg.fitness) == 0:
            print("  Status           : Not yet evaluated")
            print("  Generations run  : 0")
            print("  Best fitness     : N/A")
            print("  Best individual  : N/A")
            print()
            print("Population Statistics: No fitness data yet.")
            print("=" * 40)
            print()
            return

        # Fitness exists, maybe no best yet
        print(f"  Generations run  : {cfg.generations}")
        if cfg.best_individual and cfg.best_fitness > 0:
            bitstring = ''.join('1' if b else '0' for b in cfg.best_individual)
            print(f"  Best fitness     : {cfg.best_fitness:.4f}")
            print(f"  Best individual  : {bitstring}")
        else:
            print("  Best fitness     : Not yet determined")
            print("  Best individual  : None")

        print("=" * 40)
        print()

    # -----------------------------
    # Fitness
    # -----------------------------
    def evaluate_fitness(self) -> None:
        """
        Evaluate and update the fitness of each individual in the population.

        The fitness of an individual is defined as the ratio of satisfied
        clauses to the total number of clauses in the SAT problem:

            fitness = (# satisfied clauses) / (total clauses)

        Each individual's fitness value is stored in `self.config.fitness`
        as a float between 0.0 and 1.0, where 1.0 represents a fully
        satisfying assignment (all clauses satisfied).

        This method should be called once per generation, before selection
        or mutation operations, to ensure that the fitness values are up to date.

        Effects:
            - Updates `self.config.fitness` in place with the computed values.

        Example:
            >>> ga = GeneticAlgorithm(config)
            >>> ga.evaluate_fitness()
            >>> ga.config.fitness[:5]
            array([0.8, 0.7, 0.9, 0.6, 0.85])
        """
        # Ensure arrays are correctly sized
        self.config.fitness = np.zeros(self.config.population_size, dtype=float)

        for i, individual in enumerate(self.config.population):
            # Compute fitness for each individual
            self.config.fitness[i] = number_of_sat_clauses(
                clauses=self.config.clauses, individual=individual) / self.config.num_clauses

    # -----------------------------
    # Best Individual Tracking
    # -----------------------------       
    def update_best(self) -> None:
        """
        Update the best individual and best fitness based on the current population.

        This should be called immediately after `evaluate_fitness()` to ensure
        that the best-known solution is always up to date.

        If multiple individuals share the same best fitness, the first one is selected.

        Example:
            >>> ga.evaluate_fitness()
            >>> ga.update_best()
        """
        if len(self.config.fitness) == 0:
            return  # No fitness data yet

        best_idx = int(np.argmax(self.config.fitness))
        best_fitness = float(self.config.fitness[best_idx])

        # Update if this generation found a better individual
        if best_fitness > self.config.best_fitness:
            self.config.best_fitness = best_fitness
            self.config.best_individual = list(self.config.population[best_idx])

    # -----------------------------
    # Tournament Selection
    # -----------------------------
    def tournament_selection(self, k: int = 3) -> np.ndarray:
        """
        Select parent individuals using tournament selection.

        Args:
            k (int): Number of individuals per tournament. Defaults to 3.

        Returns:
            np.ndarray: Array of selected parent individuals.
        """
        selected = []
        for _ in range(self.config.population_size):
            competitors = np.random.choice(
                range(self.config.population_size), size=k, replace=False)
            winner = max(competitors, key=lambda i: self.config.fitness[i])
            selected.append(self.config.population[winner])
        return np.array(selected, dtype=bool)

    # -----------------------------
    # Crossover
    # -----------------------------
    def crossover(self, parents: np.ndarray) -> np.ndarray:
        """
        Perform single-point crossover between pairs of parents.

        Args:
            parents (np.ndarray): Array of parent individuals.

        Returns:
            np.ndarray: Array of offspring.
        """
        offspring = parents.copy()
        rng = np.random.default_rng()

        for i in range(0, self.config.population_size, 2):
            if rng.random() < self.config.crossover_rate:
                parent1 = parents[i]
                parent2 = parents[i + 1 if i + 1 < self.config.population_size else 0]
                point = rng.integers(1, self.config.genome_length)
                offspring[i] = np.concatenate([parent1[:point], parent2[point:]])
                if i + 1 < self.config.population_size:
                    offspring[i + 1] = np.concatenate([parent2[:point], parent1[point:]])
        return offspring

    # -----------------------------
    # Mutation
    # -----------------------------
    def mutate(self, offspring: np.ndarray) -> np.ndarray:
        """
        Apply random mutations to offspring genes.

        Args:
            offspring (np.ndarray): Array of offspring individuals.

        Returns:
            np.ndarray: Mutated offspring.
        """
        rng = np.random.default_rng()
        for i in range(self.config.population_size):
            for j in range(self.config.genome_length):
                if rng.random() < self.config.mutation_rate:
                    offspring[i][j] = not offspring[i][j]
        return offspring

    # -----------------------------
    # Evolution loop
    # -----------------------------
    def run(self) -> GeneticAlgorithmModel:
        """
        Run the genetic algorithm for a specified number of generations.

        For each generation, this method performs the following steps:
            1. Evaluates fitness for all individuals using `evaluate_fitness()`.
            2. Updates the best individual and best fitness via `update_best()`.
            3. Checks for early termination if a perfect solution (fitness = 1.0) is found.
            4. Selects parents using tournament selection.
            5. Generates offspring through crossover.
            6. Applies mutation to the offspring.

        Returns:
            GeneticAlgorithmModel: The configuration model containing the final
            evolved population, fitness values, and the best individual found.
        """
        for gen in range(self.config.max_generations):
            # Evaluate fitness and update best individual
            self.evaluate_fitness()
            self.update_best()

            # Early stop on perfect fitness
            if self.config.best_fitness == 1.0:
                self.config.generations = gen + 1
                return self.config
            
            # Selection → Crossover → Mutation
            parents = self.tournament_selection()
            offspring = self.crossover(parents)
            self.config.population = self.mutate(offspring)

        # Set the number of generations completed
        self.config.generations = self.config.max_generations
        return self.config