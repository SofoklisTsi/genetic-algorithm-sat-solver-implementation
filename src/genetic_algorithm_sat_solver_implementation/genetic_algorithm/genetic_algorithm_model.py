"""
genetic_algorithm_model.py
==========================

This module defines the `GeneticAlgorithmModel`, a Pydantic-based data model
that encapsulates the core parameters and state of a Genetic Algorithm (GA)
used to solve Boolean Satisfiability (SAT) problems.

The model ensures that all hyperparameters (such as population size, mutation rate,
and crossover rate) are valid and consistent, and automatically initializes
a randomized population of candidate solutions upon creation.

Typical usage example:

    >>> from genetic_algorithm_model import GeneticAlgorithmModel
    >>> model = GeneticAlgorithmModel(
    ...     clauses=[[1, -2, 3], [-1, 2]],
    ...     num_clauses=2,
    ...     population_size=50,
    ...     genome_length=3
    ... )
    >>> print(model.population.shape)
    (50, 3)

The population is represented as a NumPy boolean matrix, where each row is an individual
(chromosome), and each column represents a gene (variable assignment).

Key Responsibilities:
- Validate GA parameters via Pydantic field validators.
- Maintain internal GA state (fitness values, generations, etc.).
- Initialize the population upon model creation.

This model serves as the foundation for the genetic algorithm engine that evolves
candidate solutions toward satisfying all SAT clauses.
"""

import numpy as np
from typing import List
from pydantic import BaseModel, Field, field_validator, model_validator

class GeneticAlgorithmModel(BaseModel):
    """
    Represents the configuration and state of a Genetic Algorithm (GA) for SAT solving.

    This model uses Pydantic to validate and document parameters that define
    the genetic algorithm’s behavior and evolution process. It also stores
    runtime information such as population, fitness scores, and best individuals.

    Attributes:
        clauses (List[List[int]]): List of clauses representing the SAT problem.
        num_clauses (int): The total number of clauses.
        population_size (int): Number of individuals in each generation.
        genome_length (int): Number of genes per individual (problem variables).
        max_generations (int): Maximum allowed number of generations to evolve.
        crossover_rate (float): Probability that two individuals crossover.
        mutation_rate (float): Probability that a gene mutates.
        population (np.ndarray): Boolean matrix representing the population.
        fitness (List[float]): Fitness scores of each individual.
        generations (int): Number of generations evolved.
        best_individual (List[bool]): Best individual found so far.
        best_fitness (float): Fitness of the best individual found so far.
    """

    # ----------------------------------------
    # Configuration
    # ----------------------------------------

    model_config = {"arbitrary_types_allowed": True}
    DEFAULT_MAX_GENERATIONS: int = 100
        
    clauses: List[List[int]] = Field(..., description="List of clauses, each containing literals")
    num_clauses: int = Field(..., description="The total number of clauses in the SAT problem.")
    population_size: int = Field(..., description= (
            "The number of individuals in each generation. "
            "Must be greater than 0."
        )
    )    
    genome_length: int = Field(..., description=(
            "Specifies the length of each genome or chromosome, "
            "i.e., the total number of genes contained within an individual. "    
            "Must be greater than 0."
        )
    )
    max_generations: int = Field(0, description=(
            "Maximum number of generations to evolve. "
            "If set to 0, it defaults to 100 generations."
        )
    )
    crossover_rate: float = Field(0.7, description= (
            "The chance of two individuals to crossover. "
            "The value has to be bigger than 0 and equal or less than 1."
        )
    )
    mutation_rate: float = Field(0.1, description= (
            "The chance of a gene to mutate. "
            "The value has to be equal or bigger than 0 and equal or less than 1."
        )
    )
    population: np.ndarray = Field(
        default_factory=lambda: np.empty((0, 0), dtype=bool),
        exclude=True,
        description=(
            "The GA population as a NumPy boolean matrix of shape "
            "(population_size, genome_length). Each row is an individual."
        ),
    )

    # -----------------------------
    # Internal GA state fields
    # -----------------------------

    fitness: List[float] = Field(default_factory=list, init=False, description=(
        "Fitness value for each individual in the population (computed internally, values 0-1)."
    ))
    generations: int = Field(0, init=False, description=(
        "The number of generations that have been evolved so far."
    ))
    best_individual: List[bool] = Field(default_factory=list, init=False, description=(
        "The best individual found so far in the evolution process."
    ))
    best_fitness: float = Field(0.0, init=False, description=(
        "The fitness value of the best individual found so far."
    ))

    # -----------------------------
    # Field Validators
    # -----------------------------

    @field_validator('population_size')
    def validate_population_number(cls, population_size: int) -> int:
        """
        Validates that the population size is greater than zero.

        Args:
            population_size (int): The declared number of individuals.

        Returns:
            int: The validated population size.

        Raises:
            ValueError: If the population size is zero or negative.
        """
        if population_size <= 0:
            raise ValueError("Population size must be greater than 0.")
        return population_size
    
    @field_validator('genome_length')
    def validate_genome_length(cls, genome_length: int) -> int:
        """
        Validates that the genome length is a positive integer.

        Args:
            genome_length (int): The declared number of genes per genome.

        Returns:
            int: The validated genome length.

        Raises:
            ValueError: If the genome length is zero or negative.
        """
        if genome_length <= 0:
            raise ValueError("Genome length must be greater than 0.")
        return genome_length

    @field_validator('max_generations')
    def validate_generation_limit(cls, max_generations: int) -> int:
        """
        Ensure that the maximum number of generations is non-negative.

        If the user sets `max_generations` to 0, it is automatically
        replaced with the default value of 100.

        Args:
            max_generations (int): Declared maximum number of generations.

        Returns:
            int: The validated (and possibly adjusted) number of generations.

        Raises:
            ValueError: If the provided number is negative.
        """
        if max_generations < 0:
            raise ValueError("Maximum number of generations cannot be negative.")
        if max_generations == 0:
            max_generations = cls.DEFAULT_MAX_GENERATIONS
        return max_generations
    
    @field_validator('crossover_rate')
    def validate_crossover_rate(cls, crossover_rate: float) -> float:
        """
        Ensure the crossover rate is within the valid range [0, 1].

        Args:
            crossover_rate (float): Probability of crossover.

        Returns:
            float: The validated crossover rate.

        Raises:
            ValueError: If not within the interval [0, 1].
        """
        if not (crossover_rate >= 0 and crossover_rate <= 1):
            raise ValueError("Crossover rate has to be equal or bigger than 0 and equal or less than 1.")
        return crossover_rate
    
    @field_validator('mutation_rate')
    def validate_mutation_rate(cls, mutation_rate: float) -> float:
        """
        Ensure the mutation rate is within the valid range [0, 1].

        Args:
            mutation_rate (float): Probability of mutation.

        Returns:
            float: The validated mutation rate.
            
        Raises:
            ValueError: If not within the interval [0, 1].
        """
        if not (mutation_rate >= 0 and mutation_rate <= 1):
            raise ValueError("Mutation rate has to be equal or bigger than 0 and equal or less than 1.")
        return mutation_rate
    
    # -----------------------------
    # Model-level validator for post-init population creation
    # -----------------------------
    
    @model_validator(mode="after")
    def initialize_population(self) -> "GeneticAlgorithmModel":
        """
        Initialize the GA population after model creation.

        Generates a random boolean population of shape
        (population_size, genome_length), where each gene is
        randomly assigned `True` or `False`.

        Returns:
            GeneticAlgorithmModel: The model instance with initialized population.
        """
        rng = np.random.default_rng()
        self.population = rng.choice(
            [True, False],
            size=(self.population_size, self.genome_length),
        )
        return self