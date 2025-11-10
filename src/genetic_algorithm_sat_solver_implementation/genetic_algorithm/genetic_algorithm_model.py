import numpy as np
from typing import List
from pydantic import BaseModel, Field, field_validator, model_validator

class GeneticAlgorithmModel(BaseModel):
    model_config = {
        "arbitrary_types_allowed": True
    }
        
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
            "The maximum number of generations to evolve. "
            "A value of 0 indicates that there is no generation limit."
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
    total_fitness: float = Field(0.0, init=False, description=(
        "Sum of all individual fitness values (computed internally)."
    ))
    normalized_fitness: List[float] = Field(default_factory=list, init=False, description=(
        "Normalized fitness of each individual (fitness / total_fitness)."
    ))
    cumulative_fitness: List[float] = Field(default_factory=list, init=False, description=(
        "Cumulative sum of normalized fitness values (used for selection)."
    ))
    generations: int = Field(0, init=False, description=(
        "The number of generations that have been evolved so far."
    ))
    best_individual: List[bool] = Field(default_factory=list, init=False, description=(
        "The best individual found so far in the evolution process."
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
        Validates that the maximum number of generations is non-negative.

        Args:
            max_generations (int): The declared maximum number of generations.

        Returns:
            int: The validated maximum number of generations.

        Raises:
            ValueError: If the maximum number of generations is less than 0.
        """
        if max_generations < 0:
            raise ValueError("Maximum number of generations cannot be less than 0.")
        return max_generations
    
    @field_validator('crossover_rate')
    def validate_crossover_rate(cls, crossover_rate: float) -> float:
        if not (crossover_rate > 0 and crossover_rate <= 1):
            raise ValueError("Crossover rate has to be bigger than 0 and equal or less than 1.")
        return crossover_rate
    
    @field_validator('mutation_rate')
    def validate_mutation_rate(cls, mutation_rate: float) -> float:
        if not (mutation_rate >= 0 and mutation_rate <= 1):
            raise ValueError("Mutation rate has to be equal or bigger than 0 and equal or less than 1.")
        return mutation_rate
    
    # -----------------------------
    # Model-level validator for post-init population creation
    # -----------------------------
    
    @model_validator(mode='after')
    def initialize_population(self) -> 'GeneticAlgorithmModel':
        """
        Initializes the GA population after model creation.

        Generates a population of `population_size` individuals,
        each with `genome_length` random bool genes.

        Returns:
            GeneticAlgorithmModel: The current instance with the population initialized.
        """
        rng = np.random.default_rng()
        self.population = [
            [rng.choice([True, False]) for gene_idx in range(self.genome_length)]
            for individual_idx in range(self.population_size)
        ]
        return self