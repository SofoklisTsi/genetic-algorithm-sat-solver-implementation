"""
run_ga_sat_solver.py
====================

Entry point for running the Genetic Algorithm SAT solver.

This script:
1. Reads a CNF problem file in DIMACS format.
2. Initializes GA parameters dynamically based on problem size.
3. Runs the Genetic Algorithm until completion.
4. Displays the best result found and summary statistics.

Usage:
    python run_ga_sat_solver.py
"""

from typing import List
from dimacs_reader import ClauseReader, ClausesModel
from genetic_algorithm import GeneticAlgorithmModel, GeneticAlgorithm
import numpy as np

def determine_population_size(num_vars: int) -> int:
    """
    Determine a suitable population size based on the number of variables.

    This uses a heuristic that scales approximately quadratically for small
    problems and caps for very large ones.

    Args:
        num_vars (int): Number of variables in the SAT problem.

    Returns:
        int: The computed population size.
    """
    if num_vars < 1:
        raise ValueError("Invalid number of variables. Must be >= 1.")
    elif num_vars == 1:
        return 2
    elif num_vars == 2:
        return 2
    elif num_vars == 3:
        return 4
    elif num_vars == 4:
        return 10
    elif num_vars >= 35:
        return 1225
    else:
        return int(np.power(num_vars, 2))
    
def main() -> None:
    """
    Main entry point for running the GA SAT solver.
    """
    # ------------------------------------------------------------------
    # 1. Problem setup
    # ------------------------------------------------------------------
    # Path to the CNF file
    # clause_path = "./data/bcp.cnf"
    # clause_path = "./data/dpll1.cnf"
    # clause_path = "./data/dpll2.cnf"
    # clause_path = "./data/dpll5.cnf"
    clause_path = "./data/uf20-01.cnf" 

    print("Reading CNF file...")
    clauses_model: ClausesModel = ClauseReader.read_file(clause_path)
    clauses: List[List[int]] = clauses_model.clauses
    num_vars: int = clauses_model.num_vars
    num_clauses: int = clauses_model.num_clauses

    print(f"Loaded CNF problem from: {clause_path}")
    print(f"  Variables: {num_vars}")
    print(f"  Clauses:   {num_clauses}")
    print()

    # ------------------------------------------------------------------
    # 2. Determine population size
    # ------------------------------------------------------------------
    population_size = determine_population_size(num_vars)
    print(f"Computed population size: {population_size}")
    print()

    # ------------------------------------------------------------------
    # 3. Initialize GA model and instance
    # ------------------------------------------------------------------
    ga_model = GeneticAlgorithmModel(
        clauses=clauses,
        num_clauses=num_clauses,
        population_size=population_size,
        genome_length=num_vars,
        max_generations=1000,  # you can adjust or make this configurable
    )
    ga = GeneticAlgorithm(config=ga_model)

    print("Initialized Genetic Algorithm with the following configuration:")
    ga.summary()

    # ------------------------------------------------------------------
    # 4. Run the Genetic Algorithm
    # ------------------------------------------------------------------
    print("Running evolution process...\n")
    results = ga.run()

    # ------------------------------------------------------------------
    # 5. Display results
    # ------------------------------------------------------------------
    print("\nEvolution complete.")
    print("-" * 40)
    print(f"Best individual: {results.best_individual}")
    print(f"Best fitness:    {results.best_fitness:.4f}")
    print(f"Generations run: {results.generations}")
    print("-" * 40)
    print()
    print("Final summary:")
    ga.summary()

if __name__ == "__main__":
    main()