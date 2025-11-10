from typing import List
from dimacs_reader import ClauseReader, ClausesModel
from genetic_algorithm import GeneticAlgorithmModel, GeneticAlgorithm
import numpy as np

# Path to the CNF file
# clause_path = "./data/bcp.cnf"
# clause_path = "./data/dpll1.cnf"
# clause_path = "./data/dpll2.cnf"
# clause_path = "./data/dpll5.cnf"
clause_path = "./data/uf20-01.cnf"

# Read the clauses from the file and print the clauses, number of variables and number of clauses.
clauses_model: ClausesModel = ClauseReader.read_file(clause_path)
clauses: List[List[int]] = clauses_model.clauses
num_vars: int = clauses_model.num_vars
num_clauses: int = clauses_model.num_clauses
print(f"Clauses: {clauses}")
print(f"Number of Variables: {num_vars}")
print(f"Number of Clauses: {num_clauses}")

# Determine the population number
# Normally num_vars^2 
# Exeptions:
#   num_vars = 1 --> population = 2 instead of: 1
#   num_vars = 2 --> population = 2 instead of: 4
#   num_vars = 3 --> population = 4 instead of: 9
#   num_vars = 4 --> population = 10 instead of: 16
# Limit:
#   num_vars >= 35 --> population = 1.225 instead of: A higher number
if num_vars < 1:
    print("Incorrect number of variables")
elif num_vars == 1:
    population_size = 2
elif num_vars == 2:
    population_size = 2
elif num_vars == 3:
    population_size = 4
elif num_vars == 4:
    population_size = 10
elif num_vars >= 35:
    population_size = 1225
else:
    population_size = np.power(num_vars, 2) 
print(f"Population number: {population_size}")

# Create the genetic algorithm instance 
# geneticAlgorithmModel = GeneticAlgorithmModel(
#     clauses=clauses, num_clauses=num_clauses, population_size=population_size, 
#     genome_length=num_vars, max_generations=0)
geneticAlgorithmModel = GeneticAlgorithmModel(
    clauses=clauses, num_clauses=num_clauses, population_size=population_size, 
    genome_length=num_vars, max_generations=1000)
geneticAlgorithm = GeneticAlgorithm(config=geneticAlgorithmModel)
results = geneticAlgorithm.run()
# for i, indi in enumerate(results.population):
#     print(f"Individual: {indi}, fitness: {results.fitness[i]}")
print(f"Best Individual: {results.best_individual}, fitness: {max(results.fitness)}")
print(f"Generations: {results.generations}")