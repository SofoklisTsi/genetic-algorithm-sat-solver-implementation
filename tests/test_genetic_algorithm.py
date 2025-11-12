import numpy as np
from genetic_algorithm_sat_solver_implementation.genetic_algorithm.genetic_algorithm_model import GeneticAlgorithmModel
from genetic_algorithm_sat_solver_implementation.genetic_algorithm.genetic_algorithm import GeneticAlgorithm

def make_simple_model():
    return GeneticAlgorithmModel(
        clauses=[[1, -2], [-1, 2]],
        num_clauses=2,
        population_size=6,
        genome_length=2,
        max_generations=10,
    )

def test_evaluate_fitness_with_simple_clauses():
    model = make_simple_model()
    ga = GeneticAlgorithm(config=model)
    ga.evaluate_fitness()
    assert len(model.fitness) == model.population_size
    assert np.all((model.fitness >= 0) & (model.fitness <= 1))

def test_update_best_tracks_progress():
    model = make_simple_model()
    ga = GeneticAlgorithm(config=model)
    ga.evaluate_fitness()
    ga.update_best()

    # Check fitness sanity
    assert 0.0 <= model.best_fitness <= 1.0

    # Handle both cases: found or not found yet
    if model.best_fitness > 0:
        # A valid best individual should be stored
        assert len(model.best_individual) == model.genome_length
    else:
        # If all individuals had zero fitness, no best individual is expected yet
        assert model.best_individual == []

def test_run_completes_and_returns_model():
    model = make_simple_model()
    ga = GeneticAlgorithm(config=model)
    result = ga.run()
    assert isinstance(result, GeneticAlgorithmModel)
    assert result.generations > 0
    assert len(result.best_individual) == result.genome_length
