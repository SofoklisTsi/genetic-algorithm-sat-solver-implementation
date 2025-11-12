import pytest
from genetic_algorithm_sat_solver_implementation.genetic_algorithm.genetic_algorithm_model import GeneticAlgorithmModel

def test_population_initialization_shape():
    model = GeneticAlgorithmModel(
        clauses=[[1, -2], [-1, 2]],
        num_clauses=2,
        population_size=10,
        genome_length=3,
    )
    assert model.population.shape == (10, 3)

def test_default_max_generations_is_100():
    model = GeneticAlgorithmModel(
        clauses=[[1, -2]],
        num_clauses=1,
        population_size=5,
        genome_length=2,
        max_generations=0,
    )
    assert model.max_generations == 100

@pytest.mark.parametrize("invalid_size", [0, -1])
def test_invalid_population_size_raises(invalid_size):
    with pytest.raises(ValueError):
        GeneticAlgorithmModel(
            clauses=[[1, 2]],
            num_clauses=1,
            population_size=invalid_size,
            genome_length=3,
        )
