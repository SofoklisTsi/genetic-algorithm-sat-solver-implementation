from genetic_algorithm_sat_solver_implementation.genetic_algorithm.auxilary_functions import (
    is_clause_satisfied,
    number_of_sat_clauses,
)

def test_is_clause_satisfied_basic():
    clause = [1, -2, 3]  # (x1 OR ¬x2 OR x3)
    individual = [True, False, False]
    assert is_clause_satisfied(clause, individual) is True

def test_is_clause_not_satisfied():
    clause = [1, 2]
    individual = [False, False]
    assert is_clause_satisfied(clause, individual) is False

def test_number_of_sat_clauses_counts_correctly():
    clauses = [[1, -2], [-1, 2]]
    individual = [True, False]
    assert number_of_sat_clauses(clauses, individual) == 1
