"""
auxiliary_functions.py
======================

This module contains helper functions used throughout the Genetic Algorithm (GA)
SAT solver. These utilities evaluate whether candidate solutions (individuals)
satisfy SAT clauses and compute the total number of satisfied clauses.

Each clause is represented as a list of integers, where:
- A positive integer `x` corresponds to the variable `x`.
- A negative integer `-x` corresponds to the negation of variable `x`.

Each individual (genome) is represented as a list of booleans, where the `i`-th
element corresponds to the truth assignment of variable `i + 1`.

Example:
    >>> clause = [1, -2, 3]  # (x1 OR ¬x2 OR x3)
    >>> individual = [True, False, False]
    >>> is_clause_satisfied(clause, individual)
    True
"""

from typing import List

def is_clause_satisfied(clause: List[int], individual: List[bool]) -> bool:
    """
    Check whether a given CNF clause is satisfied by an individual's assignment.

    Args:
        clause (list[int]): A list of literals. Positive integers indicate
            normal variables; negative integers indicate negated variables.
        individual (list[bool]): The genome/chromosome represented as a list
            of boolean values, where each index corresponds to a variable.

    Returns:
        bool: True if the clause is satisfied by the individual's assignment,
        False otherwise.

    Example:
        >>> is_clause_satisfied([1, -2, 3], [True, False, False])
        True

    Notes:
        - Variable indices in clauses are assumed to start at 1.
        - If a clause is empty, the function always returns False.
    """
    for literal in clause:
        index = abs(literal) - 1
        if literal > 0 and individual[index]:
            return True
        elif literal < 0 and not individual[index]:
            return True
    return False

def number_of_sat_clauses(clauses: List[List[int]], individual: List[bool]) -> int:
    """
    Count the number of SAT clauses satisfied by a given individual.

    Args:
        clauses (list[list[int]]): A list of CNF clauses, each clause being
            a list of integer literals.
        individual (list[bool]): The genome/chromosome represented as a list
            of boolean values, where each index corresponds to a variable.

    Returns:
        int: The total number of clauses satisfied by the individual's assignment.

    Example:
        >>> clauses = [[1, -2], [-1, 2, 3]]
        >>> individual = [True, False, True]
        >>> number_of_sat_clauses(clauses, individual)
        2

    Notes:
        - The number of satisfied clauses can range from 0 to len(clauses).
        - Empty clause lists will return 0.
    """
    return sum(is_clause_satisfied(clause, individual) for clause in clauses)