from typing import List

def is_clause_satisfied(clause: List[int], individual: List[bool]) -> bool:
    """
    Checks whether a given CNF clause is satisfied by an individual.

    Args:
        clause (List[int]): A list of literals. Positive integers indicate normal variables,
                            negative integers indicate negated variables.
        individual (List[bool]): The genome/chromosome represented as a list of boolean genes.

    Returns:
        bool: True if the clause is satisfied, False otherwise.
    """
    for lit in clause:
        if lit > 0 and individual[lit-1]:
            return True
        elif lit < 0 and not individual[abs(lit)-1]:
            return True
    return False

def number_of_sat_clauses(clauses: List[List[int]], individual: List[bool]) -> int:
    """
    Counts the number of clauses satisfied by an individual.

    Args:
        clauses (List[List[int]]): List of CNF clauses (each clause is a list of literals).
        individual (List[bool]): The genome/chromosome represented as a list of boolean genes.

    Returns:
        int: Number of satisfied clauses.
    """
    return sum(is_clause_satisfied(clause, individual) for clause in clauses)