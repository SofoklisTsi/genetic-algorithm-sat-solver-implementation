# 🧬 Genetic Algorithm SAT Solver

A Python implementation of a **Genetic Algorithm (GA)**–based SAT solver, inspired by the methodology introduced in  
**“Using Genetic Algorithms to Solve NP-Complete Problems” (De Jong & Spears, 1998)**.  
This project demonstrates how GAs can be applied to Boolean satisfiability (SAT) as a *canonical NP-complete problem*, leveraging evolutionary search to explore solution spaces efficiently.

> 📖 *Reference:*  
> De Jong, K. A. & Spears, W. M. (1998).  
> *Using Genetic Algorithms to Solve NP-Complete Problems.*  
> [ResearchGate Link](https://www.researchgate.net/publication/2718690_Using_Genetic_Algorithms_to_Solve_NP-Complete_Problems)

---

## 🧠 Background

De Jong and Spears proposed SAT as a **GA-effective canonical NP-complete problem**, arguing that its natural binary representation makes it well-suited for evolutionary computation.  
This implementation captures that idea by encoding SAT clauses as fixed-length binary chromosomes and applying GA operators (selection, crossover, mutation) to evolve satisfying assignments.

This project also reuses and extends components from my previous SAT solver work:  
👉 [SofoklisTsi/SAT_SOLVER](https://github.com/SofoklisTsi/SAT_SOLVER)

---

## 📂 Project Structure
```bash
src/
└── genetic_algorithm_sat_solver_implementation/
├── data/ # Sample CNF problems (DIMACS format)
├── dimacs_reader/ # DIMACS CNF file parsing
├── genetic_algorithm/ # Core GA implementation
│ ├── genetic_algorithm_model.py
│ ├── genetic_algorithm.py
│ ├── auxilary_functions.py
└── run_ga_sat_solver.py
tests/ # Unit tests
```

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/SofoklisTsi/genetic-algorithm-sat-solver-implementation.git
cd genetic-algorithm-sat-solver-implementation

# Install dependencies via Poetry
poetry install
```

## 🚀 Usage

Run the solver on one of the provided CNF instances:

poetry run python src/genetic_algorithm_sat_solver_implementation/run_ga_sat_solver.py

You can modify the clause_path variable in run_ga_sat_solver.py to select a different CNF file under data/.

### Example output

```bash
Clauses: [[1, -3, 4], [-1, 2], ...]
Number of Variables: 20
Number of Clauses: 91
Population number: 400
Best Individual: [True, False, True, ...], fitness: 0.982
Generations: 145
```

## 🧩 Key Components

### GeneticAlgorithmModel
Defines GA parameters and runtime state (population, fitness, etc.).
Automatically validates and initializes defaults (e.g., max_generations=100 if unspecified).

### GeneticAlgorithm
- Core GA logic including:
    - Tournament selection
    - Single-point crossover
    - Bit-flip mutation
    - Fitness evaluation via satisfied clauses
    - Automatic tracking of the best individual

### dimacs_reader
Reads standard DIMACS CNF files and constructs internal clause models.

## 🧪 Testing

Tests are implemented with pytest and automatically run on every GitHub push via GitHub Actions.

Run tests locally:
```bash
poetry run pytest
```

## 🧱 Future Work

- Incorporate alternative fitness evaluation schemes (e.g., differential payoffs, AVE^p as in the original paper).

- Add restart-based population diversity control.

- Parallelize the GA execution to match the multi-run experiments described by De Jong & Spears.

- Compare performance to classical SAT solvers and simulated annealing.

## 🙌 Acknowledgments

- Kenneth A. De Jong and William M. Spears for the foundational research.

- Portions of the DIMACS reader were adapted from my previous work: SAT_SOLVER.

- This project serves as a research and educational implementation of their proposed GA–SAT framework.

## 📜 License

MIT License © 2025 Sofoklis Evangelos Tsiakalos