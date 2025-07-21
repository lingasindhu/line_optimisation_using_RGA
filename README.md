# Line-Optimisation Using RGA

This project implements a Genetic Algorithm (GA) using the Pymoo library to optimize chip-filling line sequences and movement speeds. The primary goal is to minimize the total processing time while satisfying critical geometric constraints, such as minimum turning radius and line spacing.

The algorithm begins by reading input parameters from a .txt file, which includes configuration details for lines, machine capabilities, and constraints. It then performs the following evolutionary steps:

Sampling: Generates an initial population of candidate line sequences.

Selection: Chooses the most promising candidates based on a fitness function related to time efficiency and constraint satisfaction.

Crossover and Mutation: Produces new candidate solutions by combining and modifying existing ones, promoting diversity and avoiding local optima.

By iteratively refining the line sequences, the algorithm identifies the optimal path and speed configuration for the chip-filling machine, ensuring efficient and constraint-aware operation.
