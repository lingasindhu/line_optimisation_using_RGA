import numpy as np
import matplotlib.pyplot as plt
from pymoo.core.sampling import Sampling
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.selection.tournament import TournamentSelection, compare
from pymoo.termination import get_termination
from pymoo.optimize import minimize
from pymoo.core.mutation import Mutation
from pymoo.core.survival import Survival
from pymoo.operators.sampling.lhs import LHS
import math
import itertools

# Define round_decimals globally for consistent output formatting
round_decimals = 4

# Read inputs from file
input_filename = "C:/Users/sindh/OneDrive/Desktop/New folder/inputs.txt"  # Specify your input file name

# ========== INPUT FROM FILE ==========
def read_inputs_from_file(filename):
    """
    Read inputs from a text file and validate them.
    File format: Each line contains 'variable=value' (e.g., 'n=5').
    Returns: Dictionary with validated input values.
    """
    inputs = {}
    required_vars = {
        'n': {'type': int, 'min_val': 2, 'max_val': 100},
        'B': {'type': float, 'min_val': 0.1},
        'i': {'type': int, 'min_val': 1},  # max_val depends on n
        'L': {'type': float, 'min_val': 0.0},
        'v_min': {'type': float, 'min_val': 0.1},
        'v_max': {'type': float},  # min_val depends on v_min
        'pop_size': {'type': int, 'min_val': 1, 'max_val': 500},
        'n_gen': {'type': int, 'min_val': 1, 'max_val': 500},
        'n_runs': {'type': int, 'min_val': 1}
    }

    try:
        with open(filename, 'r') as file:
            for line in file:
                line = line.strip()
                if not line or line.startswith('#'):  # Skip empty lines or comments
                    continue
                parts = line.split('=')
                if len(parts) != 2:
                    raise ValueError(f"Invalid line format: '{line}'. Expected 'variable=value'.")
                var_name, value = parts[0].strip(), parts[1].strip()
                if var_name not in required_vars:
                    raise ValueError(f"Unknown variable: '{var_name}'.")
                
                # Convert value to appropriate type
                try:
                    if required_vars[var_name]['type'] == int:
                        inputs[var_name] = int(value)
                    else:
                        inputs[var_name] = float(value)
                except ValueError:
                    raise ValueError(f"Invalid value for {var_name}: '{value}'. Must be a {required_vars[var_name]['type'].__name__}.")

        # Check for missing variables
        missing_vars = set(required_vars.keys()) - set(inputs.keys())
        if missing_vars:
            raise ValueError(f"Missing required variables: {missing_vars}")

        # Validate inputs
        for var_name, config in required_vars.items():
            value = inputs[var_name]
            min_val = config.get('min_val')
            max_val = config.get('max_val')

            # Special validation for 'i' (depends on n)
            if var_name == 'i':
                if value < 1 or value > inputs['n']:
                    raise ValueError(f"'i' must be an integer between 1 and {inputs['n']}, got {value}.")
            
            # Special validation for 'v_max' (depends on v_min)
            elif var_name == 'v_max':
                if value < inputs['v_min']:
                    raise ValueError(f"'v_max' must be >= {inputs['v_min']}, got {value}.")
            
            # General validation
            elif min_val is not None and value < min_val:
                raise ValueError(f"'{var_name}' must be >= {min_val}, got {value}.")
            elif max_val is not None and value > max_val:
                raise ValueError(f"'{var_name}' must be <= {max_val}, got {value}.")

        return inputs

    except FileNotFoundError:
        raise FileNotFoundError(f"Input file '{filename}' not found.")
    except Exception as e:
        raise ValueError(f"Error reading inputs: {str(e)}")

# Snap to nearest valid line number
def snap_to_line(j, n_total_lines):
    return max(1, min(n_total_lines, round(j)))

# Calculate LT_ij and related parameters for a pair (i, j)
def calculate_path_params(v, B, i_curr, j_curr, n_total_lines, L):
    beta = math.radians(22)

    if v <= 0:
        return j_curr, float('inf'), float('inf'), float('inf'), float('inf'), 0, [0,0], [0,0]

    tan_beta = math.tan(beta)
    if tan_beta == 0:
        return j_curr, float('inf'), float('inf'), float('inf'), float('inf'), 0, [0,0], [0,0]

    R = (v ** 2) / (9.8 * tan_beta)
    
    j_curr = snap_to_line(j_curr, n_total_lines)

    LV_min = 30 * v
    pi = [(i_curr-1) * B, L]

    delta_x = (j_curr-1) * B - (i_curr-1) * B
    delta_y = LV_min

    L_a = math.sqrt(delta_x**2 + delta_y**2)
    
    S_i_max = math.radians(3)

    LT_ij = float('inf')
    LV = LV_min
    s_i = 0
    pj = [(j_curr-1) * B, L + LV_min]

    try:
        X_val_from_R = (2 * R) % B if B != 0 else float('inf')

        if X_val_from_R <= B / 2 and (2 * R - abs(i_curr-j_curr) * B) > 0:
            sqrt_arg_lr = 8 * v * (2 * R - abs(i_curr-j_curr) * B)
            if sqrt_arg_lr < 0:
                L_r = float('inf')
            else:
                L_r = math.sqrt(sqrt_arg_lr) / S_i_max if S_i_max != 0 else float('inf')
            
            divisor_s_i = (2 * R - abs(i_curr-j_curr) * B)
            if divisor_s_i == 0:
                asin_arg_s_i = 0
            else:
                asin_arg_s_i = L_a / divisor_s_i

            if abs(asin_arg_s_i) > 1:
                s_i = S_i_max
            else:
                s_i = math.asin(asin_arg_s_i)

            if L_a < L_r:
                s_i = S_i_max
                arc_term_la = (2 * R - abs(i_curr-j_curr) * B)
                if arc_term_la < 0:
                    LT_ij = float('inf')
                else:
                    L_a = arc_term_la * math.sin(s_i)
                    LV = L_a * math.cos(s_i)
                    pj = [(j_curr-1) * B, L + LV]
                    LT_ij = L_a + LV + math.pi * R
            else:
                LV = LV_min
                LT_ij = L_a + LV + math.pi * R

        else:
            LT_ij = math.pi * R + B * abs(i_curr-j_curr) - 2 * R + 2 * LV_min
            LV = LV_min
            pj = [(j_curr-1) * B, L + LV]
            s_i = 0

    except ValueError:
        LT_ij = float('inf')
        LV = LV_min
        s_i = 0
    except ZeroDivisionError:
        LT_ij = float('inf')
        LV = LV_min
        s_i = 0
        
    return j_curr, R, LT_ij, L_a, LV, s_i, pi, pj


try:
    inputs = read_inputs_from_file(input_filename)
    n = inputs['n']
    B = inputs['B']
    i = inputs['i']
    L = inputs['L']
    v_min = inputs['v_min']
    v_max = inputs['v_max']
    pop_size = inputs['pop_size']
    n_gen = inputs['n_gen']
    n_runs = inputs['n_runs']
    
    print("Successfully read inputs from file:")
    for key, value in inputs.items():
        print(f"  {key} = {value}")
    
except Exception as e:
    print(f"Error: {e}")
    print("\nPlease create an 'inputs.txt' file with the following format:")
    print("n=5")
    print("B=1.0")
    print("i=1")
    print("L=100.0")
    print("v_min=5.0")
    print("v_max=15.0")
    print("pop_size=50")
    print("n_gen=100")
    print("n_runs=3")
    exit(1)

n_var = 1 + (n - 1)
xl = np.array([v_min] + [1] * (n - 1))
xu = np.array([v_max] + [n] * (n - 1))

# ========== SAMPLING ==========
class CustomSampling(Sampling):
    def _do(self, problem, n_samples, **kwargs):
        X = np.zeros((n_samples, problem.n_var))
        
        class VProblem(Problem):
            def __init__(self, xl_val, xu_val):
                super().__init__(n_var=1, xl=np.array([xl_val]), xu=np.array([xu_val]))
        
        v_problem_dummy = VProblem(problem.xl[0], problem.xu[0])
        
        lhs_sampler_v = LHS()
        X_v = lhs_sampler_v._do(v_problem_dummy, n_samples)
        X[:, 0] = X_v.flatten()

        if problem.n_var > 1:
            possible_lines_for_sequence = [line for line in range(1, n + 1) if line != i]
            for k in range(n_samples):
                np.random.shuffle(possible_lines_for_sequence)
                X[k, 1:] = possible_lines_for_sequence[:problem.n_var - 1]
        return X

# ========== SURVIVAL ==========
class FitnessSurvival(Survival):
    def __init__(self):
        super().__init__(filter_infeasible=False)

    def _do(self, problem, pop, n_survive=None, **kwargs):
        F, cv = pop.get("F", "cv")
        
        if F is None or len(F) == 0:
            return pop[:0]
            
        if F.ndim == 1:
            F = F[:, None]

        S = np.lexsort([F[:, 0], cv])
        pop.set("rank", np.argsort(S))

        if n_survive is None:
            n_survive = len(pop)
        
        return pop[S[:n_survive]]

def comp_by_cv_and_fitness(pop, P, **kwargs):
    S = np.full(P.shape[0], np.nan)
    for k in range(P.shape[0]):
        a, b = P[k, 0], P[k, 1]
        
        if pop[a].CV > 0.0 and pop[b].CV == 0.0:
            S[k] = 1
        elif pop[a].CV == 0.0 and pop[b].CV > 0.0:
            S[k] = 0
        elif pop[a].CV > 0.0 and pop[b].CV > 0.0:
            S[k] = compare(a, pop[a].CV, b, pop[b].CV, method='smaller_is_better', return_random_if_equal=True)
        else:
            S[k] = compare(a, pop[a].F, b, pop[b].F, method='smaller_is_better', return_random_if_equal=True)
    return S[:, None].astype(int)

# ========== MUTATION ==========
class CustomMutation(Mutation):
    def _do(self, problem, X, **kwargs):
        X_mutated = np.copy(X)
        mutation_rate_v = 0.1
        mutation_rate_lines = 0.2

        for k in range(len(X_mutated)):
            if np.random.rand() < mutation_rate_v:
                delta = (problem.xu[0] - problem.xl[0]) * 0.1
                mutated_v = X_mutated[k, 0] + np.random.uniform(-delta, delta)
                X_mutated[k, 0] = np.clip(mutated_v, problem.xl[0], problem.xu[0])

            if problem.n_var > 1 and np.random.rand() < mutation_rate_lines:
                if (problem.n_var - 1) >= 2:
                    line_indices_to_mutate_relative = np.random.choice(range(problem.n_var - 1), size=2, replace=False)
                    idx1_abs = line_indices_to_mutate_relative[0] + 1
                    idx2_abs = line_indices_to_mutate_relative[1] + 1
                    X_mutated[k, idx1_abs], X_mutated[k, idx2_abs] = X_mutated[k, idx2_abs], X_mutated[k, idx1_abs]
        return X_mutated

# ========== PROBLEM DEFINITION ==========
class ChipFillingProblem(Problem):
    def __init__(self):
        super().__init__(n_var=n_var, n_obj=1, n_constr=1, xl=xl, xu=xu)

    def _evaluate(self, X, out, *args, **kwargs):
        F = []
        G = []

        for x_individual in X:
            v = x_individual[0]
            remaining_lines = list(set(range(1, n + 1)) - {i})
            gene_values_for_order = x_individual[1:]
            paired_lines_and_genes = sorted(
                zip(remaining_lines, gene_values_for_order),
                key=lambda x: x[1]
            )
            ordered_remaining_lines = [item[0] for item in paired_lines_and_genes]
            sequence = [i] + ordered_remaining_lines

            if len(sequence) != n:
                constraint_violation = 10
            else:
                constraint_violation = 0

            total_LT = 0
            jump_constraint_violation = 0
            R_used = None

            for k_seq in range(len(sequence) - 1):
                curr_i_path, curr_j_path = sequence[k_seq], sequence[k_seq + 1]
                j_snap, R, LT_ij, L_a, LV, s_i, pi, pj = calculate_path_params(
                    v, B, curr_i_path, curr_j_path, n, L
                )

                total_LT += LT_ij

                x = (2 * R) % B
                jump_max = (2 * R - x) / B + i

                jump = abs(curr_i_path - curr_j_path)
                if jump > jump_max:
                    jump_constraint_violation += jump
                elif (jump_max - jump) <=2:
                    pass
                elif jump <= jump_max:
                    jump_constraint_violation += (n - jump)
                
                else:
                    jump_constraint_violation += 2 * n

                R_used = R

            if R_used is not None:
                first_val = sequence[0]
                last_val = sequence[-1]
                x_end = (2 * R_used) % B
                jump_max_end = (2 * R_used - x_end) / B + i 
                end_jump = abs(last_val - first_val)

                if end_jump > jump_max_end:
                    constraint_violation += 3 * n

            t = (L + total_LT) / v if v > 0 and total_LT != float('inf') else float('inf')

            F.append(t)
            G.append([constraint_violation + jump_constraint_violation])

        out["F"] = np.array(F)[:, None]
        out["G"] = np.array(G)

# ========== OPTIMIZATION ==========
termination = get_termination("n_gen", n_gen)
crossover = SBX(prob=0.8, eta=15)
selection = TournamentSelection(func_comp=comp_by_cv_and_fitness, pressure=2)

all_fitness_over_time = []
best_solutions = []

for run in range(n_runs):
    print(f"\n==== Run {run+1} ====")
    problem_instance = ChipFillingProblem()
    
    algorithm = GA(
        pop_size=pop_size,
        sampling=CustomSampling(),
        crossover=crossover,
        mutation=CustomMutation(),
        survival=FitnessSurvival(),
        selection=selection,
        eliminate_duplicates=True
    )
    res = minimize(
        problem_instance,
        algorithm,
        termination,
        seed=run,
        save_history=True,
        verbose=False
    )
    
    print(f"\nGeneration 0 Population for Run {run+1}:")
    if res.history and len(res.history) > 0:
        gen0_pop = res.history[0].pop
        X_gen0 = gen0_pop.get("X")
        F_gen0 = gen0_pop.get("F")
        CV_gen0 = gen0_pop.get("CV")

        if X_gen0 is not None and len(X_gen0) > 0:
            X_gen0_rounded = np.round(X_gen0, round_decimals)
            F_gen0_2D = F_gen0 if F_gen0.ndim == 2 else F_gen0[:, None]

            for k in range(len(X_gen0_rounded)):
                v = float(X_gen0_rounded[k, 0])
                remaining_lines_gen0 = list(set(range(1, n + 1)) - {i})
                gene_values_for_order_gen0 = X_gen0[k, 1:]
                paired_lines_and_genes_gen0 = sorted(
                    zip(remaining_lines_gen0, gene_values_for_order_gen0),
                    key=lambda x: x[1]
                )
                ordered_remaining_lines_gen0 = [item[0] for item in paired_lines_and_genes_gen0]
                sequence_display = [i] + ordered_remaining_lines_gen0
                print(
                    f"Individual {k+1}: v = {v:.{round_decimals}f} m/s, sequence = {sequence_display}, "
                    f"t = {float(F_gen0_2D[k][0]):.{round_decimals}f} s, CV = {float(CV_gen0[k]):.{round_decimals}f}"
                )
        else:
            print("Generation 0 population is empty or invalid.")
    else:
        print("History is not available for Generation 0.")

    print(f"\nFinal Population for Run {run+1}:")
    X_final_pop = res.pop.get("X")
    F_final_pop = res.pop.get("F")
    CV_final_pop = res.pop.get("CV")

    if X_final_pop is not None and len(X_final_pop) > 0:
        X_rounded = np.round(X_final_pop, round_decimals)
        F_final_pop_2D = F_final_pop if F_final_pop.ndim == 2 else F_final_pop[:, None]

        sorted_indices = np.argsort(F_final_pop_2D[:, 0])

        for k in sorted_indices:
            v = float(X_rounded[k, 0])
            remaining_lines_final = list(set(range(1, n + 1)) - {i})
            gene_values_for_order_final = X_final_pop[k, 1:]
            paired_lines_and_genes_final = sorted(
                zip(remaining_lines_final, gene_values_for_order_final),
                key=lambda x: x[1]
            )
            ordered_remaining_lines_final = [item[0] for item in paired_lines_and_genes_final]
            sequence_display = [i] + ordered_remaining_lines_final

            # Calculate jump_max for each leg in the sequence
            jump_max_values = []
            for k_seq in range(len(sequence_display) - 1):
                curr_i_path, curr_j_path = sequence_display[k_seq], sequence_display[k_seq + 1]
                j_snap, R, LT_ij, L_a, LV, s_i, pi, pj = calculate_path_params(
                    v, B, curr_i_path, curr_j_path, n, L
                )
                x = (2 * R) % B if B != 0 else 0
                jump_max = (2 * R - x) / B + i if B != 0 else float('inf')
                jump_max_values.append(round(jump_max, round_decimals))

            print(
                f"Individual {k+1}: v = {v:.{round_decimals}f} m/s, sequence = {sequence_display}, "
                f"t = {float(F_final_pop_2D[k][0]):.{round_decimals}f} s, CV = {float(CV_final_pop[k]):.{round_decimals}f} "
            )
    
        best = np.argmin(F_final_pop_2D[:, 0])
        best_v = X_rounded[best, 0]
        
        remaining_lines_best = list(set(range(1, n + 1)) - {i})
        gene_values_for_order_best = X_final_pop[best, 1:]
        paired_lines_and_genes_best = sorted(zip(remaining_lines_best, gene_values_for_order_best), key=lambda x: x[1])
        best_sequence = [i] + [item[0] for item in paired_lines_and_genes_best]

        total_LT_for_best = 0
        path_details = []
        for idx in range(len(best_sequence)-1):
            curr_i_path, curr_j_path = best_sequence[idx], best_sequence[idx+1]
            j_snap, R, LT_ij, L_a, LV, s_i, pi, pj = calculate_path_params(best_v, B, curr_i_path, curr_j_path, n, L)
            total_LT_for_best += LT_ij
            path_details.append((curr_i_path, curr_j_path, R, LT_ij, L_a, LV, s_i, pi, pj))
            
        best_solutions.append({
            'run': run + 1,
            'v': best_v,
            'sequence': best_sequence,
            'total_time': F_final_pop_2D[best][0],
            'total_LT': total_LT_for_best,
            'path_details': path_details,
            'n_generations': len(res.history)
        })
        
    else:
        print("Final population is empty or invalid.")

    fitness_over_time = []
    for gen_history in res.history:
        if gen_history.opt is not None and gen_history.opt.get("F") is not None and len(gen_history.opt.get("F")) > 0:
            fitness_over_time.append(round(gen_history.opt.get("F")[0][0], round_decimals))
        else:
            fitness_over_time.append(fitness_over_time[-1] if fitness_over_time else float('inf'))
    all_fitness_over_time.append(fitness_over_time)

# Print best solutions with metrics
print("\n===== Best Solutions Across All Runs =====")
for sol in best_solutions:
    print(f"\nBest Solution for Run {sol['run']} (Minimal Time):")
    print(f"Speed: {sol['v']:.{round_decimals}f} m/s")
    print(f"Sequence: {sol['sequence']}")
    print(f"Total Time: {sol['total_time']:.{round_decimals}f} seconds")
    print("Path Details (From->To | R | LT_ij | L_a | LV | s_i | Start Point | End Point):")
    print(f"{'From':<6} {'To':<6} {'R (cu)':<10} {'LT_ij (cu)':<12} {'L_a (cu)':<10} {'LV (cu)':<10} {'s_i (rad)':<12} {'Start Point (cu)':<22} {'End Point (cu)':<22}")
    print("-" * 128)
    for curr_i_path, curr_j_path, R, LT_ij, L_a, LV, s_i, pi, pj in sol['path_details']:
        pi_str = f"[{pi[0]:.{round_decimals}f}, {pi[1]:.{round_decimals}f}]"
        pj_str = f"[{pj[0]:.{round_decimals}f}, {pj[1]:.{round_decimals}f}]"
        print(f"{curr_i_path:<6} {curr_j_path:<6} {R:.{round_decimals}f} {LT_ij:.{round_decimals}f} {L_a:.{round_decimals}f} {LV:.{round_decimals}f} {s_i:.{round_decimals}f} {pi_str:<22} {pj_str:<22}")
    print(f"\nTotal Path Length (sum LT_ij): {sol['total_LT']:.{round_decimals}f} chip units")
    print(f"Total Generations Run: {sol['n_generations']}")

# ========== PLOTTING ==========
plt.figure(figsize=(10, 5))
for run, fitness in enumerate(all_fitness_over_time):
    plt.plot(fitness, marker='o', linestyle='-', label=f'Run {run+1}')
plt.xlabel('Generation')
plt.ylabel('Best Total Time (seconds)')
plt.title('Convergence Plot of Chip Filling Genetic Algorithm with Jump Constraint')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()



