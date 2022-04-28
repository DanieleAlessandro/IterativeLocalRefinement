from typing import List

from SAT_formula import SATFormula, SATLukasiewicz, SATProduct, tnorm_constructor
from utils import *
import os
import numpy as np
import pickle
import random
import time
from settings import *

seed = 0
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

# TODO:
#  - NB: w diverso da lr perchè propagato nella backward invece di essere moltiplicato sui nodi

def evaluate(formula, predictions, initial_truth_values, time, filename, hyperparams: dict):
    # Evaluation
    sat_f, norm1_f, norm2_f = evaluate_solutions(formula, predictions, initial_truth_values)
    sat_c, norm_c, n_clauses = evaluate_solutions(formula, defuzzify_list(predictions),
                                                              defuzzify(initial_truth_values), fuzzy=False)

    return (hyperparams | {  # _f: fuzzy truth values used, _c: classic logic (defuzzified)
        'formula': filename,
        'sat_f': sat_f,
        'sat_c': sat_c,
        'norm1_f': norm1_f,
        'norm2_f': norm2_f,
        'norm_c': norm_c,
        'n_clauses_satisfied_c': n_clauses,
        'time': time
    })

list_of_files = os.listdir('uf20-91')[:n_formulas]
print(list_of_files)

results_lrl = []
results_ltn = []

for problem_number, filename in enumerate(list_of_files):
    problem_number += 1
    print('Problem n. ' + str(problem_number) + '/' + str(n_formulas), flush=True)

    with open(os.path.join('uf20-91', filename), 'r') as f:
        l = f.readlines()

    # Read knowledge
    clauses, n = parse_cnf(l, amt_rules)

    for w in targets:
        print('Target: ' + str(w))
        w_tensor = torch.Tensor([w])

        # Generate initial random pre-activations
        # The initialize_pre_activations first returns a not learnable tensor (used by LRL), from
        # the second next it returns the same exact value as a new learnable parameter (used by LTN)
        generator = initialize_pre_activations(n, n_initial_vectors)

        initial_truth_values = torch.sigmoid(next(generator))

        for method in methods:
            f = SATFormula(clauses, False, tnorm_constructor(tnorm, method))
            for lrl_schedule in lrl_schedules:
                start = time.time()

                # ========================================== LRL ==========================================

                # Define the model
                lrl = LRL(f, n_steps, lrl_schedule)

                # Optimization
                lrl_predictions = lrl(initial_truth_values, w)

                # For debugging purposes
                # lrl = LRLModel(f_non_parallel, n_steps, t)
                # lrl(z, method)

                time_cost = time.time() - start
                print(f'LRL@{lrl_schedule}: {torch.mean(f.satisfaction(lrl_predictions[-1])).tolist()}     Time: {time_cost}')
                results_lrl.append(evaluate(f, lrl_predictions, initial_truth_values, time_cost, filename, {
                    'method': method,
                    'schedule': lrl_schedule,
                    'target': w
                }))

        f = SATFormula(clauses, True, tnorm_constructor(tnorm, "mean"))

        # ========================================== SGD ==========================================
        for reg_lambda in regularization_lambda_list:
            start = time.time()

            # Generate initial random pre-activations
            z = next(generator)

            # Define the model
            ltn = LTNModel(f)

            # LTN optimization
            optimizer = torch.optim.SGD([z], lr=0.1)

            ltn_predictions = [torch.sigmoid(z)]

            # TODO: This needs to be made generic
            if tnorm == "product":
                sgd_t = w_tensor.log()
            else:
                sgd_t = w_tensor

            for i in range(n_steps):
                sgd_value, _ = ltn(z)
                s = torch.linalg.vector_norm(sgd_value - sgd_t, ord=2) + \
                    reg_lambda * torch.linalg.vector_norm(torch.sigmoid(z) - initial_truth_values, ord=sgd_norm)
                s.backward()
                optimizer.step()
                ltn_predictions.append(torch.sigmoid(z))
            time_cost = time.time() - start
            print(f'LTN@{reg_lambda}: {torch.mean(f.satisfaction(ltn_predictions[-1])).tolist()}     Time: {time_cost}')

            results_ltn.append(evaluate(f, ltn_predictions, initial_truth_values, time_cost, filename, {
                'lambda': reg_lambda,
                'target': w
            }))

print('Saving results...', flush=True)
end_time = time.time()

# print('Time: ' + str(end_time - start_time) + 's')

with open('results_lrl', 'wb') as f:
    pickle.dump(results_lrl, f)

with open('results_ltn', 'wb') as f:
    pickle.dump(results_ltn, f)

# TODO:
#  - check all TODOs and debug
#  - standard deviation over all problems??
#  DONE:
#  - run it twice and check if the results are exactly the same (check the seed)
#  - dump results_ltn and results_lrl
#  - mean over all problems
#  - dump mean results
#  - plots
