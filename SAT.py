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

def evaluate(formula, predictions, initial_truth_values, time, hyperparams: dict):
    # Evaluation
    sat_f, norm1_f, norm2_f = evaluate_solutions(formula, predictions, initial_truth_values)
    sat_c, norm_c, n_clauses = evaluate_solutions(formula, defuzzify_list(predictions),
                                                              defuzzify(initial_truth_values), fuzzy=False)

    return (hyperparams | {  # _f: fuzzy truth values used, _c: classic logic (defuzzified)
        'sat_f': sat_f,
        'sat_c': sat_c,
        'norm1_f': norm1_f,
        'norm2_f': norm2_f,
        'norm_c': norm_c,
        'n_clauses_satisfied_c': n_clauses,
        'time': time
    })

list_of_files = os.listdir('uf20-91')[:n_formulas]

# device = torch.cuda.device(torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
device = torch.device("cuda") if use_cuda and torch.cuda.is_available() else torch.device("cpu")
print(device)

if verbose:
    print(list_of_files)

if not os.path.exists('results'):
    os.mkdir('results')

for amt_rulez in amt_rules:
    print("==========================")
    print("AMT RULES:", amt_rulez)
    print("===========================")
    results_lrl = []
    results_ltn = []
    for problem_number, filename in enumerate(list_of_files):
        problem_number += 1
        print('Problem n. ' + str(problem_number) + '/' + str(n_formulas), flush=True)

        with open(os.path.join('uf20-91', filename), 'r') as f:
            l = f.readlines()

        # Read knowledge
        clauses, n = parse_cnf(l, amt_rulez)

        time_problem_start = time.time()

        f = SATFormula(clauses, device)
        # f.to(device)

        for w in targets:
            if verbose:
                print('Target: ' + str(w))
            w_tensor = torch.tensor([w], device=device)

            # Generate initial random pre-activations
            # The initialize_pre_activations first returns a not learnable tensor (used by LRL), from
            # the second next it returns the same exact value as a new learnable parameter (used by LTN)
            generator = initialize_pre_activations(n, n_initial_vectors, device)

            initial_truth_values = torch.sigmoid(next(generator))

            base_dict = {
                'amt_rules': amt_rulez,
                'target': w,
                'problem_number': problem_number,
                'formula': filename,
                'tnorm': tnorm,
                'sgd_norm': sgd_norm,
            }

            for method in methods:
                f.is_sgd = False
                f.tnorm = tnorm_constructor(tnorm, method)
                for lrl_schedule in lrl_schedules:
                    start = time.time()

                    # ========================================== LRL ==========================================

                    # Define the model
                    lrl = LRL(f, n_steps, schedule=lrl_schedule)

                    # Optimization
                    lrl_predictions = lrl(initial_truth_values, w)

                    # For debugging purposes
                    # lrl = LRLModel(f_non_parallel, n_steps, t)
                    # lrl(z, method)

                    time_cost = time.time() - start
                    if verbose:
                        print(f'LRL@{lrl_schedule}: {torch.mean(f.satisfaction(lrl_predictions[-1])).tolist()}     Time: {time_cost}')
                    results_lrl.append(evaluate(f, lrl_predictions, initial_truth_values, time_cost, base_dict | {
                        'method': method,
                        'schedule': lrl_schedule,
                    }))

            f.is_sgd = True
            f.tnorm = tnorm_constructor(tnorm, 'mean')
            # ========================================== SGD ==========================================
            for sgd_method in sgd_methods:
                for reg_lambda in regularization_lambda_list:
                    start = time.time()

                    # Generate initial random pre-activations
                    z = next(generator)

                    # Define the model
                    ltn = LTNModel(f)

                    # LTN optimization
                    if sgd_method == 'sgd':
                        optimizer = torch.optim.SGD([z], lr=0.1)
                    elif sgd_method == 'adam':
                        optimizer = torch.optim.Adam([z], lr=0.1)
                    elif sgd_method == 'sgd_momentum':
                        optimizer = torch.optim.Adagrad([z], lr=0.1)

                    ltn_predictions = [torch.sigmoid(z)]

                    # TODO: This needs to be made generic
                    if tnorm == "product":
                        sgd_t = w_tensor.log()
                    else:
                        sgd_t = w_tensor

                    for i in range(n_steps):
                        optimizer.zero_grad()
                        sgd_value, _ = ltn(z)
                        s = torch.linalg.vector_norm(sgd_value - sgd_t, ord=2) + \
                            reg_lambda * torch.linalg.vector_norm(torch.sigmoid(z) - initial_truth_values, ord=sgd_norm)
                        s.backward()
                        optimizer.step()
                        ltn_predictions.append(torch.sigmoid(z))
                    time_cost = time.time() - start
                    if verbose:
                        print(f'LTN-{sgd_method}@{reg_lambda}: {torch.mean(f.satisfaction(ltn_predictions[-1])).tolist()}     Time: {time_cost}')

                    results_ltn.append(evaluate(f, ltn_predictions, initial_truth_values, time_cost, base_dict | {
                        'lambda': reg_lambda,
                        'sgd_method': sgd_method,
                    }))
        print('Problem took {} seconds'.format(time.time() - time_problem_start), flush=True)

    print('Saving results...', flush=True)
    end_time = time.time()

    # print('Time: ' + str(end_time - start_time) + 's')

    if not os.path.exists(f'results/{tnorm}'):
        os.mkdir(f'results/{tnorm}')
    with open(f'results/{tnorm}/lrl_{amt_rulez}_rules', 'wb') as f:
        pickle.dump(results_lrl, f)

    with open(f'results/{tnorm}/ltn_{amt_rulez}_rules', 'wb') as f:
        pickle.dump(results_ltn, f)

# TODO:
#  - standard deviation over all problems??

