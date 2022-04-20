from utils import *
import os
import numpy as np
import pickle
import random

random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

# Fixed settings
n_trials = 1000
n_steps = 50

# Variable settings
w_lr_list = [1.0, 0.5, 0.1, 0.01, 0.001]
regularization_lambda_list = [0.5, 0.1, 0.01, 0.0]


results_lrl = []
results_ltn = []
for filename in os.listdir('uf20-91')[:2]:  # TODO: rimuovere [:2]
    with open(os.path.join('uf20-91', filename), 'r') as f:
        l = f.readlines()

    # Read knowledge
    clauses, n = parse_cnf(l)
    predicates = create_predicates(n)
    f = create_formula(predicates, clauses)
    print(f)

    for w_lr in w_lr_list:
        generator = initialize_pre_activations(n, n_trials)

        # ========================================== LRL ==========================================

        # Generate initial random pre-activations
        z = next(generator)
        initial_truth_values = torch.sigmoid(z)

        # Define the model
        lrl = LRLModel(f, n_steps, w_lr)

        # Optimization
        print('LRL optimization...')
        lrl_predictions = lrl(z)

        # Evaluation
        print('LRL evaluation...')
        lrl_sat_f, lrl_norm_f = evaluate_solutions(f, lrl_predictions, initial_truth_values)
        lrl_sat_c, lrl_norm_c = evaluate_solutions(f, defuzzify_list(lrl_predictions), defuzzify(initial_truth_values))

        results_lrl.append({
            'formula': filename,
            'w': w_lr,
            'sat_f': lrl_sat_f,
            'sat_c': lrl_sat_c,
            'norm_f': lrl_norm_f,
            'norm_c': lrl_norm_c
        })

        # ========================================== SGD ==========================================
        for reg_lambda in regularization_lambda_list:
            # Generate initial random pre-activations
            z = next(generator)

            # Define the model
            ltn = LTNModel(f)

            # LTN optimization
            optimizer = torch.optim.SGD([z], lr=w_lr)

            ltn_predictions = [torch.sigmoid(z)]
            print('LTN optimization...')
            for i in range(n_steps):
                s = - torch.sum(ltn(z)) + \
                    reg_lambda * torch.linalg.vector_norm(torch.sigmoid(z) - initial_truth_values, ord=1)
                s.backward()
                optimizer.step()
                ltn_predictions.append(torch.sigmoid(z))

            # Evaluation
            print('LTN evaluation...')
            ltn_sat_f, ltn_norm_f = evaluate_solutions(f, ltn_predictions, initial_truth_values)
            ltn_sat_c, ltn_norm_c = evaluate_solutions(f, defuzzify_list(ltn_predictions), defuzzify(initial_truth_values))

            results_ltn.append({
                'formula': filename,
                'lr': w_lr,
                'lambda': reg_lambda,
                'sat_f': ltn_sat_f,
                'sat_c': ltn_sat_c,
                'norm_f': ltn_norm_f,
                'norm_c': ltn_norm_c
            })

with open('results_lrl2', 'wb') as f:
    pickle.dump(results_lrl, f)

with open('results_ltn2', 'wb') as f:
    pickle.dump(results_ltn, f)

# TODO:
#  - check all TODOs and debug
#  - mean over all problems (also standard deviation??)
#  - dump mean results
#  - plots
#  DONE:
#  - run it twice and check if the results are exactly the same (check the seed)
#  - dump results_ltn and results_lrl
