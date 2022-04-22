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

list_of_files = random.sample(os.listdir('uf20-91'), n_formulas)
print(list_of_files)

start_time = time.time()
results_lrl = []
results_ltn = []
for filename in list_of_files:

    with open(os.path.join('uf20-91', filename), 'r') as f:
        l = f.readlines()

    # Read knowledge
    clauses, n = parse_cnf(l)
    predicates = create_predicates(n)
    f = create_formula(predicates, clauses)
    print(f)

    # Generate initial random pre-activations
    # The initialize_pre_activations first returns a not learnable tensor (used by LRL), from
    # the second next it returns the same exact value as a new learnable parameter (used by LTN)
    generator = initialize_pre_activations(n, n_initial_vectors)
    z = next(generator)
    initial_truth_values = torch.sigmoid(z)

    for method in methods:
        for t in targets:
            t_tensor = torch.Tensor([t])
            print('Target: ' + str(t))

            # ========================================== LRL ==========================================

            # Define the model
            lrl = LRLModel(f, n_steps, t)

            # Optimization
            lrl_predictions = lrl(z, method)

            # Evaluation
            lrl_sat_f, lrl_norm_f = evaluate_solutions(f, lrl_predictions, initial_truth_values)
            lrl_sat_c, lrl_norm_c, lrl_n_clauses = evaluate_solutions(f, defuzzify_list(lrl_predictions), defuzzify(initial_truth_values), fuzzy=False)

            results_lrl.append({  #_f: fuzzy truth values used, _c: classic logic (defuzzified)
                'formula': filename,
                'target': t,
                'method':method,
                'sat_f': lrl_sat_f,
                'sat_c': lrl_sat_c,
                'norm_f': lrl_norm_f,
                'norm_c': lrl_norm_c,
                'n_clauses_satisfied_c':lrl_n_clauses
            })
            print('LRL: ' + str(torch.mean(f.satisfaction(lrl_predictions[-1])).tolist()))

    # ========================================== SGD ==========================================
    for reg_lambda in regularization_lambda_list:
        for lr in lr_list:
            print('Learning rate: ' + str(lr))

            # Generate initial random pre-activations
            z = next(generator)

            # Define the model
            ltn = LTNModel(f)

            # LTN optimization
            optimizer = torch.optim.SGD([z], lr=lr)  # TODO

            ltn_predictions = [torch.sigmoid(z)]
            for i in range(n_steps):
                s = - torch.sum(torch.minimum(ltn(z), t_tensor)) + \
                    reg_lambda * torch.linalg.vector_norm(torch.sigmoid(z) - initial_truth_values, ord=1)
                s.backward()
                optimizer.step()
                ltn_predictions.append(torch.sigmoid(z))

            # Evaluation
            ltn_sat_f, ltn_norm_f = evaluate_solutions(f, ltn_predictions, initial_truth_values)
            ltn_sat_c, ltn_norm_c, ltn_n_clauses = evaluate_solutions(f,
                                                                      defuzzify_list(ltn_predictions),
                                                                      defuzzify(initial_truth_values), fuzzy=False)

            results_ltn.append({
                'formula': filename,
                'lr': lr,
                'lambda': reg_lambda,
                'sat_f': ltn_sat_f,
                'sat_c': ltn_sat_c,
                'norm_f': ltn_norm_f,
                'norm_c': ltn_norm_c,
                'n_clauses_satisfied_c': ltn_n_clauses
            })
            print('LTN: ' + str(torch.mean(f.satisfaction(ltn_predictions[-1])).tolist()))

print('Saving results...')
end_time = time.time()

print('Time: ' + str(end_time - start_time) + 's')

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
