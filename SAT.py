from utils import *
import os
import numpy as np
import pickle
import random
import time

random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

# TODO:
#  - NB: w diverso da lr perchè propagato nella bacward invece di essere moltiplicato sui nodi
#  - w diviso da lr:
#    - adesso entrambi hanno w
#      - LTN si ferma quando supera w (l'alternativa e' aggiungere regolarizzazione per distanza da w... troppo complesso e non ha senso)
#    - lr solo per LTN (qualche tentativo)

# Fixed settings
n_trials = 30  # TODO: 10 trials??
n_steps = 100

# Variable settings
targets = [0.75, 0.9, 1.0]  # 0.5,  list of target truth values

lr_list = [0.1]  # , 0.01, 0.001]
regularization_lambda_list = [0.01, 0.0]  # [0.5, 0.1, 0.01, 0.0]

start_time = time.time()
results_lrl = []
results_ltn = []
for filename in os.listdir('uf20-91')[:10]:  # TODO: rimuovere [:2]

    with open(os.path.join('uf20-91', filename), 'r') as f:
        l = f.readlines()

    # Read knowledge
    clauses, n = parse_cnf(l)
    predicates = create_predicates(n)
    f = create_formula(predicates, clauses)
    print(f)

    for t in targets:
        t_tensor = torch.Tensor([t])
        print('Target: ' + str(t))
        generator = initialize_pre_activations(n, n_trials)

        # ========================================== LRL ==========================================

        # Generate initial random pre-activations
        z = next(generator)
        initial_truth_values = torch.sigmoid(z)

        # Define the model
        lrl = LRLModel(f, n_steps, t)

        # Optimization
        lrl_predictions = lrl(z)

        # Evaluation
        lrl_sat_f, lrl_norm_f = evaluate_solutions(f, lrl_predictions, initial_truth_values)
        lrl_sat_c, lrl_norm_c = evaluate_solutions(f, defuzzify_list(lrl_predictions), defuzzify(initial_truth_values))

        results_lrl.append({
            'formula': filename,
            'target': t,
            'sat_f': lrl_sat_f,
            'sat_c': lrl_sat_c,
            'norm_f': lrl_norm_f,
            'norm_c': lrl_norm_c
        })
        print('LRL: ' + str(torch.mean(f.satisfaction(lrl_predictions[-1])).tolist()))

        # ========================================== SGD ==========================================
        for reg_lambda in regularization_lambda_list:
            for lr in lr_list:
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
                ltn_sat_c, ltn_norm_c = evaluate_solutions(f, defuzzify_list(ltn_predictions), defuzzify(initial_truth_values))

                results_ltn.append({
                    'formula': filename,
                    'target': t,
                    'lr': lr,
                    'lambda': reg_lambda,
                    'sat_f': ltn_sat_f,
                    'sat_c': ltn_sat_c,
                    'norm_f': ltn_norm_f,
                    'norm_c': ltn_norm_c
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
#  - mean over all problems (also standard deviation??)
#  - dump mean results
#  - plots
#  DONE:
#  - run it twice and check if the results are exactly the same (check the seed)
#  - dump results_ltn and results_lrl
