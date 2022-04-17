from utils import *
import os
import numpy as np
import pickle

# Settings
n_trials = 1000
n_steps = 100
# TODO add learning rate and w (the same?)

sat_results = {}
for filename in os.listdir('uf20-91'):
    with open(os.path.join('uf20-91', filename), 'r') as f:
        l = f.readlines()

    # Read knowledge
    clauses, n = parse_cnf(l)
    predicates = create_predicates(n)
    f = create_formula(predicates, clauses)
    print(f)

    # Generate initial random pre-activations for both LRL and LTN
    z_lrl, z_ltn = initialize_pre_activations(n, n_trials)

    # Define the two models
    lrl = LRLModel(f, n_steps)
    ltn = LTNModel(f)

    # LRL optimization
    lrl_predictions = lrl(z_lrl)


    # LTN optimization
    optimizer = torch.optim.SGD([z_ltn], lr=1.)

    ltn_predictions = [torch.sigmoid(z_ltn)]
    for i in range(n_steps):
        s = - torch.sum(ltn(z_ltn))
        s.backward()
        optimizer.step()
        ltn_predictions.append(torch.sigmoid(z_ltn))

    sat_results[os.path.join('uf20-91', filename)] = [{'i':i, 'ltn':ltn_r, 'lrl':lrl_r}
                                                      for i, ltn_r, lrl_r
                                                      in zip(range(n_steps), ltn_predictions, lrl_predictions)]


with open('results', 'wb') as f:
    pickle.dump(sat_results, f)