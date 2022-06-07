import pickle
from utils import *
import matplotlib.pyplot as plt


# Settings
n_trials = 1000
n_steps = 100


def defuzzify(tensor):
    return (tensor > 0.5).float()


with open('results', 'rb') as f:
    sat_results = pickle.load(f)

ltn_results = [None] * n_steps
lrl_results = [None] * n_steps
for formula_file, results in sat_results.items():
    print(formula_file)
    with open(formula_file, 'r') as f:
        l = f.readlines()

    # Read knowledge
    clauses, n = parse_cnf(l)
    predicates = create_predicates(n)
    f = create_formula(predicates, clauses)

    for r in results:
        if ltn_results[r['i']] is None:
            ltn_results[r['i']] = []
        if lrl_results[r['i']] is None:
            lrl_results[r['i']] = []

        ltn_results[r['i']].append(torch.mean(f.satisfaction(defuzzify(r['ltn']))).tolist())
        lrl_results[r['i']].append(torch.mean(f.satisfaction(defuzzify(r['lrl']))).tolist())

ltn_average_fuzzy_sat = [np.mean(r) for r in ltn_results]
lrl_average_sat = [np.mean(r) for r in lrl_results]

with open('results_ltn_classic', 'wb') as f:
    pickle.dump(ltn_average_fuzzy_sat, f)

with open('results_lrl_classic', 'wb') as f:
    pickle.dump(lrl_average_sat, f)

x_axis = list(range(n_steps))
plt.plot(x_axis, ltn_average_fuzzy_sat, 'b', x_axis, lrl_average_sat, 'k')
plt.xlabel('iteration')
plt.ylabel('sat')
plt.show()
