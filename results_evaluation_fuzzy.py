import pickle
from utils import *
import matplotlib.pyplot as plt

# TODO: aggiungere seed

# Settings
n_trials = 1000
n_steps = 100
# TODO move settings in a different file

with open('results', 'rb') as f:
    sat_results = pickle.load(f)

ltn_results = [None] * n_steps
lrl_results = [None] * n_steps
ltn_norms = [None] * n_steps
lrl_norms = [None] * n_steps
for formula_file, results in sat_results.items():
    print(formula_file)
    with open(formula_file, 'r') as f:
        l = f.readlines()

    # Read knowledge
    clauses, n = parse_cnf(l)
    predicates = create_predicates(n)
    f = create_formula(predicates, clauses)

    initial_truth_values = results[0]['ltn']

    for r in results:
        if ltn_results[r['i']] is None:
            ltn_results[r['i']] = []
        if lrl_results[r['i']] is None:
            lrl_results[r['i']] = []
        if ltn_norms[r['i']] is None:
            ltn_norms[r['i']] = []
        if lrl_norms[r['i']] is None:
            lrl_norms[r['i']] = []

        ltn_results[r['i']].append(torch.mean(f.satisfaction(r['ltn'])).tolist())
        lrl_results[r['i']].append(torch.mean(f.satisfaction(r['lrl'])).tolist())

        ltn_norms[r['i']].append(torch.linalg.vector_norm(r['ltn'] - initial_truth_values, ord=1).tolist())
        lrl_norms[r['i']].append(torch.linalg.vector_norm(r['lrl'] - initial_truth_values, ord=1).tolist())


with open('results_ltn_', 'wb') as f:
    pickle.dump(ltn_results, f)

with open('norms_ltn_', 'wb') as f:
    pickle.dump(ltn_norms, f)

with open('results_lrl_', 'wb') as f:
    pickle.dump(lrl_results, f)

with open('norms_lrl_', 'wb') as f:
    pickle.dump(lrl_norms, f)


ltn_average_fuzzy_sat = [np.mean(r) for r in ltn_results]
lrl_average_fuzzy_sat = [np.mean(r) for r in lrl_results]
ltn_average_norm = [np.mean(r) for r in ltn_norms]
lrl_average_norm = [np.mean(r) for r in lrl_norms]

with open('results_ltn', 'wb') as f:
    pickle.dump(ltn_average_fuzzy_sat, f)

with open('norms_ltn', 'wb') as f:
    pickle.dump(ltn_average_norm, f)

with open('results_lrl', 'wb') as f:
    pickle.dump(lrl_average_fuzzy_sat, f)

with open('norms_lrl', 'wb') as f:
    pickle.dump(lrl_average_norm, f)

x_axis = list(range(n_steps))
plt.plot(x_axis, ltn_average_fuzzy_sat, 'r', label='SGD')
plt.plot(x_axis, lrl_average_fuzzy_sat, 'k', label='LRL')

plt.suptitle('Fuzzy sat during optimization')
plt.title('w=lr=1.0, lambda=0.0')
plt.xlabel('iteration')
plt.ylabel('fuzzy sat')
plt.legend(loc='upper right')

plt.savefig('mean_fuzzy_sat.png')
plt.close()


plt.plot(x_axis, ltn_average_norm, 'r', label='SGD')
plt.plot(x_axis, lrl_average_norm, 'k', label='LRL')

plt.suptitle('L1 norm during optimization')
plt.title('w=lr=1.0, lambda=0.0')
plt.xlabel('iteration')
plt.ylabel('L1 norm')
plt.legend(loc='upper left')

plt.savefig('mean_fuzzy_L1.png')
plt.close()


plt.scatter(ltn_average_fuzzy_sat, ltn_average_norm, color='r', label='SGD')
plt.scatter(lrl_average_fuzzy_sat, lrl_average_norm, color='k', label='LRL')

plt.suptitle('Fuzzy sat vs L1 Norm')
plt.title('w=lr=1.0, lambda=0.0')
plt.xlabel('fuzzy sat')
plt.ylabel('L1 norm')
plt.legend(loc='upper right')

plt.savefig('scatter_fuzzy.png')
plt.close()


