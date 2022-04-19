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

        ltn_results[r['i']].append(torch.mean(f.satisfaction(r['ltn'])).tolist())
        lrl_results[r['i']].append(torch.mean(f.satisfaction(r['lrl'])).tolist())

ltn_average_fuzzy_sat = [np.mean(r) for r in ltn_results]
lrl_average_fuzzy_sat = [np.mean(r) for r in lrl_results]

with open('results_ltn', 'wb') as f:
    pickle.dump(ltn_average_fuzzy_sat, f)

with open('results_lrl', 'wb') as f:
    pickle.dump(lrl_average_fuzzy_sat, f)

x_axis = list(range(n_steps))
plt.plot(x_axis, ltn_average_fuzzy_sat, 'b', x_axis, lrl_average_fuzzy_sat, 'k')
plt.xlabel('iteration')
plt.ylabel('fuzzy sat')
plt.show()

#     for i in range(30):
#         initial_value = f.forward()
#         # f.print_table()
#
#         if i >= 0:
#             delta = (1 - initial_value)
#         else:
#             delta = (1 - initial_value) * 0.95 #random.uniform(0,1)
#         print('i=' + str(i) +
#               ' -> mean:   ' + str(torch.mean(f.sat_sub_formulas()).tolist()) +
#               '     max:   ' + str(torch.max(f.sat_sub_formulas()).tolist()) +
#               '    ones:   ' + str(torch.mean(f.count_ones()).tolist()) +
#               '   zeros:   ' + str(torch.mean(f.count_zeros()).tolist()) +
#               '     sum:   ' + str((torch.mean(f.count_zeros()) + torch.mean(f.count_ones())).tolist()) +
#               '     all:   ' + str(torch.mean(f.count_all()).tolist()) +
#               '     sat:   ' + str(torch.mean(f.count_sat()).tolist()))
#         if sat_results[i] is None:
#             sat_results[i] = [torch.mean(f.count_sat()).tolist()]
#         else:
#             sat_results[i].append(torch.mean(f.count_sat()).tolist())
#
#         if i < 0:
#             f.backward(delta, randomized=i % 3 == 0)
#         else:
#             f.backward(delta) #, randomized=i % 10 == 0)
#
#         for predicate in predicates:
#             if i <= 0:
#                 predicate.aggregate_deltas('most_clauses')
#             else:
#                 predicate.aggregate_deltas('max')
#
# # results = []
# for i, formula in enumerate(sat_results):
#     print(str(i) + ':   ' + str(np.mean(formula)))
#     # results.append(np.mean(formula))
