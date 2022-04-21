import numpy as np
import pickle
import matplotlib.pyplot as plt
from settings import *

with open('results_lrl', 'rb') as f:
    results_lrl = pickle.load(f)

with open('results_ltn', 'rb') as f:
    results_ltn = pickle.load(f)


def generate_plots(key, label, y_label, file_name):
    plt.rcParams["figure.figsize"] = (8.5, 5)
    plt.title(label)
    plt.xlabel('iteration')
    plt.ylabel(y_label)
    plt.subplots_adjust(right=0.7)

    x_axis = list(range(n_steps + 1))

    for method in methods:
        for t in targets:
            fuzzy_sat_lrl = np.mean(np.array([p[key]
                                              for p in results_lrl
                                              if p['target'] == t and p['method'] == method]), axis=0)

            if len(fuzzy_sat_lrl) < n_steps + 1:
                to_fill = n_steps + 1 - len(fuzzy_sat_lrl)
                fuzzy_sat_lrl = np.concatenate([fuzzy_sat_lrl, np.array([fuzzy_sat_lrl[-1]] * to_fill)])

            plt.plot(x_axis, fuzzy_sat_lrl, label='LRL w=' + str(t) + ', agg=' + method)

    for reg_l in regularization_lambda_list:
        for lr in lr_list:
            fuzzy_sat_ltn = np.mean(np.array([p[key]
                                              for p in results_ltn
                                              if p['lr'] == lr and p['lambda'] == reg_l]), axis=0)

            plt.plot(x_axis, fuzzy_sat_ltn, ls='--', label='SGD lr=' + str(lr) + ' reg=' + str(reg_l))

    plt.legend(bbox_to_anchor=(1.45, 0.9), loc='upper right')

    plt.savefig('plots/' + file_name + '.png', bbox_inches="tight")
    plt.close()


generate_plots('sat_f', 'Fuzzy satisfaction over time', 'fuzzy sat', 'sat_f')
generate_plots('sat_c', 'Classic satisfaction over time', 'sat', 'sat_c')
generate_plots('norm_f', 'L1 norm over time (fuzzy setting)', 'L1 norm', 'fuzzy_norm')
generate_plots('norm_c', 'L1 norm over time (classic setting)', 'L1 norm', 'crisp_norm')
generate_plots('n_clauses_satisfied_c', 'Proportion of satisfied clauses over time', 'n sat', 'n_clauses')

