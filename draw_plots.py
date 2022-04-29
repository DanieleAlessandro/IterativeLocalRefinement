import numpy as np
import pickle
import matplotlib.pyplot as plt
from settings import *
import os

with open(f'results/{tnorm_plot}/lrl_{amt_rules_plot}_rules', 'rb') as f:
    results_lrl = pickle.load(f)

with open(f'results/{tnorm_plot}/ltn_{amt_rules_plot}_rules', 'rb') as f:
    results_ltn = pickle.load(f)


def generate_plots(key, title, y_label, file_name, basepath, axs=None, plot_row=None, grid=True):
    if not grid:
        plt.rcParams["figure.figsize"] = (8.5, 5)
        plt.title(title)
        plt.xlabel('iteration')
        plt.ylabel(y_label)
        plt.subplots_adjust(right=0.7)

    x_axis = list(range(n_steps + 1))

    for plot_column, t in enumerate(targets):
        for method in methods:
            for lrl_schedule in lrl_schedules:
                l_results = []
                for p in results_lrl:
                    if p['target'] == t and p['method'] == method and p['schedule'] == lrl_schedule:
                        # There used to be a mean here??
                        fuzzy_sat_lrl = p[key]
                        if len(fuzzy_sat_lrl) < n_steps + 1:
                            to_fill = n_steps + 1 - len(fuzzy_sat_lrl)
                            fuzzy_sat_lrl = np.concatenate([fuzzy_sat_lrl, np.array([fuzzy_sat_lrl[-1]] * to_fill)])
                        l_results.append(fuzzy_sat_lrl)
                sat_mean = np.stack(l_results).mean(axis=0)
                label = f'LRL s={lrl_schedule} m={method}'
                if grid:
                    axs[plot_row, plot_column].plot(x_axis, sat_mean, label=label)
                else:
                    plt.plot(x_axis, sat_mean, label=label)


        for reg_l in regularization_lambda_list:
            fuzzy_sat_ltn = np.mean(np.array([p[key]
                                              for p in results_ltn
                                              if p['target'] == t and p['lambda'] == reg_l]), axis=0)

            if grid:
                axs[plot_row, plot_column].plot(x_axis, fuzzy_sat_ltn, ls='--', label='SGD reg=' + str(reg_l))
            else:
                plt.plot(x_axis, fuzzy_sat_ltn, ls='--', label='SGD reg=' + str(reg_l))

        if grid:
            axs[plot_row, plot_column].set_title(title)
        else:
            plt.legend(bbox_to_anchor=(1.45, 0.9), loc='upper right')
            plt.savefig(f'{basepath}/w_{t}/{file_name}.png', bbox_inches="tight")
            plt.close()


if not os.path.exists(f'plots/{tnorm_plot}'):
    os.makedirs(f'plots/{tnorm_plot}')
base_path = f'plots/{tnorm_plot}/{amt_rules_plot}_rules'
if not os.path.exists(base_path):
    os.makedirs(base_path)
for t in targets:
    if not os.path.exists(f'{base_path}/w_{t}'):
        os.mkdir(f'{base_path}/w_{t}')

def create_figures(grid):
    if grid:
        plt.rcParams["figure.figsize"] = (28, 20)
        plt.subplots_adjust(right=0.7)
        fig, axes = plt.subplots(6, 4, sharex='col', sharey='row')
    else:
        axes = None

    generate_plots('sat_f', 'Satisfaction (fuzzy logic)', 'fuzzy sat', 'sat_f', base_path, axs=axes, plot_row=0, grid=grid)
    generate_plots('norm1_f', 'L1 norm (fuzzy logic)', 'L1 norm', 'fuzzy_norm1', base_path, axs=axes, plot_row=1, grid=grid)
    generate_plots('norm2_f', 'L2 norm (fuzzy logic)', 'L2 norm', 'fuzzy_norm2', base_path, axs=axes, plot_row=2, grid=grid)
    generate_plots('sat_c', 'Satisfaction (classic logic)', 'sat', 'sat_c', base_path, axs=axes, plot_row=3, grid=grid)
    generate_plots('n_clauses_satisfied_c', 'Proportion of satisfied clauses (classic logic)', 'n sat', 'n_clauses', base_path, axs=axes, plot_row=4, grid=grid)
    generate_plots('norm_c', 'L1 norm (classic logic)', 'L1 norm', 'crisp_norm', base_path, axs=axes, plot_row=5, grid=grid)

    if grid:
        fig.savefig(f'plots/results_{tnorm_plot}_{amt_rules_plot}.png')
        plt.close()

create_figures(False)
create_figures(True)
# row_labels = ['a','c','x','x','x']
# column_labels = ['zaza', 'c','x','x']

# for row, column, ax in zip(row_labels, column_labels, axes.flat):
#     ax.set(xlabel=row, ylabel=column)

# for ax in axs.flat:
#     ax.label_outer()

# fig.legend(loc='lower center')

