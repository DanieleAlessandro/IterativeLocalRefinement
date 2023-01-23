import numpy as np
import pickle
import matplotlib.pyplot as plt
from settings import *
import os

colors_lrl = {
    'mean': {
        1.0: 'red',
        0.1: 'black'
    },
    'max': {
        1.0: 'blue',
        0.1: 'green'
    }
}

colors_ltn = {
    'sgd': {
        0.01: 'purple',
        0.1: 'grey',
        0: 'orange'
    },
    'adam': {
        0: 'red',
        0.01: 'purple',
        0.1: 'grey',
    }
}


def find_color_lrl(method, schedule):
    return colors_lrl[method][schedule]


def find_color_ltn(method, l):
    return colors_ltn[method][l]


def find_label_lrl(method, schedule):
    return 'ILR ({})'.format(schedule)


def find_label_ltn(method, l):
    return '{} ({})'.format(method.upper(), l)


def generate_plots(key, title, axs, plot_row, plot_col, aggregate='mean', target=1.0):
    x_axis = list(range(n_steps + 1))

    for method in methods:
        for lrl_schedule in lrl_schedules:
            l_results = []
            for p in results_lrl:
                if p['target'] == target and p['method'] == method and p['schedule'] == lrl_schedule:
                    # There used to be a mean here??
                    fuzzy_sat_lrl = p[key]
                    if len(fuzzy_sat_lrl) < n_steps + 1:
                        to_fill = n_steps + 1 - len(fuzzy_sat_lrl)
                        fuzzy_sat_lrl = np.concatenate([fuzzy_sat_lrl, np.array([fuzzy_sat_lrl[-1]] * to_fill)])
                    l_results.append(fuzzy_sat_lrl)
            if aggregate == 'mean':
                sat_mean = np.stack(l_results).mean(axis=0)
            if aggregate == 'mse':
                sat_mean = np.sqrt(np.square(target - np.stack(l_results)).mean(axis=0))

            axs[plot_row, plot_col].plot(x_axis,
                                         sat_mean,
                                         color=find_color_lrl(method, lrl_schedule),
                                         label=find_label_lrl(method, lrl_schedule))

    for reg_l in regularization_lambda_list:
        for sgd_method in sgd_methods:
            fuzzy_sat_ltn = np.array([p[key]
                                      for p in results_ltn
                                      if p['target'] == target and p['lambda'] == reg_l and p['sgd_method'] == sgd_method])

            if aggregate == 'mean':
                sat_mean = fuzzy_sat_ltn.mean(axis=0)
            if aggregate == 'mse':
                sat_mean = np.sqrt(np.square(target - fuzzy_sat_ltn).mean(axis=0))
            axs[plot_row, plot_col].plot(x_axis,
                                         sat_mean,
                                         ls='--',
                                         color=find_color_ltn(sgd_method, reg_l),
                                         label=find_label_ltn(sgd_method, reg_l))

    if title:
        axs[plot_row, plot_col].set_title(title, fontweight='bold')


results_lrl = None
results_ltn = None


def create_figures(axes, col, tnorm, target):
    generate_plots('sat_f', tnorm.capitalize(), axs=axes, plot_row=0, plot_col=col, aggregate='mean', target=target)
    generate_plots('norm1_f', None, axs=axes, plot_row=1, plot_col=col, target=target)


for amt_rulez in [20, 91]:
    plt.rcParams["figure.figsize"] = (11,5)
    plt.subplots_adjust(right=0.7)
    fig, axes = plt.subplots(2, 3, sharex='col', sharey='row')
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.85,
                        wspace=0.1,
                        hspace=0.2)

    for ax in axes.flat:
        ax.set(xlabel='Iteration')
    axes[0,0].set_ylabel('Satisfaction', fontweight='bold')
    axes[0,1].set_ylabel('Satisfaction', fontweight='bold')
    axes[1,0].set_ylabel('L1 norm', fontweight='bold')
    axes[1,1].set_ylabel('L1 norm', fontweight='bold')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axes.flat:
        ax.label_outer()

    for col, tnorm in enumerate(tnorms_plot):
        if not os.path.exists(f'results/{tnorm}/lrl_{amt_rulez}_rules'):
            continue
        if not os.path.exists(f'plots/{tnorm}'):
            os.makedirs(f'plots/{tnorm}')
        base_path = f'plots/{tnorm}/{amt_rulez}_rules'
        if not os.path.exists(base_path):
            os.makedirs(base_path)

        with open(f'results/{tnorm}/lrl_{amt_rulez}_rules', 'rb') as f:
            results_lrl = pickle.load(f)

        with open(f'results/{tnorm}/ltn_{amt_rulez}_rules', 'rb') as f:
            results_ltn = pickle.load(f)

        print(f'Creating figures for {tnorm} with {amt_rulez}', flush=True)
        create_figures(axes, col, tnorm, target)
        create_figures(axes, col, tnorm, target)

    handles, labels = axes[0,0].get_legend_handles_labels()
    lg = fig.legend(handles[:5], labels[:5], loc='upper center', ncol=5, prop={'size': 12}, borderaxespad=0.2)
    # lg = fig.legend(handles[:5], labels[:5], loc='upper center', ncol=5, prop={'size': 18}, borderaxespad=0.2)

    fig.savefig(f'plots_final/results_{amt_rulez}_{target}_final.png',
                bbox_extra_artists=(lg,),
                bbox_inches='tight')
    plt.close()


