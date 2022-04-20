import numpy as np
import pickle
import matplotlib.pyplot as plt

# Fixed settings
n_trials = 1000
n_steps = 50

# Variable settings
w_lr_list = [1.0, 0.5, 0.1, 0.01, 0.001]
regularization_lambda_list = [0.5, 0.1, 0.01, 0.0]

with open('results_lrl', 'rb') as f:
    results_lrl = pickle.load(f)

with open('results_ltn', 'rb') as f:
    results_ltn = pickle.load(f)


def generate_plots(key, label, y_label, file_name):
    for w in w_lr_list:
        fuzzy_sat_lrl = np.mean(np.array([p[key]
                                          for p in results_lrl
                                          if p['w'] == w]), axis=0)

        x_axis = list(range(n_steps + 1))
        plt.plot(x_axis, fuzzy_sat_lrl, label='LRL')

        for reg_l in regularization_lambda_list:
            fuzzy_sat_ltn = np.mean(np.array([p[key]
                                              for p in results_ltn
                                              if p['lr'] == w and p['lambda'] == reg_l]), axis=0)

            plt.plot(x_axis, fuzzy_sat_ltn, label='SGD l=' + str(reg_l))

        plt.suptitle(label)
        plt.title('w=lr=' + str(w))
        plt.xlabel('iteration')
        plt.ylabel(y_label)
        plt.legend(loc='upper right')

        plt.savefig('plots/' + file_name + str(w) + '.png')
        plt.close()


# def generate_scatterplot(fuzzy, label, file_name):
#     if fuzzy:
#         key_sat = 'sat_f'
#         key_norm = 'norm_f'
#     else:
#         key_sat = 'sat_c'
#         key_norm = 'norm_c'
#
#     for w in w_lr_list:
#         fuzzy_sat_lrl = np.mean(np.array([p[key_sat]
#                                           for p in results_lrl
#                                           if p['w'] == w]), axis=0)
#
#         x_axis = list(range(n_steps + 1))
#         plt.plot(x_axis, fuzzy_sat_lrl, label='LRL')
#
#         for reg_l in regularization_lambda_list:
#             fuzzy_sat_ltn = np.mean(np.array([p[key]
#                                               for p in results_ltn
#                                               if p['lr'] == w and p['lambda'] == reg_l]), axis=0)
#
#             plt.plot(x_axis, fuzzy_sat_ltn, label='SGD l=' + str(reg_l))
#
#         plt.suptitle(label)
#         plt.title('w=lr=' + str(w))
#         plt.xlabel('iteration')
#         plt.ylabel(y_label)
#         plt.legend(loc='upper right')
#
#         plt.savefig('plots/' + file_name + str(w) + '.png')
#         plt.close()


generate_plots('sat_f', 'Evolution of fuzzy sat over time', 'fuzzy sat', 'fuzzy_sat/w_')
generate_plots('sat_c', 'Evolution of sat over time', 'sat', 'crisp_sat/w_')
generate_plots('norm_f', 'Evolution of L1 norm over time (fuzzy setting)', 'L1 norm', 'fuzzy_norm/w_')
generate_plots('norm_c', 'Evolution of L1 norm over time (classic setting)', 'L1 norm', 'crisp_norm/w_')

