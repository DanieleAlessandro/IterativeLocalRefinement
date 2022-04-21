import numpy as np
import pickle
import matplotlib.pyplot as plt

# Fixed settings
n_trials = 30  # TODO: 10 trials??
n_steps = 100

# Variable settings
targets = [0.75, 0.9, 1.0]  # 0.5,  list of target truth values

lr_list = [0.1]  # , 0.01, 0.001]
regularization_lambda_list = [0.01, 0.0]  # [0.5, 0.1, 0.01, 0.0]


with open('results_lrl', 'rb') as f:
    results_lrl = pickle.load(f)

with open('results_ltn', 'rb') as f:
    results_ltn = pickle.load(f)

# results_ltn.append({
#     'formula': filename,
#     'target': t,
#     'lr': lr,
#     'lambda': reg_lambda,
#     'sat_f': ltn_sat_f,
#     'sat_c': ltn_sat_c,
#     'norm_f': ltn_norm_f,
#     'norm_c': ltn_norm_c
# })


def generate_plots(key, label, y_label, file_name):
    for t in targets:
        plt.suptitle(label + '\n w=' + str(t))
        # plt.title()
        plt.xlabel('iteration')
        plt.ylabel(y_label)
        plt.subplots_adjust(right=0.7)
        plt.rcParams["figure.figsize"] = (8.5, 5)

        fuzzy_sat_lrl = np.mean(np.array([p[key]
                                          for p in results_lrl
                                          if p['target'] == t]), axis=0)

        x_axis = list(range(n_steps + 1))

        if len(fuzzy_sat_lrl) < n_steps + 1:
            to_fill = n_steps + 1 - len(fuzzy_sat_lrl)
            fuzzy_sat_lrl = np.concatenate([fuzzy_sat_lrl, np.array([fuzzy_sat_lrl[-1]] * to_fill)])

        plt.plot(x_axis, fuzzy_sat_lrl, color='k', label='LRL')

        for reg_l in regularization_lambda_list:
            for lr in lr_list:
                fuzzy_sat_ltn = np.mean(np.array([p[key]
                                                  for p in results_ltn
                                                  if p['target'] == t and p['lr'] == lr and p['lambda'] == reg_l]), axis=0)

                plt.plot(x_axis, fuzzy_sat_ltn, label='SGD (' + str(reg_l) + ',' + str(lr) + ')')

        plt.legend(bbox_to_anchor=(1.45, 0.9), loc='upper right')

        plt.savefig('plots/' + file_name + str(t) + '.png', bbox_inches="tight")
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

