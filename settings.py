# Fixed settings
n_initial_vectors = 10
n_steps = 100
n_formulas = 100

# Variable settings
targets = [0.3, 0.5, 0.8, 1.0]  # list of target truth values for LRL
methods = ['max', 'mean']  # list of aggregation methods for LRL

regularization_lambda_list = [1.0, 0.1, 0.01, 0.0]  # regularization hyperparameter for LTN
lr_list = [1.0, 0.1, 0.01, 0.001]  # learning rate list for LTN

