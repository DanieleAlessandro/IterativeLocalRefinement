# Fixed settings
n_trials = 30
n_steps = 200

# Variable settings
targets = [0.5, 0.75, 0.9, 1.0]  # list of target truth values for LRL
methods = ['mean', 'mean']  # list of aggregation methods for LRL

lr_list = [1.0, 0.5, 0.1, 0.01, 0.001]  # learning rate list for LTN
regularization_lambda_list = [1.0, 0.5, 0.1, 0.01, 0.0]  # regularization hyperparameter for LTN
