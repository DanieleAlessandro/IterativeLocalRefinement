# Fixed settings
n_initial_vectors = 10
n_steps = 100
n_formulas = 100

# Hyper parameters
targets = [0.3, 0.5, 0.8, 1.0]  # list of target truth values for both LRL and LTN
# methods = ['max', 'mean']  # list of aggregation methods for LRL
methods = ['mean']  # list of aggregation methods for LRL TODO: Currently only mean is implemented
regularization_lambda_list = [1.0, 0.1, 0.01, 0.0]  # regularization hyperparameter for LTN