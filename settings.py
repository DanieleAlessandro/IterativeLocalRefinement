# Fixed settings
n_initial_vectors = 10
n_steps = 100
n_formulas = 10

amt_rules = 10 # Max is 91

# Hyper parameters
targets = [0.3, 0.5, 0.8, 1.0]  # list of target truth values for both LRL and LTN
# methods = ['max', 'mean', 'min']  # list of aggregation methods for LRL
methods = ["mean", "max"]  # list of aggregation methods for LRL
regularization_lambda_list = [1.0, 0.1, 0.01, 0.0]  # regularization hyperparameter for LTN
# regularization_lambda_list = []  # regularization hyperparameter for LTN
lrl_schedules = [1.0, 0.1] # The scheduling on the delta of LRL