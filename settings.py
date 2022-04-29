# Fixed settings
n_initial_vectors = 10
n_steps = 300
n_formulas = 1000

amt_rules = [10] # Max is 91 [10, 20, 30, 40, 50, 60, 70, 80, 91]

# Hyper parameters
tnorm = 'godel' # Choose from 'godel', 'lukasiewicz', 'product'
sgd_norm = 1 if tnorm == 'product' else 2
targets = [0.3, 0.5, 0.8, 1.0]  # list of target truth values for both LRL and LTN
methods = ['mean', "max"]  # list of aggregation methods for LRL
regularization_lambda_list = [0.5, 0.1, 0.01, 0.0]  # regularization hyperparameter for LTN
# regularization_lambda_list = []  # regularization hyperparameter for LTN
lrl_schedules = [1.0, 0.1] # The scheduling on the delta of LRL

verbose = True
use_cuda = False

# Plotting
amt_rules_plot = 10
tnorm_plot = tnorm
