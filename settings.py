# Fixed settings
n_initial_vectors = 10
n_steps = 300
n_formulas = 10

amt_rules = [10] # Max is 91 [10, 20, 30, 40, 50, 60, 70, 80, 91]

# Hyper parameters
## Runs
tnorm = 'lukasiewicz' # Choose from 'godel', 'lukasiewicz', 'product'
targets = [0.3, 0.5, 0.8, 1]  # list of target truth values for both LRL and LTN

## LRL
methods = ['mean', 'max']  # list of aggregation methods for LRL
lrl_schedules = [1.0, 0.1] # The scheduling on the delta of LRL

## SGD
regularization_lambda_list = [0.01, 0.1, 0]  # regularization hyperparameter for LTN
sgd_norm = 1 if tnorm == 'product' else 2
sgd_methods = ['adam'] # ['sgd', 'adam']  # list of SGD methods
# regularization_lambda_list = []  # regularization hyperparameter for LTN

verbose = True
test_correctness = True
use_cuda = False

# Plotting
amt_rules_plot = 10
tnorms_plot = ["product", "godel", "lukasiewicz"]
