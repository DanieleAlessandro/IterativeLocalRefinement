from Formula import *
from Godel import *
from logical_layers import *


def parse_cnf(in_data):
    cnf = list()
    cnf.append(list())
    maxvar = 0

    for line in in_data:
        tokens = line.split()

        if len(tokens) != 0:
            if tokens[0] not in ("p", "c", "%"):
                for tok in tokens:
                    lit = int(tok)
                    maxvar = max(maxvar, abs(lit))
                    if lit == 0:
                        cnf.append(list())
                    else:
                        lit = lit - 1 if lit > 0 else lit + 1
                        cnf[-1].append(lit)
            elif tokens[0] == 'p':
                number_of_variables = int(tokens[2])


    assert len(cnf[-1]) == 0
    cnf.pop()
    cnf.pop()
    return cnf, number_of_variables


def initialize_pre_activations(number_of_variables, number_of_trials):
    """Initialize pre-activations for both LTN and LRL (same initial value)

    :param number_of_variables: number of propositions
    :param number_of_trials: number of different initial truth values
    """
    z = (torch.rand([number_of_trials, number_of_variables]) - 0.5) * 10

    yield z
    while True:
        yield torch.nn.Parameter(torch.clone(z), requires_grad=True)  # TODO: check if it works properly


def create_predicates(number_of_variables):
    predicates = []
    for i in range(number_of_variables):
        predicates.append(Predicate('x_' + str(i), i))

    return predicates


def create_formula(predicates, clauses):
    or_sub_formulas = []

    for clause in clauses:
        sub_formula_literals = []
        for literal in clause:
            if literal < 0:
                sub_formula_literals.append(NOT(predicates[-literal]))
            else:
                sub_formula_literals.append(predicates[literal])

        or_sub_formulas.append(OR(sub_formula_literals))

    return AND(or_sub_formulas)


def defuzzify(tensor):
    return (tensor > 0.5).float()


def defuzzify_list(tensors_list):
    return [defuzzify(tensor) for tensor in tensors_list]


def evaluate_solutions(formula, predictions_list, initial_predictions):
    '''Returns the level of satisfaction of ain interpretation of a formula and the L1 norm of the delta values.

    :param formula: the formula to be optimized
    :param predictions_list: a list of predictions, each corresponding to a specific optimization step
    :param initial_predictions: the initial random interpretation
    :return: a list of satisfaction levels, one for each step; a list of l1 norms of the change with respect the
    initial value
    '''
    satisfactions = []
    norms = []

    for predictions in predictions_list:
        satisfactions.append(torch.mean(formula.satisfaction(predictions)).tolist())
        norms.append(torch.linalg.vector_norm(predictions - initial_predictions, ord=1).tolist())

    return satisfactions, norms


class LRLModel(torch.nn.Module):
    def __init__(self, formula, n_layers, w):
        super().__init__()
        self.formula = formula
        self.layers = []
        self.w = w
        for _ in range(n_layers):
            self.layers.append(LRL(formula))

    def forward(self, pre_activations):
        predictions = [torch.sigmoid(pre_activations)]

        for l in self.layers:
            predictions.append(l(predictions[-1], self.w))  # TODO: need to check

        return predictions


class LTNModel(torch.nn.Module):
    def __init__(self, formula):
        super().__init__()
        self.layer = LTN(formula)

    def forward(self, truth_values):  # Returns the satisfaction of the constraints
        return self.layer(torch.sigmoid(truth_values), 1.)