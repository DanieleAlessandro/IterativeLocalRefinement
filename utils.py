import torch.linalg

from Formula import *
from Godel import *
from logical_layers import *


def parse_cnf(in_data, amt_rules: int):
    cnf = list()
    cnf.append(list())
    maxvar = 0

    for line in in_data:
        tokens = line.split()

        if len(tokens) != 0:
            if tokens[0] not in ("p", "c", "%"):
                if len(cnf) == amt_rules + 2:
                    break
                for tok in tokens:
                    lit = int(tok)
                    maxvar = max(maxvar, abs(lit))
                    if lit == 0:
                        cnf.append(list())
                    else:
                        # lit = lit - 1 if lit > 0 else lit + 1
                        cnf[-1].append(lit)

            elif tokens[0] == 'p':
                number_of_variables = int(tokens[2])

    assert len(cnf[-1]) == 0
    cnf.pop()
    cnf.pop()
    return cnf, number_of_variables


def initialize_pre_activations(number_of_variables, number_of_trials, device):
    """Initialize pre-activations for both LTN and LRL (same initial value)

    :param number_of_variables: number of propositions
    :param number_of_trials: number of different initial truth values
    """
    t = torch.rand([number_of_trials, number_of_variables], device=device)
    z = torch.logit(t)
    # z.to(device=device)

    yield z
    while True:
        yield torch.nn.Parameter(torch.clone(z), requires_grad=True)


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


def evaluate_solutions(formula, predictions_list, initial_predictions, fuzzy=True):
    '''Returns the level of satisfaction of ain interpretation of a formula and the L1 norm of the delta values.

    :param formula: the formula to be optimized
    :param predictions_list: a list of predictions, each corresponding to a specific optimization step
    :param initial_predictions: the initial random interpretation
    :return: a list of satisfaction levels, one for each step; a list of l1 norms of the change with respect the
    initial value
    '''
    satisfactions = []
    norm1s = []
    if fuzzy:
        norm2s = []
        for predictions in predictions_list:
            form_sat = formula.satisfaction(predictions)
            satisfactions.append(torch.mean(form_sat).tolist())
            norm1s.append(torch.linalg.vector_norm(predictions - initial_predictions, ord=1).tolist())
            norm2s.append(torch.linalg.vector_norm(predictions - initial_predictions, ord=2).tolist())

        return satisfactions, norm1s, norm2s
    else:
        n_clauses = []

        for predictions in predictions_list:
            satisfactions.append(torch.mean(formula.satisfaction(predictions)).tolist())
            norm1s.append(torch.linalg.vector_norm(predictions - initial_predictions, ord=1).tolist())
            n_clauses.append(torch.mean(formula.sat_sub_formulas(predictions)).tolist())

        return satisfactions, norm1s, n_clauses


class LTNModel(torch.nn.Module):
    def __init__(self, formula):
        super().__init__()
        self.layer = LTN(formula)

    def forward(self, truth_values):  # Returns the satisfaction of the constraints
        return self.layer(torch.sigmoid(truth_values))