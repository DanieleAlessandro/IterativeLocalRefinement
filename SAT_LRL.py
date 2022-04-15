from Formula import *
from Godel import *
import os
import numpy as np
import pickle

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


def create_predicates(number_of_variables, number_of_trials):
    predicates = []
    for i in range(number_of_variables):
        predicates.append(Predicate('x_' + str(i), torch.rand([number_of_trials, 1])))

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

sat_results = [None] * 30
for filename in os.listdir('uf20-91'):
    with open(os.path.join('uf20-91', filename), 'r') as f:
        l = f.readlines()

    clauses, n = parse_cnf(l)
    predicates = create_predicates(n, 1000)
    f = create_formula(predicates, clauses)
    print(f)

    for i in range(30):
        initial_value = f.forward()
        # f.print_table()

        if i >= 0:
            delta = (1 - initial_value)
        else:
            delta = (1 - initial_value) * 0.95 #random.uniform(0,1)
        print('i=' + str(i) +
              ' -> mean:   ' + str(torch.mean(f.sat_sub_formulas()).tolist()) +
              '     max:   ' + str(torch.max(f.sat_sub_formulas()).tolist()) +
              '    ones:   ' + str(torch.mean(f.count_ones()).tolist()) +
              '   zeros:   ' + str(torch.mean(f.count_zeros()).tolist()) +
              '     sum:   ' + str((torch.mean(f.count_zeros()) + torch.mean(f.count_ones())).tolist()) +
              '     all:   ' + str(torch.mean(f.count_all()).tolist()) +
              '     sat:   ' + str(torch.mean(f.count_sat()).tolist()))
        if sat_results[i] is None:
            sat_results[i] = [torch.mean(f.count_sat()).tolist()]
        else:
            sat_results[i].append(torch.mean(f.count_sat()).tolist())

        if i < 0:
            f.backward(delta, randomized=i % 3 == 0)
        else:
            f.backward(delta) #, randomized=i % 10 == 0)

        for predicate in predicates:
            if i <= 0:
                predicate.update('most_clauses')
            else:
                predicate.update('max')

# results = []
for i, formula in enumerate(sat_results):
    print(str(i) + ':   ' + str(np.mean(formula)))
    # results.append(np.mean(formula))

with open('results', 'wb') as f:
    pickle.dump(sat_results, f)