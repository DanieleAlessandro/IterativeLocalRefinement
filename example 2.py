from Godel import *
from Formula import *

# TODO: idea: randomized BFs -> t-conorm: softmax used to sample the value to be increased
# TODO: understand KENN with different formulas
# TODO: idea: change randomly a solution where a subformula is not satisfied -> just maxsat, TBFs allows to move faster (the initial random solution can be changed)

A = Predicate('A', torch.tensor([[0.1,0.5,0,0.8]]).t())
B = Predicate('B', torch.tensor([[0.4,0.2,0.6,0.1]]).t())
C = Predicate('C', torch.tensor([[0.6,0.1,0.8,0.5]]).t())

# FIXED behaviour:
# C = Predicate('C', torch.tensor([[0.6,0.5,0.8,0.5]]).t()) (for w < 1)

# f = AND([NOT(A),OR([A,C])])
# f = AND([OR([NOT(A),NOT(B)]),OR([A,C])])
f = AND([OR([NOT(A),B]),OR([A,C])])


print('Before:')

print('Predicates initial values:\nA')
print(A.value)
print('B')
print(B.value)
print('C')
print(C.value)

initial_value = f.forward()
f.print_table()

n_iterations = 2

for _ in range(n_iterations):
    initial_value = f.forward()
    # f.print_table()
    delta = (1 - initial_value)
    f.backward(delta)
    A.aggregate_deltas()
    B.aggregate_deltas()
    C.aggregate_deltas()

print('After:')
f.print_table()

print('Predicates final values:\nA')
print(A.value)
print('B')
print(B.value)
print('C')
print(C.value)
