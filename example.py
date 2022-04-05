from Godel import *
from Formula import *


def print_values(A, B, V):
    for a, b, v in zip(A.value, B.value, V):
        print(str(a[0]) + ', ' + str(b[0]) + ', ' + str(v[0]))


A = Predicate('A', torch.tensor([[0.1,0.5,0,0.8]]).t())
B = Predicate('B', torch.tensor([[0.4,0.2,0.6,0.1]]).t())
C = Predicate('C', torch.tensor([[0.6,0.1,0.8,0.3]]).t())
f = AND([NOT(A),OR([B,C])])
print(f.get_name())


print('Before:')

print('Predicates final values:\nA')
print(A.value)
print('B')
print(B.value)
print('C')
print(C.value)

initial_value = f.forward()
f.print_table()
delta = (1 - initial_value) * 0.5
f.backward(delta)

print('After:')
f.print_table()

print('Predicates final values:\nA')
print(A.value)
print('B')
print(B.value)
print('C')
print(C.value)
