from Godel import *
from Formula import *
from logical_layers import *
from utils import *


# Define the settings of the experiments
lrl = True  # If True, use LRL. If False, LTN
number_of_steps = 2  # Number of optimization steps


# Define the knowledge
A = Predicate('A', 0)
B = Predicate('B', 1)
C = Predicate('C', 2)

# f = AND([NOT(A),OR([A,C])])
# f = AND([OR([NOT(A),NOT(B)]),OR([A,C])])
f = AND([OR([NOT(A),B]),OR([A,C])])

# Define the initial truth values and pre-activations
t = torch.tensor([[0.1,0.5,0,0.8],[0.4,0.2,0.6,0.1],[0.6,0.1,0.8,0.5]]).t()
z = torch.nn.Parameter(torch.logit(t), requires_grad=True)

# Define the model
if lrl:
    m = LRLModel(f, number_of_steps)
else:
    m = LTNModel(f)


print('Before optimization:')
print('Truth values:')
print(torch.sigmoid(z))
print('Satisfaction of the constraints:')
print(f.satisfaction(torch.sigmoid(z)))


print('After optimization:')
if lrl:
    print('Truth values:')
    print(m(z))
    print('Satisfaction of the constraints:')
    print(f.satisfaction(m(z)))
else:
    optimizer = torch.optim.SGD([z], lr=1.)

    for i in range(number_of_steps):
        s = - torch.sum(m(z))
        s.backward()
        optimizer.step()

    print('Truth values:')
    print(torch.sigmoid(z))
    print('Satisfaction of the constraints:')
    print(m(z))
