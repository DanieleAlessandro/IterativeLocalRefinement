from Godel import *
from Formula import *
from LRL import *


class LRLModel(torch.nn.Module):
    def __init__(self, formula, n_layers):
        super().__init__()
        self.layers = []
        for _ in range(n_layers):
            self.layers.append(LRL(formula))

    def forward(self, truth_values):
        x = torch.sigmoid(truth_values)
        for l in self.layers:
            x = l(x, 1.)

        return x


class LTNModel(torch.nn.Module):
    def __init__(self, formula):
        super().__init__()
        self.layer = LTN(formula)

    def forward(self, truth_values):
        return self.layer(torch.sigmoid(truth_values), 1.)


# Define the settings of the experiments
lrl = True  # If false, then we use LTN
number_of_steps = 2


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
print(t)

print('After optimization:')
if lrl:
    print('Truth values:')
    print(m(z))
    print('Satisfaction of the constraints:')
    print(f.forward(m(z)))
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
