from Godel import *
from Formula import *
from LRL import LRL


class Model(torch.nn.Module):
    def __init__(self, formula, n_layers, backward):
        super().__init__()
        self.layers = []
        for _ in range(n_layers):
            self.layers.append(LRL(formula, backward))

    def forward(self, truth_values):
        x = torch.sigmoid(truth_values)
        for l in self.layers:
            x = l(x, 1.)

        return x




A = Predicate('A', 0)
B = Predicate('B', 1)
C = Predicate('C', 2)

# FIXED behaviour:
# C = Predicate('C', torch.tensor([[0.6,0.5,0.8,0.5]]).t()) (for w < 1)

# f = AND([NOT(A),OR([A,C])])
# f = AND([OR([NOT(A),NOT(B)]),OR([A,C])])
f = AND([OR([NOT(A),B]),OR([A,C])])

number_of_steps = 2

m = Model(f, number_of_steps, True)
# m = Model(f, 1, False)




t = torch.tensor([[0.1,0.5,0,0.8],[0.4,0.2,0.6,0.1],[0.6,0.1,0.8,0.5]]).t()
z = torch.nn.Parameter(torch.logit(t), requires_grad=True)
# t = torch.nn.Parameter(t, requires_grad=True)

print('Before:')
print(t)
print('After')
print(m(z))
# optimizer = torch.optim.SGD([z], lr=1.)
#
# for i in range(number_of_steps):
#     s = - torch.sum(m(z))
#     s.backward()
#     optimizer.step()
#
# print(m(z))
# print(torch.sigmoid(z))