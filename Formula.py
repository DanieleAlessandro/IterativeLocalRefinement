import torch
from prettytable import PrettyTable


class Formula:
    def __init__(self, sub_formulas):
        self.sub_formulas = sub_formulas
        self.input_tensor = None

    def function(self, truth_values):
        pass

    def boost_function(self, truth_values, delta):
        pass

    def get_name(self, parenthesis=False):
        pass

    def print_table(self):
        header = []
        for sf in self.sub_formulas:
            header.append(sf.get_name(parenthesis=True))
        header.append(self.get_name())

        pt = PrettyTable(header)
        results = self.forward()
        pt.add_rows(torch.concat([self.input_tensor, results], 1).numpy())
        print(pt)

    def __str__(self):
        s = self.get_name() + '\n'
        s += str(self.input_tensor)

        return s

    def forward(self):
        inputs = []
        for sf in self.sub_formulas:
            inputs.append(sf.forward())

        self.input_tensor = torch.concat(inputs, 1)
        return self.function(self.input_tensor)

    def backward(self, delta, randomized=False):
        deltas = self.boost_function(self.input_tensor, delta)
        if randomized:
            deltas = deltas * torch.rand(deltas.shape)

        for sf, d in zip(self.sub_formulas, deltas.t()):
            sf.backward(torch.unsqueeze(d, 0).t())


class Predicate(Formula):
    def __init__(self, name, value):
        super().__init__(None)
        self.name = name
        self.value = value
        self.deltas = []

    def forward(self):
        return self.value

    def backward(self, delta):
        self.deltas.append(delta)

    def update(self, method='max'):
        if method == 'most_clauses':
            deltas = torch.concat(self.deltas, 1)
            positive = torch.sum(deltas > 0., 1, keepdim=True) - torch.sum(deltas <= 0., 1, keepdim=True) >= 0
            max, _ = torch.max(deltas, 1, keepdim=True)
            min, _ = torch.min(deltas, 1, keepdim=True)

            self.value = self.value + torch.where(positive, max, min)
            self.deltas = []
        if method == 'mean':
            deltas = torch.concat(self.deltas, 1)

            self.value = self.value + torch.mean(deltas, 1, keepdim=True)
            self.deltas = []
        if method == 'max':
            deltas = torch.concat(self.deltas, 1)
            abs_deltas = deltas.abs()

            i = torch.argmax(abs_deltas, 1, keepdim=True)

            self.value = self.value + torch.gather(deltas, 1, i)
            self.deltas = []
        if method == 'min':
            deltas = torch.concat(self.deltas, 1)
            abs_deltas = deltas.abs()

            i = torch.argmin(abs_deltas, 1, keepdim=True)

            self.value = self.value + torch.gather(deltas, 1, i)
            self.deltas = []

    def get_name(self, parenthesis=False):
        return self.name


class NOT(Formula):
    def __init__(self, sub_formula):
        super().__init__([sub_formula])

    def function(self, truth_values):
        return 1 - truth_values

    def boost_function(self, truth_values, delta):
        return - delta

    def get_name(self, parenthesis=False):
        return 'NOT(' + self.sub_formulas[0].get_name() + ')'