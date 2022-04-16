import torch


class LRL(torch.nn.Module):
    def __init__(self, formula):
        super().__init__()
        self.formula = formula

    def forward(self, truth_values, w):
        satisfaction = self.formula.forward(truth_values)

        delta_sat = 1. - satisfaction * w
        self.formula.backward(delta_sat)

        return truth_values + self.formula.get_delta_tensor(truth_values)


class LTN(torch.nn.Module):
    def __init__(self, formula):
        super().__init__()
        self.formula = formula

    def forward(self, truth_values, w):
        return self.formula.forward(truth_values)
