import torch


class LRL(torch.nn.Module):
    def __init__(self, formula, backward=True):
        super().__init__()
        self.formula = formula
        self.bw = backward

    def forward(self, truth_values, w):
        satisfaction = self.formula.forward(truth_values)

        if self.bw:
            delta_sat = 1. - satisfaction * w
            self.formula.backward(delta_sat)

            return truth_values + self.formula.get_delta_tensor(truth_values)
        else:
            return satisfaction