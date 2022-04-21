import torch


class LRL(torch.nn.Module):
    def __init__(self, formula):
        super().__init__()
        self.formula = formula

    def forward(self, truth_values, w):
        satisfaction = self.formula.forward(truth_values)

        # delta_sat = (1. - satisfaction) * w  # TODO: which one?
        # delta_sat = torch.where(1. - satisfaction < w, (1. - satisfaction).double(), w).float()
        condition = w - satisfaction > 0
        if torch.sum(condition.int()) == 0:
            return None

        delta_sat = torch.where(w - satisfaction > 0, (w - satisfaction).double(), 0.).float() / 10

        self.formula.backward(delta_sat)
        delta_tensor = self.formula.get_delta_tensor(truth_values)

        # The clip function is called to remove numeric errors (small negative values)
        return torch.clip(truth_values + delta_tensor, min=0.0, max=1.0)


class LTN(torch.nn.Module):
    def __init__(self, formula):
        super().__init__()
        self.formula = formula

    def forward(self, truth_values, w):
        return self.formula.forward(truth_values)
