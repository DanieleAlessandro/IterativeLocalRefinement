import torch

class LRL(torch.nn.Module):
    def __init__(self, formula, max_iterations, method='mean', schedule=1.0, convergence_condition=1e-4, min_iterations=10):
        super().__init__()
        self.formula = formula
        self.schedule = schedule
        self.max_iterations = max_iterations
        self.method = method
        self.conv_condition = convergence_condition
        self.min_iterations = min_iterations

    def forward(self, initial_t, w):
        predictions = [initial_t]
        satisfactions = []
        truth_values = initial_t
        prev_satisfaction = torch.tensor(10000.)
        for i in range(self.max_iterations):
            satisfaction = self.formula.forward(truth_values)
            # print(satisfaction)
            satisfactions.append(satisfaction)

            # Convergence criterion has an hyperparameter. Make sure it's not too small.
            condition = (prev_satisfaction - satisfaction).abs() > self.conv_condition
            if i > self.min_iterations and torch.sum(condition.int()) == 0:
                break

            # delta_sat = torch.where(target - satisfaction > 0, (target - satisfaction).double(), 0.).float() * self.schedule
            active_mask = (w - satisfaction).abs() > self.conv_condition
            # print((w - satisfaction).abs(), (w - prev_satisfaction).abs())
            delta_sat = torch.where(
                active_mask,
                (w - satisfaction).double() * self.schedule,
                0.).float()

            self.formula.backward(delta_sat)
            delta_tensor = self.formula.get_delta_tensor(truth_values, self.method)
            delta_tensor[~active_mask] = torch.zeros((delta_tensor.shape[-1]))

            # The clip function is called to remove numeric errors (small negative values)
            truth_values = torch.clip(truth_values + delta_tensor, min=0.0, max=1.0)
            predictions.append(truth_values)
            prev_satisfaction = satisfaction
        return predictions


class LTN(torch.nn.Module):
    def __init__(self, formula):
        super().__init__()
        self.formula = formula

    def forward(self, truth_values):
        return self.formula.forward(truth_values)
