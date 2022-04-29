import torch

class LRL(torch.nn.Module):
    def __init__(self, formula, max_iterations, method='mean', schedule=1.0, convergence_condition=1e-5):
        super().__init__()
        self.formula = formula
        self.schedule = schedule
        self.max_iterations = max_iterations
        self.method = method
        self.conv_condition = convergence_condition

    def forward(self, initial_t, w):
        predictions = [initial_t]
        truth_values = initial_t
        prev_satisfaction = w
        for i in range(self.max_iterations):
            satisfaction = self.formula.forward(truth_values)

            # delta_sat = (1. - satisfaction) * w  # TODO: which one?
            # delta_sat = torch.where(1. - satisfaction < w, (1. - satisfaction).double(), w).float()
            # Convergence criterion has an hyperparameter. Make sure it's not too small.


            condition = (prev_satisfaction - satisfaction).abs() > self.conv_condition
            if torch.sum(condition.int()) == 0:
                break

            # TODO: What about this weird parameter here? It seems to really smooth out the trajectory
            # delta_sat = torch.where(target - satisfaction > 0, (target - satisfaction).double(), 0.).float() * self.schedule
            delta_sat = torch.where(
                (w - satisfaction).abs() > self.conv_condition,
                (w - satisfaction).double() * self.schedule,
                0.).float()

            self.formula.backward(delta_sat)
            delta_tensor = self.formula.get_delta_tensor(truth_values, self.method)

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
