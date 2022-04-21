import torch
import numpy as np
from Formula import Formula

class AND(Formula):
    def function(self, truth_values):
        m, _ = torch.min(truth_values, 1)
        self.forward_output = torch.unsqueeze(m, 0).t()

        return self.forward_output

    def boost_function(self, truth_values, delta):
        l = self.forward_output + delta
        return ((truth_values <= l) * l + (truth_values > l) * truth_values) - truth_values

    def get_name(self, parenthesis=False):
        s = ''
        for sf in self.sub_formulas[:-1]:
            s += sf.get_name(parenthesis=True) + ' AND '

        s += self.sub_formulas[-1].get_name(parenthesis=True)

        if parenthesis:
            return '(' + s + ')'
        else:
            return s

    def sat_sub_formulas(self, truth_values):
        inputs = []
        for sf in self.sub_formulas:
            inputs.append(sf.satisfaction(truth_values))

        if len(inputs) > 1:
            sf_satisfaction = torch.concat(inputs, 1)
        else:
            sf_satisfaction = inputs[0]

        return torch.mean(sf_satisfaction, 1, keepdim=True)

    def count_ones(self):
        return torch.sum((self.input_tensor == 1.).int(), 1, keepdim=True).float()

    def count_sat(self):
        return (torch.sum((self.input_tensor == 1.).int(), 1, keepdim=True) == self.input_tensor.shape[1]).float()

    def count_zeros(self):
        return torch.sum((self.input_tensor == 0.).int(), 1, keepdim=True).float()

    def count_all(self):
        return torch.tensor(self.input_tensor.shape[1]).float()


class OR(Formula):
    def function(self, truth_values):
        m, _ = torch.max(truth_values, 1)

        return torch.unsqueeze(m, 0).t()

    def boost_function(self, truth_values, delta):
        # TODO: remove or substitute: here if two values are both the maximum, it choose randomly which one to increase
        #   - slower
        #   - it should always reach a solution (if such a solution exists) with w=1 for the SAT problem
        # indexes = []
        # for row in truth_values:
        #     indexes.append(np.random.choice(torch.flatten(torch.argwhere(row == torch.max(row)))))
        #
        # # am = torch.argmax(truth_values, 1)
        # indexes = torch.tensor(np.array(indexes, dtype=np.int64))
        # return torch.nn.functional.one_hot(indexes, truth_values.shape[1]) * delta

        am = torch.argmax(truth_values, 1)

        return torch.nn.functional.one_hot(am, truth_values.shape[1]) * delta

    def get_name(self, parenthesis=False):
        s = ''
        for sf in self.sub_formulas[:-1]:
            s += sf.get_name(parenthesis=True) + ' OR '

        s += self.sub_formulas[-1].get_name(parenthesis=True)

        if parenthesis:
            return '(' + s + ')'
        else:
            return s