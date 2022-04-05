import torch
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


class OR(Formula):
    def function(self, truth_values):
        m, _ = torch.max(truth_values, 1)

        return torch.unsqueeze(m, 0).t()

    def boost_function(self, truth_values, delta):
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