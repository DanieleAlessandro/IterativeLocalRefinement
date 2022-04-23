import torch

from Formula import Formula


class SATFormula(Formula):
    def __init__(self, clauses):
        self.clause_t = None
        formula_tensor = torch.tensor(clauses)
        self.prop_index = formula_tensor.abs() - 1
        self.sign = formula_tensor.sign()

    def function(self, truth_values):
        indexed_t = truth_values[..., self.prop_index]
        indexed_t[..., self.sign < 0] = 1 - indexed_t[..., self.sign < 0]

        self.input_tensor = indexed_t

        # TODO: Implement also for other t-norms
        self.clause_t = torch.max(indexed_t, dim=2)
        self.formula_t = torch.min(self.clause_t[0], dim=1)[0]
        return self.formula_t

    def boost_function(self, truth_values, delta):
        lambda_min = (self.formula_t + delta).unsqueeze(1)
        # t-norm TBF
        delta_min = (self.clause_t[0] <= lambda_min) * (lambda_min - self.clause_t[0])
        max_indices = self.clause_t[1].unsqueeze(-1)

        # The delta applied on the max truth values on each clause
        # Find max inputs for t-conorm TBF. Multiply with sign to deal with negated literals
        delta_max = torch.take_along_dim(self.sign.unsqueeze(0), max_indices, -1).squeeze() * delta_min

        # Find what propositions the results belong to
        propositions_max_indices = torch.take_along_dim(self.prop_index.unsqueeze(0), max_indices, -1).squeeze()
        # Uses the mean to combine the deltas on the same proposition
        self.prop_deltas = torch.scatter_reduce(delta_max, -1, propositions_max_indices, "mean")

        # print(self.prop_deltas, self.prop_deltas.shape)

        return delta_max

    def get_name(self, parenthesis=False):
        pass

    def __str__(self):
        s = self.get_name() + '\n'
        # s += str(self.input_tensor)

        return s

    def forward(self, truth_values):
        return self.function(truth_values)

    def backward(self, delta, randomized=False):
        return self.boost_function(self.input_tensor, delta)

        # TODO: Randomized is not implemented for now
        # if randomized:
        #     deltas = deltas * torch.rand(deltas.shape)

    def get_delta_tensor(self, truth_values, method='mean'):
        return self.prop_deltas

    def reset_deltas(self):
        pass

    def satisfaction(self, truth_values):
        s = self.forward(truth_values)
        self.reset_deltas()

        return s

    def sat_sub_formulas(self, truth_values):
        return torch.mean(self.clause_t[0], 1, keepdim=True)

#
# class Predicate(Formula):
#     def __init__(self, name, index):
#         super().__init__(None)
#         self.name = name
#         self.index = index
#         self.deltas = []
#         self.predicates = [self]
#
#     def forward(self, truth_values):
#         return torch.unsqueeze(truth_values[:, self.index], 1)
#
#     def backward(self, delta, randomized=False):  # TODO: implement the usage of randomized
#         self.deltas.append(delta)
#
#     def reset_deltas(self):
#         self.deltas = []
#
#     def aggregate_deltas(self, method='mean') -> Tuple[int, torch.Tensor]:
#         if method == 'most_clauses':
#             deltas = torch.concat(self.deltas, 1)
#             positive = torch.sum(deltas > 0., 1, keepdim=True) - torch.sum(deltas <= 0., 1, keepdim=True) >= 0
#             max, _ = torch.max(deltas, 1, keepdim=True)
#             min, _ = torch.min(deltas, 1, keepdim=True)
#
#             return self.index, torch.where(positive, max, min)
#         if method == 'mean':
#             deltas = torch.concat(self.deltas, 1)
#
#             return self.index, torch.nan_to_num(
#                 torch.sum(deltas, 1, keepdim=True) / torch.sum(deltas != 0.0, 1, keepdim=True))
#         if method == 'max':
#             deltas = torch.concat(self.deltas, 1)
#             abs_deltas = deltas.abs()
#
#             i = torch.argmax(abs_deltas, 1, keepdim=True)
#
#             return self.index, torch.gather(deltas, 1, i)
#         if method == 'min':
#             deltas = torch.concat(self.deltas, 1)
#             deltas = torch.where(deltas == 0, 100., deltas.double()).float()
#             abs_deltas = deltas.abs()
#
#             i = torch.argmin(abs_deltas, 1, keepdim=True)
#             final_deltas = torch.gather(deltas, 1, i)
#
#             return self.index, torch.where(final_deltas == 100., 0.0, final_deltas.double()).float()
#         if method == 'randomized_direction':
#             deltas = torch.concat(self.deltas, 1)
#             abs_deltas = deltas.abs()
#
#             if random.getrandbits(1):
#                 deltas_no_zeros = torch.where(abs_deltas == 0, 100., abs_deltas.double()).float()
#                 i = torch.argmin(deltas_no_zeros, 1, keepdim=True)
#                 final_deltas = torch.gather(deltas, 1, i)
#
#                 return self.index, torch.where(final_deltas == 100., 0.0, final_deltas.double()).float()
#             else:
#                 i = torch.argmax(abs_deltas, 1, keepdim=True)
#
#                 return self.index, torch.gather(deltas, 1, i)
#
#     def get_name(self, parenthesis=False):
#         return self.name
#
#
# class NOT(Formula):
#     def __init__(self, sub_formula):
#         super().__init__([sub_formula])
#
#     def function(self, truth_values):
#         return 1 - truth_values
#
#     def boost_function(self, truth_values, delta):
#         return - delta
#
#     def get_name(self, parenthesis=False):
#         return 'NOT(' + self.sub_formulas[0].get_name() + ')'
