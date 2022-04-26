from typing import Tuple

import torch
from abc import ABC, abstractmethod

from Formula import Formula

class SATTNorm(ABC):

    clause_t: torch.Tensor

    @abstractmethod
    def forward(self, indexed_t: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def boost_function(self, delta: torch.Tensor, prop_index: torch.Tensor, sign: torch.Tensor) -> torch.Tensor:
        pass

    def sgd_function(self, indexed_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        t = self.forward(indexed_t)[0]
        return t, t

class SATGodel(SATTNorm):

    def forward(self, indexed_t: torch.Tensor) -> torch.Tensor:
        self.clause_t_g = torch.max(indexed_t, dim=2)
        self.clause_t = self.clause_t_g[0]
        self.formula_t = torch.min(self.clause_t, dim=1)[0]
        return self.formula_t

    def boost_function(self, delta, prop_index, sign) -> torch.Tensor:
        lambda_min = (self.formula_t + delta).unsqueeze(1)
        # t-norm TBF
        delta_min = (self.clause_t <= lambda_min) * (lambda_min - self.clause_t)
        max_indices = self.clause_t_g[1].unsqueeze(-1)

        # The delta applied on the max truth values on each clause
        # Find max inputs for t-conorm TBF. Multiply with sign to deal with negated literals
        delta_max = torch.take_along_dim(sign.unsqueeze(0), max_indices, -1).squeeze() * delta_min

        # Find what propositions the results belong to
        propositions_max_indices = torch.take_along_dim(prop_index.unsqueeze(0), max_indices, -1).squeeze()
        # Uses the mean to combine the deltas on the same proposition
        # How to compute the mean on nonzero values: Take the sum (scatter_add), then count amount of nonzero values
        prop_deltas = torch.scatter_reduce(delta_max, -1, propositions_max_indices, 'sum')

        amt_nonzero = torch.scatter_reduce((delta_max != 0).float(), -1, propositions_max_indices, 'sum')
        mask = amt_nonzero > 0
        prop_deltas[mask] = prop_deltas[mask] / amt_nonzero[mask]
        return prop_deltas

class SATLukasiewicz(SATTNorm):

    def forward(self, indexed_t: torch.Tensor) -> torch.Tensor:
        self.indexed_t = indexed_t
        self.clause_t = torch.clip(torch.sum(indexed_t, dim=-1), 0, 1)
        n = self.clause_t.shape[-1]
        self.formula_t = torch.clip(torch.sum(self.clause_t, dim=-1) - (n-1), 0, 1)
        return self.formula_t

    def boost_function(self, delta, prop_index, sign) -> torch.Tensor:
        # Compute the t-norm boost function
        w = (self.formula_t + delta).unsqueeze(1)
        n = self.clause_t.shape[-1]
        sorted_clauses = torch.sort(self.clause_t, dim=-1, descending=False)

        # Find the delta_M for each M
        M = torch.arange(1, n+1)
        delta_M = (w + M - 1 - torch.cumsum(sorted_clauses[0], dim=-1)) / M

        cond = delta_M < 1 - sorted_clauses[0]
        # Do cond - cond (shifted to the left), then take the argmax.
        # This finds the largest index for which the condition is true
        M_star = torch.argmax(cond.float() - torch.roll(cond.float(), -1, -1), dim=-1)
        # If the condition holds for all clauses, then the max index is the last one
        M_star[cond.all(dim=-1)] = n - 1
        delta_M_star = torch.gather(delta_M, -1, M_star.unsqueeze(-1))

        # Assign the respective deltas to the clause truth values
        delta_sorted_clauses = torch.where(cond, delta_M_star, torch.tensor(0.0))
        delta_clauses = torch.scatter_reduce(delta_sorted_clauses, -1, sorted_clauses[1], 'sum')

        # Compute the t-conorm boost function
        clauses_w = torch.clip(self.clause_t + delta_clauses, 0, 1)
        delta_literals = (torch.clip(clauses_w - torch.sum(self.indexed_t, dim=-1), 0, 1) / 3.0).unsqueeze(-1) * sign

        # Distribute the deltas over the propositions using the mean aggregation
        prop_index_expand = prop_index.unsqueeze(0).expand(delta_literals.shape).flatten(1, 2)
        delta_literals_flat = delta_literals.flatten(1, 2)
        prop_deltas = torch.scatter_reduce(delta_literals_flat, -1, prop_index_expand, 'sum')
        nonzero_deltas = torch.scatter_reduce((delta_literals_flat != 0).float(), -1, prop_index_expand, 'sum')
        mask = nonzero_deltas > 0
        prop_deltas[mask] = prop_deltas[mask] / nonzero_deltas[mask]

        return prop_deltas

    def sgd_function(self, indexed_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.clause_t = torch.clip(torch.sum(indexed_t, dim=-1), 0, 1)
        n = self.clause_t.shape[-1]
        s = torch.sum(self.clause_t, dim=-1) - (n - 1)
        formula_t = torch.clip(s, 0, 1)
        return s, formula_t



class SATFormula(Formula):
    def __init__(self, clauses, is_sgd=False, tnorm=SATGodel()):
        self.clause_t = None
        self.is_sgd = is_sgd
        formula_tensor = torch.tensor(clauses)
        self.prop_index = formula_tensor.abs() - 1
        # self.prop_index = formula_tensor.abs()
        self.sign = formula_tensor.sign()
        self.tnorm = tnorm

    def function(self, indexed_t) -> torch.Tensor:
        self.input_tensor = indexed_t

        self.formula_t = self.tnorm.forward(indexed_t)
        return self.formula_t

    def boost_function(self, truth_values, delta) -> torch.Tensor:
        return self.tnorm.boost_function(delta, self.prop_index, self.sign)

    def get_name(self, parenthesis=False):
        pass

    def __str__(self):
        return self.get_name() + '\n'

    def forward(self, truth_values):
        indexed_t = truth_values[..., self.prop_index]
        indexed_t[..., self.sign < 0] = 1 - indexed_t[..., self.sign < 0]
        if self.is_sgd:
            return self.tnorm.sgd_function(indexed_t)
        return self.function(indexed_t)

    def backward(self, delta, randomized=False):
        self.prop_deltas = self.boost_function(self.input_tensor, delta)
        return self.prop_deltas

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
        if self.is_sgd:
            return s[1]
        return s

    def sat_sub_formulas(self, truth_values):
        return self.tnorm.clause_t

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
