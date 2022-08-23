from typing import Tuple

import torch
from abc import ABC, abstractmethod

from Formula import Formula
from torch_scatter import scatter_max, scatter_min, scatter_sum

from settings import test_correctness


def aggregate_mean(delta_literals: torch.Tensor, prop_index: torch.Tensor, props=20) -> torch.Tensor:
    # Uses the mean to combine the deltas on the same proposition
    # How to compute the mean on nonzero values: Take the sum (scatter_add), then count amount of nonzero values
    prop_deltas = scatter_sum(delta_literals, prop_index, -1, dim_size=props)

    # Check what values are nonzero (or not almost zero)
    amt_nonzero = scatter_sum((torch.abs(delta_literals) > 1e-6).float(), prop_index, -1, dim_size=props)
    mask = amt_nonzero > 0
    prop_deltas[mask] = prop_deltas[mask] / amt_nonzero[mask]
    return prop_deltas

def aggregate_max(delta_literals: torch.Tensor, prop_index: torch.Tensor, randomized=False, props=20) -> torch.Tensor:
    # Uses the absolute max value to combine the deltas on the same proposition
    prop_deltas_max = scatter_max(delta_literals, prop_index, -1, dim_size=props)[0]
    prop_deltas_min = scatter_min(delta_literals, prop_index, -1, dim_size=props)[0]
    if randomized:
        prob = torch.sigmoid((prop_deltas_max.abs() - prop_deltas_min.abs())/0.01)
        prop_deltas = torch.where(torch.bernoulli(prob).bool(), prop_deltas_max, prop_deltas_min)
    else:
        cond: torch.Tensor = prop_deltas_max.abs() > prop_deltas_min.abs()
        prop_deltas = torch.where(cond, prop_deltas_max, prop_deltas_min)
    return prop_deltas

def aggregate_min(delta_literals: torch.Tensor, prop_index: torch.Tensor, threshold=0.01, props=20) -> torch.Tensor:
    # Uses the absolute min values, among values larger than threshold, to combine the deltas on the same proposition
    prop_deltas_max = scatter_max(
        delta_literals * (delta_literals < -threshold) - (delta_literals >= -threshold).float(),
        prop_index, -1, dim_size=props)[0]
    prop_deltas_min = scatter_min(
        delta_literals * (delta_literals >  threshold) + (delta_literals <=  threshold).float(),
        prop_index, -1, dim_size=props)[0]
    prop_deltas = torch.where(prop_deltas_max.abs() < prop_deltas_min.abs(), prop_deltas_max, prop_deltas_min)
    prop_deltas[prop_deltas == 1] = 0
    prop_deltas[prop_deltas == -1] = 0
    return prop_deltas

def aggregate(delta_literals: torch.Tensor, prop_index: torch.Tensor, method: str, **kwargs) -> torch.Tensor:
    if method == 'mean':
        return aggregate_mean(delta_literals, prop_index, **kwargs)
    elif method == 'max':
        return aggregate_max(delta_literals, prop_index, **kwargs)
    elif method == 'min':
        return aggregate_min(delta_literals, prop_index, **kwargs)


def tnorm_constructor(tnorm: str, aggregate_func):
    if tnorm == 'godel':
        return SATGodel(aggregate_func )
    elif tnorm == 'lukasiewicz':
        return SATLukasiewicz(aggregate_func)
    elif tnorm == 'product':
        return SATProduct(aggregate_func)
    else:
        raise ValueError("Unknown tnorm: {}".format(tnorm))

class SATTNorm(ABC):

    clause_t: torch.Tensor

    @abstractmethod
    def forward(self, indexed_t: torch.Tensor) -> torch.Tensor:
        pass

    def boost_function(self, delta: torch.Tensor, prop_index: torch.Tensor, sign: torch.Tensor) -> torch.Tensor:
        cond = delta > 0
        require_max = cond.any()
        if require_max:
            boost_max = self.boost_function_max(delta, prop_index, sign)
            if (delta < 0).any():
                boost_min = self.boost_function_min(delta, prop_index, sign)
                # TODO Set deltas to 0 if the delta given is 0.
                return torch.where(cond.unsqueeze(-1), boost_max, boost_min)
            return boost_max
        return self.boost_function_min(delta, prop_index, sign)

    @abstractmethod
    def boost_function_min(self, delta: torch.Tensor, prop_index: torch.Tensor, sign: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def boost_function_max(self, delta: torch.Tensor, prop_index: torch.Tensor, sign: torch.Tensor) -> torch.Tensor:
        pass

    def sgd_function(self, indexed_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        t = self.forward(indexed_t)
        return t, t

class SATGodel(SATTNorm):

    def __init__(self, aggregate_func):
        self.aggregate_func = aggregate_func

    def forward(self, indexed_t: torch.Tensor) -> torch.Tensor:
        self.indexed_t = indexed_t
        self.clause_t_g = torch.max(indexed_t, dim=2)
        self.clause_t = self.clause_t_g[0]
        self.formula_t = torch.min(self.clause_t, dim=1)[0]
        return self.formula_t

    def boost_function_max(self, delta, prop_index, sign) -> torch.Tensor:
        w_min = (self.formula_t + delta).unsqueeze(1)
        # t-norm TBF
        delta_min = (self.clause_t <= w_min) * (w_min - self.clause_t)
        max_indices = self.clause_t_g[1].unsqueeze(-1)

        # The delta applied on the max truth values on each clause
        # Find max inputs for t-conorm TBF. Multiply with sign to deal with negated literals
        delta_max = torch.take_along_dim(sign.unsqueeze(0), max_indices, -1).squeeze() * delta_min

        # Find what propositions the results belong to
        propositions_max_indices = torch.take_along_dim(prop_index.unsqueeze(0), max_indices, -1).squeeze()
        prop_deltas = aggregate(delta_max, propositions_max_indices, self.aggregate_func)
        return prop_deltas

    def boost_function_min(self, delta, prop_index, sign) -> torch.Tensor:
        w_min = (self.formula_t + delta).unsqueeze(1)
        # t-norm TBF
        min_clauses = torch.min(self.clause_t, dim=-1)
        min_clause_literals = self.indexed_t[torch.arange(self.indexed_t.shape[0]), min_clauses[1]]
        delta_literals = torch.where(min_clause_literals > w_min,
                                     w_min - min_clause_literals,
                                     torch.zeros_like(min_clause_literals)) * sign[min_clauses[1]]
        prop_index_max = prop_index[min_clauses[1]]
        prop_deltas = aggregate(delta_literals, prop_index_max, self.aggregate_func)
        return prop_deltas



class SATLukasiewicz(SATTNorm):

    def __init__(self, aggregate_func):
        self.aggregate_func = aggregate_func

    def forward(self, indexed_t: torch.Tensor) -> torch.Tensor:
        self.indexed_t = indexed_t
        self.clause_t = torch.clip(torch.sum(indexed_t, dim=-1), 0, 1)
        n = self.clause_t.shape[-1]
        self.formula_t = torch.clip(torch.sum(self.clause_t, dim=-1) - (n-1), 0, 1)
        return self.formula_t

    def boost_function_max(self, delta, prop_index, sign) -> torch.Tensor:
        # Compute the t-norm boost function
        w = (self.formula_t + delta).unsqueeze(1)
        n = self.clause_t.shape[-1]
        sorted_clauses = torch.sort(self.clause_t, dim=-1, descending=False)

        # Find the delta_M for each M
        M = torch.arange(1, n+1)
        l = torch.cumsum(sorted_clauses[0], dim=-1)
        delta_M = (w + M - 1 - torch.cumsum(sorted_clauses[0], dim=-1)) / M

        cond = delta_M < 1 - sorted_clauses[0]
        # Do cond - cond (shifted to the left), then take the argmax.
        # This finds the largest index for which the condition is true
        M_star = torch.argmax(cond.float() - torch.roll(cond.float(), -1, -1), dim=-1)
        # If the condition holds for all clauses, then the max index is the last one
        M_star[cond.all(dim=-1)] = n - 1
        delta_M_star = torch.gather(delta_M, -1, M_star.unsqueeze(-1))

        # Assign the respective deltas to the clause truth values
        delta_sorted_clauses = torch.where(cond, delta_M_star, 1 - sorted_clauses[0])
        delta_clauses = scatter_sum(delta_sorted_clauses, sorted_clauses[1], -1)

        # Compute the t-conorm boost function
        clauses_w = self.clause_t + delta_clauses

        if test_correctness:
            new_w = torch.clip(torch.sum(clauses_w, dim=-1) - (n-1), 0, 1)
            assert (clauses_w <= 1.0).all() and (clauses_w >= 0.0).all()
            assert (new_w <= 1.0).all() and (new_w >= 0.0).all()
            assert torch.logical_or(delta < 0, torch.isclose(new_w, w.squeeze(-1), atol=1e-4)).all(), (new_w, w, delta < 0)

        delta_literals = (torch.clip(clauses_w - torch.sum(self.indexed_t, dim=-1), min=0) / 3.0).unsqueeze(-1)

        if test_correctness:
            new_literals = self.indexed_t + delta_literals
            new_w = torch.clip(torch.sum(new_literals, dim=-1), 0, 1)
            assert (new_literals <= 1.0).all() and (new_literals >= 0.0).all()
            assert (new_w <= 1.0).all() and (new_w >= 0.0).all()
            assert torch.logical_or(delta.unsqueeze(-1) < 0, torch.isclose(new_w, clauses_w)).all(), (new_w, clauses_w, delta < 0)

        delta_literals = delta_literals * sign
        # Distribute the deltas over the propositions using the mean aggregation
        prop_index_expand = prop_index.unsqueeze(0).expand(delta_literals.shape).flatten(1, 2)
        delta_literals_flat = delta_literals.flatten(1, 2)
        prop_deltas = aggregate(delta_literals_flat, prop_index_expand, self.aggregate_func)

        return prop_deltas

    def boost_function_min(self, delta, prop_index, sign) -> torch.Tensor:
        # Compute the t-norm boost function
        w = self.formula_t + delta
        n = self.clause_t.shape[-1]
        delta_clauses = torch.clip(w - 1 + n - torch.sum(self.clause_t, dim=-1), max=0) / n
        w_clauses = (self.clause_t + delta_clauses.unsqueeze(-1)).unsqueeze(-1)

        if test_correctness:
            new_w = torch.clip(torch.sum(w_clauses.squeeze(-1), dim=-1) - (n-1), 0, 1)
            assert (w_clauses <= 1.0).all() and (w_clauses >= 0.0).all()
            assert (new_w <= 1.0).all() and (new_w >= 0.0).all()
            assert torch.logical_or(delta > 0, torch.isclose(new_w, w.squeeze(-1), atol=1e-4)).all(), (new_w, w, delta > 0)

        sorted_literals = torch.sort(self.indexed_t, dim=-1, descending=True)
        n_literals = self.indexed_t[0].shape[-1]
        # Find the delta_M for each M
        M = torch.arange(1, n_literals+1).unsqueeze(0).unsqueeze(0)
        literals_sum = torch.cumsum(sorted_literals[0], dim=-1)
        delta_M = (w_clauses - literals_sum) / M

        # delta_M_star = torch.max(delta_M, dim=-1)[0].unsqueeze(-1)
        #
        # delta_literals = torch.where(self.indexed_t < delta_M_star, -delta_M_star, -self.indexed_t) * sign

        cond = delta_M > -sorted_literals[0]
        # Do cond - cond (shifted to the left), then take the argmax.
        # This finds the largest index for which the condition is true
        M_star = torch.argmax(cond.float() - torch.roll(cond.float(), -1, -1), dim=-1)
        # If the condition holds for all literals, then the max index is the last one
        M_star[cond.all(dim=-1)] = n_literals - 1
        delta_M_star = torch.gather(delta_M, -1, M_star.unsqueeze(-1))

        # Assign the respective deltas to the clause truth values
        delta_sorted_literals = torch.where(cond, delta_M_star, -sorted_literals[0])
        delta_literals = scatter_sum(delta_sorted_literals, sorted_literals[1], -1)

        if test_correctness:
            new_literals = self.indexed_t + delta_literals
            new_w = torch.clip(torch.sum(new_literals, dim=-1), 0, 1)
            assert (new_literals <= 1.0).all() and (new_literals >= 0.0).all()
            assert (new_w <= 1.0).all() and (new_w >= 0.0).all()
            assert torch.logical_or(delta.unsqueeze(-1) > 0, torch.isclose(new_w, w_clauses.squeeze(-1))).all(), (new_w, w_clauses.squeeze(-1), delta > 0)

        delta_literals = delta_literals * sign

        # Distribute the deltas over the propositions using the mean aggregation
        prop_index_expand = prop_index.unsqueeze(0).expand(delta_literals.shape).flatten(1, 2)
        delta_literals_flat = delta_literals.flatten(1, 2)
        prop_deltas = aggregate(delta_literals_flat, prop_index_expand, self.aggregate_func)

        return prop_deltas

    def sgd_function(self, indexed_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.clause_t = torch.clip(torch.sum(indexed_t, dim=-1), 0, 1)
        n = self.clause_t.shape[-1]
        s = torch.sum(self.clause_t, dim=-1) - (n - 1)
        formula_t = torch.clip(s, 0, 1)
        return s, formula_t

class SATProduct(SATTNorm):

    def __init__(self, aggregate_func='max'):
        self.aggregate_func = aggregate_func
    """
    We could technically reuse this for other norms with p=1
    """
    def forward(self, indexed_t: torch.Tensor) -> torch.Tensor:
        self.indexed_t = indexed_t
        self.clause_t = 1 - torch.prod(1-indexed_t, dim=-1)
        self.formula_t = torch.prod(self.clause_t, dim=-1)

        return self.formula_t

    def boost_function_max(self, delta, prop_index, sign) -> torch.Tensor:
        # Compute the t-norm boost function
        w = (self.formula_t + delta).unsqueeze(1)
        sorted_clauses = torch.sort(self.clause_t, dim=-1, descending=True)
        n = self.clause_t.shape[-1]
        onez = torch.ones(self.clause_t.shape[0], 1)

        # Compute the truth value lambda for each clause
        # We make the assumption that there is at least one nonzero value, ie N does not contain all integers.
        lamd = torch.pow(
            w / torch.cat([onez, torch.cumprod(sorted_clauses[0][..., :-1], dim=-1)], dim=-1),
            1 / (n - torch.arange(0, n).unsqueeze(0)))

        # OLD CODE: This should compute the same thing but is slower
        # Choose the right value lambda depending on what values exceed it
        cond = lamd < torch.cat([onez, sorted_clauses[0][..., :-1]], dim=-1)
        i_lambda = torch.argmax(cond.float() - torch.roll(cond.float(), -1, -1), dim=-1)
        i_lambda[cond.all(dim=-1)] = n - 1
        # chosen_lambdas_old = torch.gather(lamd, -1, i_lambda.unsqueeze(-1))
        # delta_clauses_old = torch.where(self.clause_t < chosen_lambdas_old, chosen_lambdas_old - self.clause_t,
        #                             torch.tensor(0.0))

        # NEW CODE: Apparently, the smallest lambda finds exactly the same as i_lambda in the previous snippet
        # We don't have a proof for this but it works...
        chosen_lambdas = torch.min(lamd, dim=-1, keepdim=True)
        # if (chosen_lambdas[1].squeeze(-1) != i_lambda).any():
        #     print(chosen_lambdas[1].squeeze(-1), i_lambda)
        delta_clauses = torch.where(self.clause_t < chosen_lambdas[0], chosen_lambdas[0] - self.clause_t, torch.tensor(0.0))

        clauses_w = self.clause_t + delta_clauses

        if test_correctness:
            new_w = torch.prod(clauses_w, dim=-1)
            assert (clauses_w <= 1.0).all() and (clauses_w >= 0.0).all()
            assert (new_w <= 1.0).all() and (new_w >= 0.0).all()
            assert torch.logical_or(delta < 0, torch.isclose(new_w, w.squeeze(-1))).all(), (new_w, delta < 0)

        # TODO: Check that chosen_lambdas[1] == i_lambda

        # Compute the t-conorm boost function
        # Note: This code is equal to that of the Godel!
        max_literals = self.indexed_t.max(dim=-1)
        max_indices = max_literals[1].unsqueeze(-1)
        # Computes the delta for the t-conorm boost function
        # We multiply here with 1 - max_literals so that it is excluded from the division
        # TODO: What if the divisor approaches 0 if the clause is almost satisfied?
        delta_divisor = (1 - max_literals[0]) * (1-clauses_w) / (torch.prod(1 - self.indexed_t, dim=-1) + 1e-10)
        delta_max_literal = (1 - max_literals[0] - delta_divisor)

        if test_correctness:
            new_literals = self.indexed_t.clone()
            mask = torch.arange(3).reshape(1, 1, 3) == max_indices
            new_literals[mask] = new_literals[mask] + delta_max_literal.flatten()
            new_w = 1 - torch.prod(1 - new_literals, dim=-1)
            assert (new_literals <= 1.0).all() and (new_literals >= 0.0).all()
            assert (new_w <= 1.0).all() and (new_w >= 0.0).all()
            assert torch.logical_or(delta.unsqueeze(-1) < 0, torch.isclose(new_w, clauses_w)).all(), (new_w, clauses_w, delta < 0)

        delta_max_prop = delta_max_literal * torch.take_along_dim(sign.unsqueeze(0), max_indices, -1).squeeze()
        # Find what propositions the results belong to
        propositions_max_indices = torch.take_along_dim(prop_index.unsqueeze(0), max_indices, -1).squeeze()

        prop_deltas = aggregate(delta_max_prop, propositions_max_indices, self.aggregate_func)

        return prop_deltas

    def boost_function_min(self, delta, prop_index, sign) -> torch.Tensor:
        # Compute the t-norm boost function
        w = (self.formula_t + delta)

        min_clause = torch.min(self.clause_t, dim=-1)
        w_min_clause = min_clause[0] * w / self.formula_t

        if test_correctness:
            new_w = w_min_clause * self.formula_t / min_clause[0]
            assert torch.logical_or(delta > 0, w_min_clause <= 1.0).all() and (w_min_clause >= 0.0).all()
            assert (new_w <= 1.0).all() and (new_w >= 0.0).all()
            assert torch.logical_or(delta > 0, torch.isclose(new_w, w.squeeze(-1))).all(), (new_w, delta > 0)

        # Compute the t-conorm boost function
        min_clause_literals = self.indexed_t[range(self.indexed_t.shape[0]), min_clause[1]]
        sorted_literals = torch.sort(min_clause_literals, dim=-1, descending=False)[0]
        n = sorted_literals.shape[-1]
        onez = torch.ones(sorted_literals.shape[0], 1)

        # Compute the truth value lambda for each clause
        # We make the assumption that there is at least one nonzero value, ie N does not contain all integers.
        lamd = 1. - torch.pow(
            (1 - w_min_clause.unsqueeze(-1)) / torch.cat([onez, torch.cumprod(1.0 - sorted_literals[..., :-1], dim=-1)], dim=-1),
            1 / (n - torch.arange(0, n).unsqueeze(0)))

        # OLD CODE: This should compute the same thing but is slower
        # # Choose the right value lambda depending on what values exceed it
        # cond = lamd < torch.cat([onez, sorted_clauses[0]], dim=-1)
        # i_lambda = torch.argmax(cond.float() - torch.roll(cond.float(), -1, -1), dim=-1)
        # i_lambda[cond.all(dim=-1)] = n - 1
        # chosen_lambdas = torch.gather(lamd, -1, i_lambda.unsqueeze(-1))
        # # We choose cond[..., 1:] because the first value is always 1
        # delta_sorted_clauses = torch.where(cond[..., 1:], torch.tensor(0.0), chosen_lambdas - sorted_clauses[0])
        # delta_clauses = torch.scatter_reduce(delta_sorted_clauses, -1, sorted_clauses[1], 'sum')

        # NEW CODE: Apparently, the smallest lambda finds exactly the same as i_lambda in the previous snippet
        # We don't have a proof for this but it works...
        chosen_lambdas = torch.max(lamd, dim=-1, keepdim=True)[0]
        delta_min_clause_literals = torch.where(min_clause_literals > chosen_lambdas,
                                            chosen_lambdas - min_clause_literals,
                                            torch.tensor(0.0))

        if test_correctness:
            new_literals = min_clause_literals + delta_min_clause_literals
            new_w = 1 - torch.prod(1 - new_literals, dim=-1)
            assert (new_literals <= 1.0).all() and (new_literals >= 0.0).all()
            assert (new_w <= 1.0).all() and (new_w >= 0.0).all()
            assert torch.logical_or(delta > 0, torch.isclose(new_w, w_min_clause)).all(), (new_w, w_min_clause, delta > 0)


        prop_index_max = prop_index[min_clause[1]]
        prop_deltas = aggregate(delta_min_clause_literals * sign[min_clause[1]], prop_index_max, self.aggregate_func)

        return prop_deltas

    def sgd_function(self, indexed_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Compute the t-norm boost function
        self.clause_t = 1 - torch.prod(1 - indexed_t, dim=-1)
        self.formula_t = torch.prod(self.clause_t, dim=-1)
        log_s = torch.log(self.clause_t).sum(dim=-1)
        return log_s, self.formula_t


class SATFormula(Formula):
    def __init__(self, clauses, device):
        super().__init__([])
        self.clause_t = None
        self.is_sgd = None
        formula_tensor = torch.tensor(clauses, device=device)
        self.prop_index = formula_tensor.abs() - 1
        # self.prop_index = formula_tensor.abs()
        self.sign = formula_tensor.sign()
        self.tnorm = None

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

