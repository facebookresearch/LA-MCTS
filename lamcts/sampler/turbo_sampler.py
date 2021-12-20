# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import math
import random
from typing import Optional, Tuple, Dict

import gpytorch
import numpy as np
import torch
from gpytorch.constraints.constraints import Interval
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.models import ExactGP
from sklearn.preprocessing import StandardScaler

from ..func import Func, FuncStats
from ..node import Path
from ..sampler.random_sampler import RandomSampler
from ..type import Bag
from ..utils import get_logger, bo_acquisition, confidence_bounds

logger = get_logger("turbo")


# GP Model
class GP(ExactGP):
    def __init__(self, train_x, train_y, likelihood, lengthscale_constraint, outputscale_constraint, ard_dims, nu):
        super(GP, self).__init__(train_x, train_y, likelihood)
        self.ard_dims = ard_dims
        self.mean_module = ConstantMean()
        base_kernel = MaternKernel(lengthscale_constraint=lengthscale_constraint, ard_num_dims=ard_dims, nu=nu)
        self.covar_module = ScaleKernel(base_kernel, outputscale_constraint=outputscale_constraint)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class TuRBOSampler(RandomSampler):
    def __init__(
            self,
            func: Func,
            func_stats: FuncStats,
            acquisition: str = "ei",
            nu: float = 1.5,
            gp_max_samples: int = 0,
            gp_num_cands: int = 0,
            gp_training_steps: int = 50,
            gp_max_cholesky_size: int = 2000,
            batch_size: int = 5,
            init_tr_length: float = 0.8,
            min_tr_length: float = 0.5 ** 7,
            max_tr_length: float = 1.6,
            tr_length_multiplier: float = 2.0,
            fail_threshold: int = 3,
            succ_threshold: int = 3,
            device: str = "cuda"):
        """
        TuRBO implementation

        :param func: function to be optmized
        :param func_stats: function stats tracker
        :param acquisition: acquisition function, ei (expected improvement), pi (probability of improvement) or cb
        :param nu: smoothness of the learned function for Maven kernel
        :param gp_max_samples: maximum number of samples for GP, 0 if no limit
        :param gp_num_cands: Gaussian process number of candidates
        :param gp_training_steps: Gaussian process training steps
        :param gp_max_cholesky_size: Gaussian process max size to switch to Cholesky
        :param gp_use_ard: Gaussian process use ard or not
        :param batch_size: number of points in each batch
        :param init_tr_length: True region initial length
        :param min_tr_length: True region minimum length
        :param max_tr_length: True region maximum length
        :param tr_length_multiplier: True region shrink or expand multiplier
        :param fail_threshold: threshold for failures of finding better solution
        :param succ_threshold: threshold for successes of finding better solution
        :param device: 'cpu' or 'cuda:x'
        """
        super().__init__(func, func_stats)
        self._acquisition = acquisition
        self._nu = nu
        self._gp_max_samples = gp_max_samples
        self._gp_num_cands = gp_num_cands if gp_num_cands > 0 else min(100 * self._func.dims, 5000)
        self._gp_training_steps = gp_training_steps
        self._gp_max_cholesky_size = gp_max_cholesky_size
        self._batch_size = batch_size
        gpu_count = torch.cuda.device_count()
        device = "cpu" if device == "cpu" or gpu_count == 0 else f"cuda:{random.randrange(0, gpu_count)}"
        self._device = torch.device(device)

        self._init_tr_length = init_tr_length
        self._min_tr_length = min_tr_length
        self._max_tr_length = max_tr_length
        self._tr_length_multiplier = tr_length_multiplier

        self._fail_threshold = fail_threshold
        self._succ_threshold = succ_threshold

    def _train_gp(self, train_x: torch.Tensor, train_y: torch.Tensor, hypers=None):
        """Fit a GP model where train_x is in [0, 1]^d and train_y is standardized."""

        lengthscale_constraint = Interval(0.005, 2.0)
        outputscale_constraint = Interval(0.05, 20.0)

        # Create models
        # likelihood = GaussianLikelihood(noise_constraint=noise_constraint).to(
        #     device=train_x.device, dtype=train_y.dtype)
        likelihood = GaussianLikelihood().to(device=train_x.device, dtype=train_y.dtype)
        model = GP(
            train_x=train_x,
            train_y=train_y,
            likelihood=likelihood,
            lengthscale_constraint=lengthscale_constraint,
            outputscale_constraint=outputscale_constraint,
            ard_dims=self._func.dims,
            nu=self._nu
        ).to(device=train_x.device, dtype=train_x.dtype)

        # Find optimal model hyperparameters
        model.train()
        likelihood.train()

        # "Loss" for GPs - the marginal log likelihood
        mll = ExactMarginalLogLikelihood(likelihood, model)

        # Initialize model hypers
        if hypers:
            model.load_state_dict(hypers)
        else:
            hypers = {}
            hypers["covar_module.outputscale"] = 1.0
            hypers["covar_module.base_kernel.lengthscale"] = 0.5
            hypers["likelihood.noise"] = 0.005
            model.initialize(**hypers)

        # Use the adam optimizer
        optimizer = torch.optim.Adam([{"params": model.parameters()}], lr=0.1)

        for _ in range(self._gp_training_steps):
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()

        # Switch to eval mode
        model.eval()
        likelihood.eval()

        return model

    def _gp_sample(self, bag: Bag, tr_length: float, hypers=None) -> \
            Tuple[Optional[np.ndarray], Optional[Dict]]:
        if len(bag) > self._gp_max_samples > 0:
            picks = np.argsort(bag.fxs)
            if self._func.is_minimizing:
                picks = picks[:self._gp_max_samples]
            else:
                picks = picks[-self._gp_max_samples:]
            xs = bag.xs[picks]
            fxs = bag.fxs[picks]
        else:
            xs = bag.xs
            fxs = bag.fxs
        x_scaler = StandardScaler()
        x_scaler.fit(xs)
        y_scaler = StandardScaler()
        fxs = y_scaler.fit_transform(fxs.reshape((-1, 1))).reshape((-1,))

        # We use CG + Lanczos for training if we have enough data
        with gpytorch.settings.max_cholesky_size(self._gp_max_cholesky_size):
            try:
                gp = self._train_gp(
                    train_x=torch.tensor(x_scaler.transform(xs), device=self._device, dtype=torch.float32),
                    train_y=torch.tensor(fxs, device=self._device, dtype=torch.float32),
                    hypers=hypers)
                hypers = gp.state_dict()
            except TimeoutError as err:
                raise err
            except Exception:
                return None, hypers

        # Create the trust region boundaries
        center = xs.mean(axis=0)
        xs_lb, xs_ub = confidence_bounds(xs)
        weights = gp.covar_module.base_kernel.lengthscale.cpu().detach().numpy().ravel()
        weights = weights / weights.mean()  # This will make the next line more stable
        weights = weights / np.prod(np.power(weights, 1.0 / len(weights)))  # We now have weights.prod() = 1
        radius = weights * tr_length * (xs_ub - xs_lb) / 2.0
        lb = np.maximum(center - radius, xs_lb)
        ub = np.minimum(center + radius, xs_ub)
        x_cand = self._func.gen_random_inputs(self._gp_num_cands, lb, ub)
        if len(x_cand) == 0:
            return None, hypers

        # We use Lanczos for sampling if we have enough data
        with torch.no_grad(), gpytorch.settings.max_cholesky_size(self._gp_max_cholesky_size):
            try:
                y_pred = gp(torch.tensor(x_scaler.transform(x_cand), device=self._device, dtype=torch.float32))
                y_pred_mean = y_pred.mean.cpu().detach().numpy()
                y_pred_std = y_pred.stddev.cpu().detach().numpy()
            except TimeoutError as err:
                raise err
            except Exception:
                return None, hypers

        # De-standardize the sampled values
        y_pred_std[y_pred_std == 0.0] = 1e-10

        af = bo_acquisition(fxs, y_pred_mean, y_pred_std, self._acquisition, self._func.is_minimizing)
        choices = np.argsort(af)[-self._batch_size:]
        return x_cand[choices], hypers

    def sample(self, num_samples: int, path: Optional[Path] = None, **kwargs) -> Bag:
        samples = self._func.gen_sample_bag()
        if path is not None and len(path) > 0:
            gp_bag = path[-1].bag.clone()
        else:
            return super().sample(num_samples, path, **kwargs)
        call_budget = kwargs["call_budget"] if "call_budget" in kwargs else float('inf')
        hypers = None
        while (len(samples) < num_samples and
               self._func_stats.stats.total_calls <= call_budget):
            fail_th = (self._fail_threshold if self._fail_threshold > 0
                       else int(math.ceil(max(4.0 / self._batch_size, self._func.dims / self._batch_size))))
            succ_th = self._succ_threshold
            fail_count = 0
            succ_count = 0
            tr_length = self._init_tr_length
            curr_best = gp_bag.best.clone()

            old_sample_size = len(samples)
            while (len(samples) < num_samples and
                   self._func_stats.stats.total_calls <= call_budget and
                   tr_length >= self._min_tr_length):
                xs_cand, hypers = self._gp_sample(gp_bag, tr_length, hypers)
                if xs_cand is None or len(xs_cand) == 0:
                    break
                next_bag = self._func.gen_sample_bag(xs_cand)
                next_best = next_bag.best
                gp_bag.extend(next_bag)
                samples.extend(next_bag)

                # Update trust region
                if self._func.compare(next_best.fx, curr_best.fx, math.fabs(1e-3 * curr_best.fx)) > 0:
                    curr_best = next_best.clone()
                    succ_count += 1
                    fail_count = 0
                else:
                    succ_count = 0
                    fail_count += 1

                if succ_count >= succ_th:  # Expand trust region
                    tr_length = min(tr_length * self._tr_length_multiplier, self._max_tr_length)
                    succ_count = 0
                elif fail_count >= fail_th:  # Shrink trust region
                    tr_length = tr_length / self._tr_length_multiplier
                    fail_count = 0

            if len(samples) == old_sample_size:  # no new sample found, stop
                break
        return samples

    def __str__(self) -> str:
        return f"TuRBOSampler"
