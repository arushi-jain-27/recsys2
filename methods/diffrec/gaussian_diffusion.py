import math
import enum

import numpy as np
import torch as th
import torch.nn as nn


class ModelMeanType(enum.Enum):
    START_X = enum.auto()
    EPSILON = enum.auto()


class GaussianDiffusion(nn.Module):
    def __init__(self, mean_type, noise_schedule, noise_scale, noise_min, noise_max,
                 steps, device, history_num_per_term=10, beta_fixed=True):
        self.mean_type = mean_type
        self.noise_schedule = noise_schedule
        self.noise_scale = noise_scale
        self.noise_min = noise_min
        self.noise_max = noise_max
        self.steps = steps
        self.device = device
        self.history_num_per_term = history_num_per_term
        self.Lt_history = th.zeros(steps, history_num_per_term, dtype=th.float64).to(device)
        self.Lt_count = th.zeros(steps, dtype=int).to(device)

        if noise_scale != 0.:
            self.betas = th.tensor(self.get_betas(), dtype=th.float64).to(self.device)
            if beta_fixed:
                self.betas[0] = 0.00001
            self.calculate_for_diffusion()
        super().__init__()

    def get_betas(self):
        start = self.noise_scale * self.noise_min
        end = self.noise_scale * self.noise_max
        if self.noise_schedule == "linear":
            return np.linspace(start, end, self.steps, dtype=np.float64)
        return betas_from_linear_variance(
            self.steps, np.linspace(start, end, self.steps, dtype=np.float64)
        )

    def calculate_for_diffusion(self):
        alphas = 1.0 - self.betas
        self.alphas_cumprod = th.cumprod(alphas, axis=0).to(self.device)
        self.alphas_cumprod_prev = th.cat(
            [th.tensor([1.0]).to(self.device), self.alphas_cumprod[:-1]]
        )
        self.sqrt_alphas_cumprod = th.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = th.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = th.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = th.sqrt(1.0 / self.alphas_cumprod - 1)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = th.log(
            th.cat([self.posterior_variance[1].unsqueeze(0), self.posterior_variance[1:]])
        )
        self.posterior_mean_coef1 = (
            self.betas * th.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * th.sqrt(alphas) / (1.0 - self.alphas_cumprod)
        )

    def p_sample(self, model, x_start, steps, sampling_noise=False):
        if steps == 0:
            x_t = x_start
        else:
            t = th.tensor([steps - 1] * x_start.shape[0]).to(x_start.device)
            x_t = self.q_sample(x_start, t)
        for i in list(range(self.steps))[::-1]:
            t = th.tensor([i] * x_t.shape[0]).to(x_start.device)
            out = self.p_mean_variance(model, x_t, t)
            if sampling_noise:
                noise = th.randn_like(x_t)
                nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
                x_t = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
            else:
                x_t = out["mean"]
        return x_t

    def training_losses(self, model, x_start, reweight=False):
        batch_size, device = x_start.size(0), x_start.device
        ts, pt = self.sample_timesteps(batch_size, device, "importance")
        noise = th.randn_like(x_start)
        x_t = self.q_sample(x_start, ts, noise) if self.noise_scale != 0. else x_start
        model_output = model(x_t, ts)
        target = x_start if self.mean_type == ModelMeanType.START_X else noise
        mse = mean_flat((target - model_output) ** 2)

        if reweight and self.mean_type == ModelMeanType.START_X:
            weight = self.SNR(ts - 1) - self.SNR(ts)
            weight = th.where((ts == 0), 1.0, weight)
            loss = mse
        elif reweight and self.mean_type == ModelMeanType.EPSILON:
            weight = (1 - self.alphas_cumprod[ts]) / (
                (1 - self.alphas_cumprod_prev[ts]) ** 2 * (1 - self.betas[ts])
            )
            weight = th.where((ts == 0), 1.0, weight)
            likelihood = mean_flat(
                (x_start - self._predict_xstart_from_eps(x_t, ts, model_output)) ** 2 / 2.0
            )
            loss = th.where((ts == 0), likelihood, mse)
        else:
            weight = th.ones(len(target), device=device)
            loss = mse

        terms = {"loss": weight * loss}
        for t, tloss in zip(ts, terms["loss"]):
            if self.Lt_count[t] == self.history_num_per_term:
                self.Lt_history[t, :-1] = self.Lt_history[t, 1:].clone()
                self.Lt_history[t, -1] = tloss.detach()
            else:
                self.Lt_history[t, self.Lt_count[t]] = tloss.detach()
                self.Lt_count[t] += 1
        terms["loss"] /= pt
        return terms

    def sample_timesteps(self, batch_size, device, method="uniform", uniform_prob=0.001):
        if method == "importance":
            if not (self.Lt_count == self.history_num_per_term).all():
                return self.sample_timesteps(batch_size, device, method="uniform")
            Lt_sqrt = th.sqrt(th.mean(self.Lt_history ** 2, axis=-1))
            pt_all = Lt_sqrt / th.sum(Lt_sqrt)
            pt_all = pt_all * (1 - uniform_prob) + uniform_prob / len(pt_all)
            t = th.multinomial(pt_all, num_samples=batch_size, replacement=True)
            pt = pt_all.gather(dim=0, index=t) * len(pt_all)
            return t, pt
        t = th.randint(0, self.steps, (batch_size,), device=device).long()
        return t, th.ones_like(t).float()

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = th.randn_like(x_start)
        return (
            _extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        posterior_mean = (
            _extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        return (
            posterior_mean,
            _extract(self.posterior_variance, t, x_t.shape),
            _extract(self.posterior_log_variance_clipped, t, x_t.shape),
        )

    def p_mean_variance(self, model, x, t):
        model_output = model(x, t)
        if self.mean_type == ModelMeanType.START_X:
            pred_xstart = model_output
        else:
            pred_xstart = self._predict_xstart_from_eps(x, t, model_output)
        model_mean, _, _ = self.q_posterior_mean_variance(pred_xstart, x, t)
        return {
            "mean": model_mean,
            "log_variance": _extract(self.posterior_log_variance_clipped, t, x.shape),
            "pred_xstart": pred_xstart,
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        return (
            _extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def SNR(self, t):
        return self.alphas_cumprod[t] / (1 - self.alphas_cumprod[t])


def betas_from_linear_variance(steps, variance, max_beta=0.999):
    alpha_bar = 1 - variance
    betas = [1 - alpha_bar[0]]
    for i in range(1, steps):
        betas.append(min(1 - alpha_bar[i] / alpha_bar[i - 1], max_beta))
    return np.array(betas)


def _extract(arr, timesteps, broadcast_shape):
    res = arr.to(timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


def mean_flat(tensor):
    return tensor.mean(dim=list(range(1, len(tensor.shape))))
