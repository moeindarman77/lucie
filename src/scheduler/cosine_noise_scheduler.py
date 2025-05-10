import torch
import numpy as np


class CosineNoiseScheduler:
    r"""
    Class for the cosine noise scheduler introduced in:
    "Improved Denoising Diffusion Probabilistic Models" (Nichol & Dhariwal, 2021).
    """

    def __init__(self, num_timesteps, s=0.008):
        self.num_timesteps = num_timesteps
        self.s = s

        # Generate alpha_bar (cumulative alpha) using cosine schedule
        timesteps = torch.linspace(0, num_timesteps, num_timesteps + 1, dtype=torch.float64) / num_timesteps
        alpha_bar = torch.cos(((timesteps + s) / (1 + s)) * np.pi * 0.5) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]  # normalize

        # Compute discrete betas from alpha_bar
        self.alpha_cum_prod = alpha_bar[:-1].float()
        self.betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
        self.betas = torch.clamp(self.betas, max=0.999).float()

        # Derived values
        self.alphas = 1. - self.betas
        self.sqrt_alpha_cum_prod = torch.sqrt(self.alpha_cum_prod)
        self.sqrt_one_minus_alpha_cum_prod = torch.sqrt(1 - self.alpha_cum_prod)

    def add_noise(self, original, noise, t):
        r"""
        Forward method for diffusion
        :param original: Clean image tensor (x₀)
        :param noise: Random noise tensor (ϵ)
        :param t: Timestep tensor of shape (B,)
        :return: Noised image xₜ
        """
        original_shape = original.shape
        batch_size = original_shape[0]

        sqrt_alpha_cum_prod = self.sqrt_alpha_cum_prod.to(original.device)[t].reshape(batch_size)
        sqrt_one_minus_alpha_cum_prod = self.sqrt_one_minus_alpha_cum_prod.to(original.device)[t].reshape(batch_size)

        # Reshape (B,) to (B,1,1,1) for broadcasting over image dims
        for _ in range(len(original_shape) - 1):
            sqrt_alpha_cum_prod = sqrt_alpha_cum_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_cum_prod = sqrt_one_minus_alpha_cum_prod.unsqueeze(-1)

        return (
            sqrt_alpha_cum_prod * original +
            sqrt_one_minus_alpha_cum_prod * noise
        )

    def sample_prev_timestep(self, xt, noise_pred, t):
        r"""
        Use model prediction of noise to sample x_{t-1} from x_t
        :param xt: Noisy input at timestep t
        :param noise_pred: Model's noise prediction ϵ_θ(xₜ, t)
        :param t: Current timestep (int or scalar tensor)
        :return: x_{t-1} sample, and x₀ estimate
        """
        device = xt.device
        t = t if isinstance(t, torch.Tensor) else torch.tensor(t, device=device)

        sqrt_alpha_cum_prod_t = self.sqrt_alpha_cum_prod.to(device)[t]
        sqrt_one_minus_alpha_cum_prod_t = self.sqrt_one_minus_alpha_cum_prod.to(device)[t]
        alpha_cum_prod_t = self.alpha_cum_prod.to(device)[t]
        alpha_t = self.alphas.to(device)[t]
        beta_t = self.betas.to(device)[t]

        # Estimate x0 from predicted noise
        x0 = (xt - sqrt_one_minus_alpha_cum_prod_t * noise_pred) / torch.sqrt(alpha_cum_prod_t)

        # Compute mean
        mean = xt - beta_t * noise_pred / sqrt_one_minus_alpha_cum_prod_t
        mean = mean / torch.sqrt(alpha_t)

        if t == 0:
            return mean, x0
        else:
            alpha_cum_prod_prev = self.alpha_cum_prod.to(device)[t - 1]
            variance = (1 - alpha_cum_prod_prev) / (1 - alpha_cum_prod_t)
            variance = variance * beta_t
            sigma = torch.sqrt(variance)
            z = torch.randn_like(xt)
            return mean + sigma * z, x0