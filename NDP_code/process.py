from typing import Protocol, Tuple

import jax
import jax.numpy as jnp
from check_shapes import check_shapes

from custom_types import Batch, Rng, ndarray


class EpsModel(Protocol):
    @check_shapes("t: []", "yt: [N, y_dim]", "x: [N, x_dim]", "mask: [N,]", "return: [N, y_dim]")
    def __call__(self, t: ndarray, yt: ndarray, x: ndarray, mask: ndarray, *, key: Rng) -> ndarray:
        ...


def expand_to(a, b):
    new_shape = a.shape + (1,) * (b.ndim - a.ndim)
    return a.reshape(new_shape)


def cosine_schedule(beta_start, beta_end, timesteps, s=0.008, **kwargs):
    x = jnp.linspace(0, timesteps, timesteps + 1)
    ft = jnp.cos(((x / timesteps) + s) / (1 + s) * jnp.pi * 0.5) ** 2
    alphas_cumprod = ft / ft[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = jnp.clip(betas, 0.0001, 0.9999)
    betas = (betas - betas.min()) / (betas.max() - betas.min())
    return betas * (beta_end - beta_start) + beta_start


class GaussianDiffusion:
    betas: ndarray
    alphas: ndarray
    alpha_bars: ndarray

    def __init__(self, betas):
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alpha_bars = jnp.cumprod(1.0 - betas)

    @check_shapes("y0: [N, y_dim]", "t: []", "return[0]: [N, y_dim]", "return[1]: [N, y_dim]")
    def pt0(self, y0: ndarray, t: ndarray) -> Tuple[ndarray, ndarray]:
        alpha_bars = expand_to(self.alpha_bars[t], y0)
        m_t0 = jnp.sqrt(alpha_bars) * y0
        v_t0 = (1.0 - alpha_bars) * jnp.ones_like(y0)
        return m_t0, v_t0

    @check_shapes("y0: [N, y_dim]", "t: []", "return[0]: [N, y_dim]", "return[1]: [N, y_dim]")
    def forward(self, key: Rng, y0: ndarray, t: ndarray) -> Tuple[ndarray, ndarray]:
        m_t0, v_t0 = self.pt0(y0, t)
        noise = jax.random.normal(key, y0.shape)
        yt = m_t0 + jnp.sqrt(v_t0) * noise
        return yt, noise

    def ddpm_backward_step(self, key: Rng, noise: ndarray, yt: ndarray, t: ndarray) -> ndarray:
        beta_t = expand_to(self.betas[t], yt)
        alpha_t = expand_to(self.alphas[t], yt)
        alpha_bar_t = expand_to(self.alpha_bars[t], yt)

        z = (t > 0) * jax.random.normal(key, shape=yt.shape, dtype=yt.dtype)

        a = 1.0 / jnp.sqrt(alpha_t)
        b = beta_t / jnp.sqrt(1.0 - alpha_bar_t)
        yt_minus_one = a * (yt - b * noise) + jnp.sqrt(beta_t) * z
        return yt_minus_one

    def ddpm_backward_mean_var(self, noise: ndarray, yt: ndarray, t: ndarray) -> ndarray:
        beta_t = expand_to(self.betas[t], yt)
        alpha_t = expand_to(self.alphas[t], yt)
        alpha_bar_t = expand_to(self.alpha_bars[t], yt)

        a = 1.0 / jnp.sqrt(alpha_t)
        b = beta_t / jnp.sqrt(1.0 - alpha_bar_t)
        m = a * (yt - b * noise)
        v = beta_t * jnp.ones_like(yt) * (t > 0)
        v = jnp.maximum(v, jnp.ones_like(v) * 1e-3)
        return m, v

    def sample(self, key, x, mask, *, model_fn: EpsModel, output_dim: int = 1):
        key, ykey = jax.random.split(key)
        yT = jax.random.normal(ykey, (len(x), output_dim))

        if mask is None:
            mask = jnp.zeros_like(x[:, 0])

        @jax.jit
        def scan_fn(y, inputs):
            t, key = inputs
            mkey, rkey = jax.random.split(key)
            noise_hat = model_fn(t, y, x, mask, key=mkey)
            y = self.ddpm_backward_step(key=rkey, noise=noise_hat, yt=y, t=t)
            return y, None

        ts = jnp.arange(len(self.betas))[::-1]
        keys = jax.random.split(key, len(ts))
        yf, yt = jax.lax.scan(scan_fn, yT, (ts, keys))
        return yt if yt is not None else yf

    def conditional_sample(
        self,
        key,
        x,
        mask,
        *,
        x_context,
        y_context,
        mask_context,
        model_fn: EpsModel,
        num_inner_steps: int = 5,
        method: str = "repaint",
    ):
        if mask is None:
            mask = jnp.zeros_like(x[:, 0])

        if mask_context is None:
            mask_context = jnp.zeros_like(x_context[:, 0])

        key, ykey = jax.random.split(key)
        x_augmented = jnp.concatenate([x_context, x], axis=0)
        mask_augmented = jnp.concatenate([mask_context, mask], axis=0)
        num_context = len(x_context)

        @jax.jit
        def repaint_inner(yt_target, inputs):
            t, key = inputs
            key, fkey, mkey, bkey = jax.random.split(key, 4)
            # one step backward: t -> t-1
            yt_context = self.forward(fkey, y_context, t)[0]
            y_augmented = jnp.concatenate([yt_context, yt_target], axis=0)
            noise_hat = model_fn(t, y_augmented, x_augmented, mask_augmented, key=mkey)
            y = self.ddpm_backward_step(key=bkey, noise=noise_hat, yt=y_augmented, t=t)
            y = y[num_context:]
            # one step forward: t-1 -> t
            z = jax.random.normal(key, shape=y.shape)
            beta__t_minus_1 = expand_to(self.betas[t - 1], y)
            y = jnp.sqrt(1.0 - beta__t_minus_1) * y + jnp.sqrt(beta__t_minus_1) * z
            return y, None

        @jax.jit
        def repaint_outer(y, inputs):
            t, key = inputs
            # loop
            key, ikey = jax.random.split(key)
            ts = jnp.ones((num_inner_steps,), dtype=jnp.int32) * t
            keys = jax.random.split(ikey, num_inner_steps)
            y, _ = jax.lax.scan(repaint_inner, y, (ts, keys))

            # step backward: t -> t-1
            key, fkey, mkey, bkey = jax.random.split(key, 4)
            yt_context = self.forward(fkey, y_context, t)[0]
            y_augmented = jnp.concatenate([yt_context, y], axis=0)
            noise_hat = model_fn(t, y_augmented, x_augmented, mask_augmented, key=mkey)
            y = self.ddpm_backward_step(key=bkey, noise=noise_hat, yt=y_augmented, t=t)
            y = y[num_context:]
            return y, None

        ts = jnp.arange(len(self.betas))[::-1]
        keys = jax.random.split(key, len(ts))
        yT_target = jax.random.normal(ykey, (len(x), y_context.shape[-1])) # this is how it knows what the output dim is

        y, _ = jax.lax.scan(repaint_outer, yT_target, (ts[:-1], keys[:-1]))
        return y


def loss(
    process: GaussianDiffusion,
    network: EpsModel,
    batch: Batch,
    key: Rng,
    *,
    num_timesteps: int,
    loss_type: str = "l1",
):
    if loss_type == "l1":

        def loss_metric(a, b):
            return jnp.abs(a - b)

    elif loss_type == "l2":

        def loss_metric(a, b):
            return (a - b) ** 2

    else:
        raise ValueError(f"Unknown loss type {loss_type}")

    @check_shapes(
        "t: []", "y: [N, y_dim]", "x: [N, x_dim]", "mask: [N,] if mask is not None", "return: []"
    )
    def loss_fn(key, t, y, x, mask):
        yt, noise = process.forward(key, y, t)
        noise_hat = network(t, yt, x, mask, key=key)
        loss_value = jnp.sum(loss_metric(noise, noise_hat), axis=1)  # [N,]
        loss_value = loss_value * (1.0 - mask)
        num_points = len(mask) - jnp.count_nonzero(mask)
        return jnp.sum(loss_value) / num_points

    batch_size = len(batch.x_target)

    key, tkey = jax.random.split(key)
    # Low-discrepancy sampling over t to reduce variance
    t = jax.random.uniform(tkey, (batch_size,), minval=0, maxval=num_timesteps / batch_size)
    t = t + (num_timesteps / batch_size) * jnp.arange(batch_size)
    t = t.astype(jnp.int32)

    keys = jax.random.split(key, batch_size)

    if batch.mask_target is None:
        # consider all points
        mask_target = jnp.zeros_like(batch.x_target[..., 0])
    else:
        mask_target = batch.mask_target

    losses = jax.vmap(loss_fn)(keys, t, batch.y_target, batch.x_target, mask_target)
    return jnp.mean(losses)
