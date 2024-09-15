from typing import Mapping, Tuple
import os
import pprint
import string
import random
import datetime
import pathlib
import jax
import haiku as hk
import jax.numpy as jnp
import tqdm
import numpy as np
import tensorflow as tf
# import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import optax
from functools import partial
from dataclasses import asdict

import jaxlinop
from gpjax.gaussian_distribution import GaussianDistribution # maybe from gpjax.distributions import GaussianDistribution

# Disable all GPUs for TensorFlow. Load data using CPU.
tf.config.set_visible_devices([], 'GPU')
AUTOTUNE = tf.data.experimental.AUTOTUNE

from ml_tools.config_utils import setup_config
from ml_tools.state_utils import TrainingState
from ml_tools import state_utils
from ml_tools import writers
from ml_tools import actions

from custom_types import Dataset, Batch, Rng
from model import BiDimensionalAttentionModel, AttentionModel
from process import cosine_schedule, GaussianDiffusion, loss
# from neural_diffusion_processes.gp import predict


from config import Config

print("checkpoint 1")

# all the set up and clean documentation stuff
EXPERIMENT = "regression-Sep06-eval"
EXPERIMENT_NAME = None
DATETIME = datetime.datetime.now().strftime("%b%d_%H%M%S")
HERE = pathlib.Path(__file__).parent
LOG_DIR = 'logs'


def get_experiment_name(config: Config):

    global EXPERIMENT_NAME

    if EXPERIMENT_NAME is None:
        letters = string.ascii_lowercase
        id = ''.join(random.choice(letters) for i in range(4))
        EXPERIMENT_NAME = f"{DATETIME}_{config.dataset}_{id}"
    
    return EXPERIMENT_NAME
    

# used later with ml-tools to log experiment
def get_experiment_dir(config: Config, output: str = "root", exist_ok: bool = True) -> pathlib.Path:
    experiment_name = get_experiment_name(config)
    root = HERE / LOG_DIR / EXPERIMENT / experiment_name

    if output == "root":
        dir_ = root
    elif output == "plots":
        dir_ = root / "plots"

    # ignore tensorboard?    
    elif output == "tensorboard":
        # All tensorboard logs are stored in a single directory
        # Run tensorboard with:
        # tensorboard --logdir logs/{EXPERIMENT-NAME}/tensorboard
        dir_ = HERE / LOG_DIR / EXPERIMENT / "tensorboard" / experiment_name
    else:
        raise ValueError("Unknown output: %s" % output)

    dir_.mkdir(parents=True, exist_ok=exist_ok)
    return dir_



def get_data(
    dataset: str,
    input_dim: int = 1,
    train: bool = True,
    batch_size: int = 1024, # number of samples per batch
    num_epochs: int = 1,
) -> Dataset:
    task = "training" if train else "interpolation"    
    data = np.load(f"/data/localhost/not-backed-up/beriegler/projects/NDP/experiments/regression/data/{dataset}_{input_dim}_{task}.npz")
    ds = tf.data.Dataset.from_tensor_slices({
        "x_target": data["x_target"].astype(np.float32),
        "y_target": data["y_target"].astype(np.float32),
        "x_context": data["x_context"].astype(np.float32),
        "y_context": data["y_context"].astype(np.float32),
        "mask_context": data["mask_context"].astype(np.float32),
        # "mask_target": data["mask_target"].astype(np.float32),
    })
    if train:
        ds = ds.repeat(count=num_epochs) # repeat data so it can be passed over num_epoch times in training
        ds = ds.shuffle(seed=42, buffer_size=1000)
    ds = ds.batch(batch_size, drop_remainder=True) # splits data into batches of size batch_size
    ds = ds.prefetch(AUTOTUNE) # for comp efficiency
    ds = ds.as_numpy_iterator() # making ds usable by JAX which operates on nparrays
    return map(lambda d: Batch(**d), ds) # creates object of class Batch from each "batch" in ds

print("checkpoint 2")

config: Config = setup_config(Config) # initialize an instance of class Config, whose attributes are all the relevant settings to tun stuff
key = jax.random.PRNGKey(config.seed)
beta_t = cosine_schedule(config.diffusion.beta_start, config.diffusion.beta_end, config.diffusion.timesteps)
process = GaussianDiffusion(beta_t)


if config.output_dim == 2:
    @hk.without_apply_rng
    @hk.transform
    def network(t, y, x, mask):
        model = AttentionModel(
            n_layers=config.network.n_layers,
            hidden_dim=config.network.hidden_dim,
            num_heads=config.network.num_heads,
            output_dim=config.output_dim,
            sparse=config.network.sparse_attention,
        )
        return model(x, y, t, mask)
    
elif config.output_dim == 1:
    @hk.without_apply_rng
    @hk.transform # effect: once this is run there are two functions availabel: network.apply() and network.init()
    def network(t, y, x, mask):
        model = BiDimensionalAttentionModel(
            n_layers=config.network.n_layers,
            hidden_dim=config.network.hidden_dim,
            num_heads=config.network.num_heads,
        )
        return model(x, y, t, mask)

# from now on network.apply() and network.init() are available with the specified n_layers...

@jax.jit
def net(params, t, yt, x, mask, *, key):
    del key  # the network is deterministic
    #NOTE: Network awkwardly requires a batch dimension for the inputs
    return network.apply(params, t[None], yt[None], x[None], mask[None])[0] # [None] adds batch dim because haiku needs that, but this is immeadiately removed with [0]

# returns the loss (a numba) calculated from the process.loss fct given models params (and accross batches)
# Note the model defined by network() is available gloabally and used here by net()
def loss_fn(params, batch: Batch, key):
    net_with_params = partial(net, params) # partial returns function net but params already plugged in, note: this is an EpsModel (as required by process.loss)!
    kwargs = dict(num_timesteps=config.diffusion.timesteps, loss_type=config.loss_type)
    return loss(process, net_with_params, batch, key, **kwargs) # the loss from process script


learning_rate_schedule = optax.warmup_cosine_decay_schedule(
    init_value=config.optimizer.init_lr,
    peak_value=config.optimizer.peak_lr,
    warmup_steps=config.steps_per_epoch * config.optimizer.num_warmup_epochs,
    decay_steps=config.steps_per_epoch * config.optimizer.num_decay_epochs,
    end_value=config.optimizer.end_lr,
)

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.scale_by_adam(),
    optax.scale_by_schedule(learning_rate_schedule),
    optax.scale(-1.0),
)

@jax.jit
def init(batch: Batch, key: Rng) -> TrainingState:
    key, init_rng = jax.random.split(key)
    t = 1. * jnp.zeros((batch.x_target.shape[0]))

    # use .init function made available by hk.transform
    initial_params = network.init(
        init_rng, t=t, y=batch.y_target, x=batch.x_target, mask=batch.mask_target
    )
    initial_opt_state = optimizer.init(initial_params)
    return TrainingState(
        params=initial_params,
        params_ema=initial_params,
        opt_state=initial_opt_state,
        key=key,
        step=0,
    )

print("checkpoint 3")

# exponential moving average update
@jax.jit
def ema_update(decay, ema_params, new_params):
    def _ema(ema_params, new_params):
        return decay * ema_params + (1.0 - decay) * new_params
    
    # jax.tree_map recursively applies _ema to corresponding old and new parametrers
    # haiku parameters are stored in a tree like structure making this neceassary
    return jax.tree_map(_ema, ema_params, new_params)


@jax.jit
def update_step(state: TrainingState, batch: Batch) -> Tuple[TrainingState, Mapping]:
    new_key, loss_key = jax.random.split(state.key)

    # get value and gradient of the loss_fn
    loss_and_grad_fn = jax.value_and_grad(loss_fn)
    loss_value, grads = loss_and_grad_fn(state.params, batch, loss_key)

    updates, new_opt_state = optimizer.update(grads, state.opt_state)

    new_params = optax.apply_updates(state.params, updates)

    new_params_ema = ema_update(config.optimizer.ema_rate, state.params_ema, new_params)

    new_state = TrainingState(
        params=new_params,
        params_ema=new_params_ema,
        opt_state=new_opt_state,
        key=new_key,
        step=state.step + 1
    )
    metrics = {
        'loss': loss_value,
        'step': state.step
    }
    return new_state, metrics


@jax.jit
def sample_prior(state: TrainingState, key: Rng):
    x = jnp.linspace(-2, 2, 60)[:, None]
    net_with_params = partial(net, state.params_ema)
    y0 = process.sample(key, x, mask=None, model_fn=net_with_params, output_dim= config.output_dim)
    return x, y0


@jax.jit
def sample_conditional(state: TrainingState, key: Rng):
    x = jnp.linspace(-2, 2, 57)[:, None]
    xc = jnp.array([-1., 0., 1.]).reshape(-1, 1)
    yc = jnp.array([0., -1., 1.]).reshape(-1, 1)
    net_with_params = partial(net, state.params_ema)
    y0 = process.conditional_sample(key, x, mask=None, x_context=xc, y_context=yc, mask_context=None, model_fn=net_with_params)
    return x, y0, xc, yc

print("checkpoint 4")

def plots(state: TrainingState, key: Rng):
    if config.input_dim != 1: return {}  # only plot for 1D inputs
    # prior
    fig_prior, ax = plt.subplots()
    x, y0 = jax.vmap(lambda k: sample_prior(state, k))(jax.random.split(key, 10))
    ax.plot(x[...,0].T, y0[...,0].T, color="C0", alpha=0.5)

    # conditional
    fig_cond, ax = plt.subplots()
    x, y0, xc, yc = jax.vmap(lambda k: sample_conditional(state, k))(jax.random.split(key, 10))
    ax.plot(x[...,0].T, y0[...,0].T, "C0", alpha=0.5)
    ax.plot(xc[...,0].T, yc[...,0].T, "C3o")
    return {"prior": fig_prior, "conditional": fig_cond}

print("checkpoint 5")

batch_init = Batch(
    x_target=jnp.zeros((config.batch_size, 10, config.input_dim)),
    y_target=jnp.zeros((config.batch_size, 10, config.output_dim)),
    x_context=jnp.zeros((config.batch_size, 10, config.input_dim)),
    y_context=jnp.zeros((config.batch_size, 10, config.output_dim)),
    mask_context=jnp.zeros((config.batch_size, 10)),
    mask_target=jnp.zeros((config.batch_size, 10)),
)

print("checkpoint 6")

# initialize model parameters and optimizer

# init needs batch to initialize sensible values for params
state = init(batch_init, jax.random.PRNGKey(config.seed))

experiment_dir_if_exists = pathlib.Path(config.restore)
if (experiment_dir_if_exists / "checkpoints").exists():
    index = state_utils.find_latest_checkpoint_step_index(str(experiment_dir_if_exists))
    if index is not None:
        state = state_utils.load_checkpoint(state, str(experiment_dir_if_exists), step_index=index)
        print("Restored checkpoint at step {}".format(state.step))

# writers from ml-tools
exp_root_dir = get_experiment_dir(config)

# rewrote code to only contain local writer, wet git for original
# local_writer = writers.LocalWriter(str(exp_root_dir), flush_every_n=100)
print("checkpoint 7")

writer = writers.LocalWriter(str(exp_root_dir), flush_every_n=100)

# # maybe ignore tb
# tb_writer = writers.TensorBoardWriter(get_experiment_dir(config, "tensorboard"))

# aim_writer = writers.AimWriter(EXPERIMENT)

# # also here
# writer = writers.MultiWriter([aim_writer, tb_writer, local_writer])
# writer.log_hparams(asdict(config))



# take out wirters you dont want here
actions = [
    actions.PeriodicCallback(
        every_steps=10,
        callback_fn=lambda step, t, **kwargs: writer.write_scalars(step, kwargs["metrics"])
    ),
    actions.PeriodicCallback(
        every_steps=config.total_steps // 8,
        callback_fn=lambda step, t, **kwargs: writer.write_figures(step, plots(kwargs["state"], kwargs["key"]))
    ),
    actions.PeriodicCallback(
        every_steps=config.total_steps // 20,
        callback_fn=lambda step, t, **kwargs: state_utils.save_checkpoint(kwargs["state"], exp_root_dir, step)
    ),
]

ds_train: Dataset = get_data(
    config.dataset,
    input_dim=config.input_dim,
    train=True,
    batch_size=config.batch_size,
    num_epochs=config.num_epochs,
)

print("checkpoint 8")

steps = range(state.step + 1, config.total_steps + 1)

print("checkpoint 9")

progress_bar = tqdm.tqdm(steps)



# the actual training
print("checkpoint 10")

for step, batch in zip(progress_bar, ds_train):
    if step < state.step: continue  # wait for the state to catch up in case of restarts

    state, metrics = update_step(state, batch)
    metrics["lr"] = learning_rate_schedule(state.step)

    for action in actions:
        action(step, t=None, metrics=metrics, state=state, key=key)

    if step % 100 == 0:
        progress_bar.set_description(f"loss {metrics['loss']:.2f}") # the loss on the current batch
        

print("EVALUATION")
if config.eval.float64:
    from jax.config import config as jax_config
    jax_config.update("jax_enable_x64", True)

# get the trained model
net_with_params = partial(net, state.params_ema)

#set number of functions to sample for evaluation metrics
n_samples = config.eval.num_samples

# start evaluation of trained model

@jax.jit
@partial(jax.vmap, in_axes=(0, None, None, None, None))
def sample_n_conditionals(key, x_test, x_context, y_context, mask_context):
    return process.conditional_sample(
        key, x_test, mask=None, x_context=x_context, y_context=y_context, mask_context=mask_context, model_fn=net_with_params)


def plot(batch):
    batch_size = len(batch.x_context)
    n = int(batch_size ** 0.5)
    fig, axes = plt.subplots(n,n,figsize=(5,5), sharex=True, sharey=True)
    axes = np.array(axes).reshape(-1)
    x_test = jnp.linspace(-2, 2, 57)[:, None]
    for i in range(batch_size):
        samples = sample_n_conditionals(
            jax.random.split(jax.random.PRNGKey(42), 8), # 8 functions per plot
            x_test,
            batch.x_context[i],
            batch.y_context[i],
            batch.mask_context[i],
        )
        mean, var = jnp.mean(samples, axis=0).squeeze(axis=1), jnp.var(samples, axis=0).squeeze(axis=1)
        axes[i].plot(x_test, samples[..., 0].T, "C0", lw=1)
        axes[i].plot(x_test, mean, "k")
        axes[i].fill_between(
            x_test.squeeze(axis=1),
            mean - 1.96 * jnp.sqrt(var),
            mean + 1.96 * jnp.sqrt(var),
            color="k",
            alpha=0.1,
        )
        xc = batch.x_context[i] + 1e3 * batch.mask_context[i][:, None]
        axes[i].plot(xc, batch.y_context[i], "C3o")
        axes[i].set_xlim(-2.05, 2.05)
    return fig


@jax.jit
@partial(jax.vmap, in_axes=(None, 0, 0, 0, 0, 0))
def eval_conditional(key, x_test, y_test, x_context, y_context, mask_context):

    samples = sample_n_conditionals(jax.random.split(key, n_samples), x_test, x_context, y_context, mask_context)
    samples = samples.squeeze(axis=-1)
    mean = jnp.mean(samples, axis=0)
    centered_samples = samples - mean
    covariance = jnp.dot(centered_samples.T, centered_samples) / (samples.shape[0] - 1)
    covariance = covariance + jnp.eye(covariance.shape[0]) * 1e-6
    post = GaussianDistribution(
        loc=mean.squeeze(),
        scale=jaxlinop.DenseLinearOperator(covariance),
    )
    ll = post.log_prob(y_test.squeeze()) / len(x_test)
    mse = jnp.mean((post.mean() - y_test.squeeze()) ** 2)
    num_context = len(x_context) - jnp.count_nonzero(mask_context)
    return {"mse": mse, "ll": ll, "nc": num_context}


def summary_stats(metrics):
    err = lambda v: 1.96 * jnp.std(v.reshape(-1)) / jnp.sqrt(len(v.reshape(-1)))
    summary_stats = [("mean", jnp.mean), ("std", jnp.std), ("err", err)]
    metrics = {f"{k}_{n}": s(jnp.stack(v)) for k, v in metrics.items() for n, s in summary_stats}
    return metrics


ds_test = get_data(
    config.dataset,
    input_dim=config.input_dim,
    train=False,
    batch_size=config.eval.batch_size,
    num_epochs=1,
)

metrics = {"mse": [], "ll": [], "nc": []}

from tqdm.contrib import tenumerate

for i, batch in tenumerate(ds_test, total=128 // config.eval.batch_size):
    key, ekey = jax.random.split(key)
    m = eval_conditional(ekey, batch.x_target, batch.y_target, batch.x_context, batch.y_context, batch.mask_context)
    for k, v in m.items():
        metrics[k].append(v)

    summary = summary_stats(metrics)
    summary = {"eval_" + k: v for k, v in summary.items()}
    writer.write_scalars(i, summary)
    if config.input_dim == 1:
        fig = plot(batch)
        writer.write_figures(i, {"eval_conditional_sample": fig})


metrics = summary_stats(metrics)
pprint.pprint(metrics)
