from __future__ import annotations
from typing import Tuple, List, Callable, Mapping, Optional

import abc
from dataclasses import dataclass
import gpjax.kernels as jaxkern
import jax
import gpjax
import jax.numpy as jnp
from jaxtyping import Float, Array
import distrax
from gpjax.gps import Prior
import xarray as xr

import numpy as np
import jax

from custom_types import Batch


@dataclass
class UniformDiscrete:
    lower: int
    upper: int

    def sample(self, key, shape):
        if self.lower == self.upper:
            return jnp.ones(shape, dtype=jnp.int32) * self.lower
        return jax.random.randint(key, shape, minval=self.lower, maxval=self.upper + 1)


DATASETS = [
    "se",
    "matern",
    "sawtooth",
    "step",
    "step2D",
    "sin",
    "swirl",
    "era5"
]

TASKS = [
    "training",
    "interpolation",
]


@dataclass
class TaskConfig:
    x_context_dist: distrax.Distribution
    x_target_dist: distrax.Distribution


@dataclass
class DatasetConfig:
    is_gp: bool
    max_input_dim: int  # (incl.)
    min_input_dim: int = 1
    eval_num_target: UniformDiscrete = UniformDiscrete(50, 50)
    eval_num_context: UniformDiscrete = UniformDiscrete(1, 10)
    grid: bool = False
    grid_dim: tuple = (20,20)


_NOISE_VAR = 0.05**2
_KERNEL_VAR = 1.0
_LENGTHSCALE = .25

_DATASET_CONFIGS = {
    "se": DatasetConfig(max_input_dim=3, is_gp=True),

    "matern": DatasetConfig(max_input_dim=3, is_gp=True),

    "sawtooth": DatasetConfig(max_input_dim=1, is_gp=False),

    "step": DatasetConfig(max_input_dim=1, is_gp=False),

    "step2D": DatasetConfig(min_input_dim=2, max_input_dim=2, 
                          eval_num_target = UniformDiscrete(1216, 1216),
                          is_gp=False, grid = True, grid_dim = (32, 38)),

    "sin": DatasetConfig(max_input_dim=1, is_gp=False),

    "swirl": DatasetConfig(min_input_dim=2, max_input_dim=2, 
                           eval_num_target = UniformDiscrete(400, 400), 
                           is_gp=False, grid = True),

    "era5": DatasetConfig(min_input_dim=2, max_input_dim=2, 
                          eval_num_target = UniformDiscrete(1216, 1216),
                          is_gp=False, grid = True, grid_dim = (32, 38)),
    
}

_TASK_CONFIGS = {
    "training": TaskConfig(
        x_context_dist=distrax.Uniform(-2, 2),
        x_target_dist=distrax.Uniform(-2, 2),
    ),
    "interpolation": TaskConfig(
        x_context_dist=distrax.Uniform(-2, 2),
        x_target_dist=distrax.Uniform(-2, 2),
    ),
}


@dataclass
class FuntionalDistribution(abc.ABC):
    # All GP datasets are naturally normalized so do not need additional normalization.
    # Sawtooth is not normalized so we need to normalize it in the Mixture but not when used
    # in isolation.
    is_data_naturally_normalized: bool = True
    normalize: bool = False

    @abc.abstractmethod
    def sample(self, key, x: Float[Array, "N D"]) -> Float[Array, "N 1"]:
        raise NotImplementedError()
        

class GPFunctionalDistribution(FuntionalDistribution):

    def __init__(self, kernel: jaxkern.AbstractKernel, params: Mapping):
        self.kernel = kernel
        self.params = params
        self.mean = gpjax.mean_functions.Zero()
        self.prior = Prior(self.kernel, self.mean)
    
    def sample(self, key, x: Float[Array, "N D"]) -> Float[Array, "N 1"]:
        f = self.prior.predict(self.params)(x).sample(seed=key, sample_shape=()).reshape(x.shape[0], 1)
        sigma2 = self.params["noise_variance"]
        y = f + (jax.random.normal(key, shape=f.shape) * jnp.sqrt(sigma2))
        return y


DatasetFactory: Callable[[List[int]], FuntionalDistribution]

_DATASET_FACTORIES: Mapping[str, DatasetFactory] = {}

# this wrapper adds a factory to the dictionary _DATASET_FACTORIES
def register_dataset_factory(name: str):

    def wrap(f: DatasetFactory):
        _DATASET_FACTORIES[name] = f
    
    return wrap

    
@register_dataset_factory("se")
def _se_dataset_factory(active_dim: List[int]):
    k = jaxkern.RBF(active_dims=active_dim)
    input_dim = len(active_dim)
    factor = jnp.sqrt(input_dim)
    params = {
        "mean_function": {},
        "kernel": {"lengthscale": _LENGTHSCALE * factor, "variance": _KERNEL_VAR,},
        "noise_variance": _NOISE_VAR
    }
    return GPFunctionalDistribution(k, params)


@register_dataset_factory("matern")
def _matern_dataset_factory(active_dim: List[int]):
    k = jaxkern.Matern52(active_dims=active_dim)
    input_dim = len(active_dim)
    factor = jnp.sqrt(input_dim)
    params = {
        "mean_function": {},
        "kernel": {"lengthscale": _LENGTHSCALE * factor, "variance": _KERNEL_VAR,},
        "noise_variance": _NOISE_VAR
    }
    return GPFunctionalDistribution(k, params)


class Sawtooth(FuntionalDistribution):

    A = 1.
    K_max = 20
    mean = 0.5
    variance = 0.07965

    """ See appendix H: https://arxiv.org/pdf/2007.01332.pdf"""
    def sample(self, key, x: Float[Array, "N 1"]) -> Float[Array, "N 1"]:
        fkey, skey, kkey = jax.random.split(key, 3)
        f = jax.random.uniform(fkey, (), minval=3., maxval=5.)
        s = jax.random.uniform(skey, (), minval=-5., maxval=5.)
        ks = jnp.arange(1, self.K_max + 1, dtype=x.dtype)[None, :]
        vals = (-1.) ** ks * jnp.sin(2. * jnp.pi * ks * f * (x - s)) / ks
        k = jax.random.randint(kkey, (), minval=10, maxval=self.K_max + 1)
        mask = jnp.where(ks < k, jnp.ones_like(ks), jnp.zeros_like(ks))
        # we substract the mean A/2
        fs = self.A/2 + self.A/jnp.pi * jnp.sum(vals * mask, axis=1, keepdims=True)
        fs = fs - self.mean
        if self.normalize:
            fs = fs / jnp.sqrt(self.variance)
        return fs


@register_dataset_factory("sawtooth")
def _sawtooth_dataset_factory(*_):
    return Sawtooth(is_data_naturally_normalized=False, normalize=False)


class Step(FuntionalDistribution):

    """ See appendix H: https://arxiv.org/pdf/2007.01332.pdf"""
    def sample(self, key, x: Float[Array, "N 1"]) -> Float[Array, "N 1"]:
        s = jax.random.uniform(key, (), minval=-2., maxval=2.)
        fs = jnp.where(x < s, jnp.zeros_like(x), jnp.ones_like(x))
        return fs


@register_dataset_factory("step")
def _sawtooth_dataset_factory(*_):
    return Step()


class Step2D(FuntionalDistribution):

    def sample(self, key, x: Float[Array, "N 2"]) -> Float[Array, "N 1"]:
        s = jax.random.uniform(key, (), minval=-2., maxval=2.)

        fs = jnp.where(x[:, 0] < s, jnp.zeros_like(x[:, 0]), jnp.ones_like(x[:, 0]))
        return jnp.reshape(fs, (-1,1))


@register_dataset_factory("step2D")
def _sawtooth_dataset_factory(*_):
    return Step2D()


class SinDist(FuntionalDistribution):

    """ Sin with random period"""
    def sample(self, key, x: Float[Array, "N 1"]) -> Float[Array, "N 1"]:

        akey, bkey = jax.random.split(key, 2)
        a = jax.random.uniform(akey, (), minval=1., maxval=6.)
        # b = jax.random.uniform(bkey, (), minval=-2., maxval=2.)

        fab = jnp.sin(a*x)
        return fab


@register_dataset_factory("sin")
def _sawtooth_dataset_factory(*_):
    return SinDist(is_data_naturally_normalized=False, normalize=False)



class SwirlDist(FuntionalDistribution):

    """ inward vector field with random center, fixed input locations"""
    def sample(self, key, x: Float[Array, "N 2"] = None) -> Float[Array, "N 2"]:

        akey, bkey, ckey, dkey = jax.random.split(key, 4)
        a = jax.random.uniform(akey, (), minval=-1.5, maxval=1.5)
        b = jax.random.uniform(bkey, (), minval=-1.5, maxval=1.5)
        c = jax.random.uniform(bkey, (), minval=0, maxval=2)
        d = jax.random.choice(dkey, jnp.array([1., -1.]))

        

        x1 = x[:,0]
        x2 = x[:,1]

        u = d*x2 - c*x1 -b
        v = -d*x1- c*x2 -a

        u = jnp.reshape(u,(-1,1))
        v = jnp.reshape(v,(-1,1))
        fab = jnp.column_stack((u,v))

        return fab


@register_dataset_factory("swirl")
def _sawtooth_dataset_factory(*_):
    return SwirlDist(is_data_naturally_normalized=False, normalize=False)

@dataclass
class era5Dist(FuntionalDistribution):
    data = np.load("sim2real_data/era5/era5.npy")

    def sample(self, key, x: Float[Array, "N 2"]) -> Float[Array, "N 1"]: 
    
        t = jax.random.randint(key, shape=(), minval=0, maxval=52583)

        ft = jnp.array(self.data)[t]

        mu = jnp.mean(ft)

        std = jnp.std(ft)

        ft = (ft - mu) / std

        return ft 
   
@register_dataset_factory("era5")
def _sawtooth_dataset_factory(*_):

    return era5Dist(is_data_naturally_normalized=False, normalize=True)

# at this point _DATASET_FACTORIES contains all the data set factories defined above

# get batch will return an object of class Batch, defined in ndp.types
def get_batch(key, batch_size: int, name: str, task: str, input_dim: int):
    if name not in DATASETS: # list of se, mattern,...
        raise NotImplementedError("Unknown dataset: %s." % name)
    if task not in TASKS: # train or interpolate
        raise NotImplementedError("Unknown task: %s." % task)
    
    
    if input_dim > _DATASET_CONFIGS[name].max_input_dim:
        raise NotImplementedError(
            "Too many active dims for dataset %s. Max: %d, got: %d." % (
                name, _DATASET_CONFIGS[name].max_input_dim, len(active_dims)
            )
        )

    if task == "training":
        min_n_target = _DATASET_CONFIGS[name].eval_num_target.lower

        # for grid data need to fix number of inputs
        if not _DATASET_CONFIGS[name].grid:
            max_n_target = (
                _DATASET_CONFIGS[name].eval_num_target.upper
                + _DATASET_CONFIGS[name].eval_num_context.upper * input_dim
            )  # input_dim * num_context + num_target
        
        else:
            max_n_target = _DATASET_CONFIGS[name].eval_num_target.upper

        max_n_context = 0

    else:
        max_n_target = _DATASET_CONFIGS[name].eval_num_target.upper

        # for grid data need to fix number of inputs
        if not _DATASET_CONFIGS[name].grid:
            max_n_context = _DATASET_CONFIGS[name].eval_num_context.upper * input_dim
        
        else:
            max_n_context = _DATASET_CONFIGS[name].eval_num_context.upper

    key, ckey, tkey, mkey = jax.random.split(key, 4)

    if not _DATASET_CONFIGS[name].grid:

        x_context = _TASK_CONFIGS[task].x_context_dist.sample(seed=ckey, sample_shape=(batch_size, max_n_context, input_dim)) #[B, Nc, Dx]

        x_target = _TASK_CONFIGS[task].x_target_dist.sample(seed=tkey, sample_shape=(batch_size, max_n_target, input_dim)) #[B, Nt, Dx]
        
        # locations to sample at
        x = jnp.concatenate([x_context, x_target], axis=1) #[B, Nc+Nt, Dx]

    else:
        
        dim = _DATASET_CONFIGS[name].grid_dim

        lon = jnp.linspace(-2,2,dim[1])
        lat = jnp.linspace(-2,2,dim[0])

        lon, lat = jnp.meshgrid(lon, lat)
        lat = -lat

        lon = jnp.reshape(lon, (-1,1))
        lat = jnp.reshape(lat, (-1,1))

        x = jnp.column_stack((lon,lat))

        # ctxt_list = []
        # trgt_list = []

        # for _ in range(batch_size):
            
        #     x = jax.random.permutation(ckey, x)
              
        #     ctxt = x[:max_n_context]
        #     trgt = x[max_n_context:]    
            
        #     ctxt_list.append(ctxt)
        #     trgt_list.append(trgt)

        
        x_context = x[:max_n_context] #[Nc, Dx]
        x_target = x[max_n_context:]

        x_context = jnp.tile(x_context, (batch_size, 1, 1))
        x_target = jnp.tile(x_target, (batch_size, 1, 1))

        # locations to sample at
        x = jnp.concatenate([x_context, x_target], axis=1) #[B, Nc+Nt, Dx]


    if task == "training":
        num_keep_target = jax.random.randint(mkey, (), minval=min_n_target, maxval=max_n_target) # () outputs scalar, not array

        # create mask with zeros for first num_keep_target entries
        mask_target = jnp.where(
            jnp.arange(max_n_target)[None, :] < num_keep_target,
            jnp.zeros_like(x_target)[..., 0],  # keep
            jnp.ones_like(x_target)[..., 0]  # ignore
        )
        mask_context = jnp.zeros_like(x_context[..., 0])

 
    elif task == "interpolation":
        num_keep_context = jax.random.randint(mkey, (), minval=1, maxval=max_n_context)
        mask_context = jnp.where(
            jnp.arange(max_n_context)[None, :] < num_keep_context,
            jnp.zeros_like(x_context)[..., 0],  # keep
            jnp.ones_like(x_context)[..., 0]  # ignore
        )
        mask_target = jnp.zeros_like(x_target[..., 0])
 
    keys = jax.random.split(key, batch_size)

    active_dims = list(range(input_dim))

    #calling factory will return distribution, which has a sample method
    sample_func = _DATASET_FACTORIES[name](active_dims).sample # a callable function

    y = jax.vmap(sample_func)(keys, x) # [B, Nc+Nt] # apply the sample function independently to each batch member

    return Batch(
        x_target=x_target,
        y_target=y[:, max_n_context:, :], # retain the part of y corresponding to context
        x_context=x_context,
        y_context=y[:, :max_n_context, :],
        mask_target=mask_target,
        mask_context=mask_context,
    )