import jax
import haiku as hk
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
# import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import optax
from functools import partial
from dataclasses import asdict
import os


import cartopy.crs as ccrs
import cartopy.feature as cfeature

import jaxlinop

# Disable all GPUs for TensorFlow. Load data using CPU.
tf.config.set_visible_devices([], 'GPU')
AUTOTUNE = tf.data.experimental.AUTOTUNE

from ml_tools.config_utils import setup_config
from ml_tools.state_utils import TrainingState

from custom_types import Dataset, Batch, Rng
from model import BiDimensionalAttentionModel, AttentionModel
from process import cosine_schedule, GaussianDiffusion, loss
# from neural_diffusion_processes.gp import predict


from config import Config

print("checkpoint 1")

print("checkpoint 2")

config: Config = setup_config(Config) # initialize an instance of class Config, whose attributes are all the relevant settings to tun stuff

####################################################################

# ####################
# set config.output_dim
# set config.input_dim
# set config.dataset
# set dim
# #####################


dim = (20,20) 
#dim = (32,38)
 
####################################################################

key = jax.random.PRNGKey(config.seed)
beta_t = cosine_schedule(config.diffusion.beta_start, config.diffusion.beta_end, config.diffusion.timesteps)
process = GaussianDiffusion(beta_t)


if config.output_dim == 2: #or config.dataset == "era5"
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



@jax.jit
def sample_prior(state: TrainingState, key: Rng, ran: tuple = None):

    if ran == None:
        x = jnp.linspace(-2, 2, 57)[:, None]

    else:
        l = ran[0]
        u = ran[1]
        x = jnp.linspace(l, u, 150)[:, None]

    net_with_params = partial(net, state.params_ema)
    y0 = process.sample(key, x, mask=None, model_fn=net_with_params, output_dim = config.output_dim)
    return x, y0


@jax.jit
def sample_conditional(state: TrainingState, key: Rng, xc = None, yc = None, ran: tuple = None ):
    
    if ran == None:
        x = jnp.linspace(-2, 2, 57)[:, None]

    else:
        l = ran[0]
        u = ran[1]
        x = jnp.linspace(l, u, 150)[:, None]

    if xc == None:
        xc = jnp.array([-1., 0., 1.]).reshape(-1, 1)
        yc = jnp.array([0., -1., 1.]).reshape(-1, 1)
    
    else:
        xc = jnp.array(xc).reshape(-1, 1)
        yc = jnp.array(yc).reshape(-1, 1)

    net_with_params = partial(net, state.params_ema)
    y0 = process.conditional_sample(key, x, mask=None, x_context=xc, y_context=yc, mask_context=None, model_fn=net_with_params)
    return x, y0, xc, yc

print("checkpoint 4")

def plot_cond_ci(state, key, n, xc, yc, ran, save = None, name = None):

    x, y0, xc, yc = jax.vmap(lambda k: sample_conditional(state, k, xc, yc, ran))(jax.random.split(key, n))



    mean, var = jnp.mean(y0, axis=0).squeeze(axis=1), jnp.var(y0, axis=0).squeeze(axis=1)

    fig, ax = plt.subplots(dpi = 600)

    x = x[0].squeeze(axis=1)
    ax.plot(x, mean, color = "C0")
    ax.plot(xc[...,0].T, yc[...,0].T, "ko")

    ax.fill_between(
            x,
            mean - 1.96 * jnp.sqrt(var),
            mean + 1.96 * jnp.sqrt(var),
            color='tab:blue',
            alpha=0.2
       )
    
    if save is not None:

        i = save
        dir = "plots"

        desc = f"{name}_CI_{i}"
        pathname = os.path.join(dir, desc)
        plt.savefig(pathname)
    #plt.show()


def plots(state: TrainingState, key: Rng, n_funct: int = 5, xc = None, yc = None, ran = None, save = None, name = None ):
    if config.input_dim != 1: return {}  # only plot for 1D inputs
    # prior
    fig_prior, ax = plt.subplots(dpi = 600)
    x, y0 = jax.vmap(lambda k: sample_prior(state, k, ran))(jax.random.split(key, n_funct))
    ax.plot(x[...,0].T, y0[...,0].T, color="C0", alpha=0.5)
    ax.set_title("prior")

    if save is not None:

        i = save
        dir = "plots"

        desc = f"{name}_prior_{i}"
        pathname = os.path.join(dir, desc)
        plt.savefig(pathname)


    # conditional
    fig_cond, ax = plt.subplots(dpi = 600)
    x, y0, xc, yc = jax.vmap(lambda k: sample_conditional(state, k, xc, yc, ran))(jax.random.split(key, n_funct))
    ax.plot(x[...,0].T, y0[...,0].T, color="C0", alpha=0.5)
    ax.plot(xc[...,0].T, yc[...,0].T, "ko")
    ax.set_title("conditional")

    if save is not None:

        i = save
        dir = "plots"

        desc = f"{name}_posterior_{i}"
        pathname = os.path.join(dir, desc)
        plt.savefig(pathname)


    #return {"prior": fig_prior, "conditional": fig_cond}

def get_grid_data():

    lat = jnp.linspace(-2,2,dim[0])
    lon = jnp.linspace(-2,2,dim[1])
    

    lon, lat = jnp.meshgrid(lon, lat)
    lat = -lat

    lon = jnp.reshape(lon, (-1,1))
    lat = jnp.reshape(lat, (-1,1))

    x = jnp.column_stack((lon,lat))

    return x

# for grid data
@jax.jit
def sample_prior_grid(state: TrainingState, key: Rng):

    x = get_grid_data()
    net_with_params = partial(net, state.params_ema)

    #need x.shape (N,2)
    y0 = process.sample(key, x, mask=None, model_fn=net_with_params, output_dim = config.output_dim)
    return x, y0


@jax.jit
def sample_conditional_grid(state: TrainingState, key: Rng, xc, yc ):
    

    x = get_grid_data()


    net_with_params = partial(net, state.params_ema)
    y0 = process.conditional_sample(key, x, mask=None, x_context=xc, y_context=yc, mask_context=None, model_fn=net_with_params)
    return x, y0, xc, yc


def plot_germany(data, target = None, ctxt_i = None, save = None):

    if target is not None:

        # Coordinates (longitude and latitude) for the data
        
        lat = np.linspace(47.3, 55.1, data.shape[0])  # Latitude range for Germany  (down up)
        lon = np.linspace(5.9, 15.0, data.shape[1])  # Longitude range for Germany (left right)

        x1 = jnp.arange(38)
        x2 = jnp.arange(32)  
            
        x1, x2 = jnp.meshgrid(x1, x2)

        x1 = jnp.reshape(x1, (-1,1))
        x2 = jnp.reshape(x2, (-1,1))

        x = jnp.column_stack((x2,x1))
        

        x_ctxt = x[ctxt_i, :]

        # Create the figure with 1 row and 2 columns of subplots
        fig, axs = plt.subplots(1, 2, figsize=(14, 7), subplot_kw={'projection': ccrs.PlateCarree()}, dpi = 600)

        # Adjust the space between the subplots
        plt.subplots_adjust(wspace=0.1)

        # Define the extent (longitude and latitude boundaries) for the plots
        extent = [lon.min(), lon.max(), lat.min(), lat.max()]

        vmin = min(data.min(), target.min())  
        vmax = max(data.max(), target.max()) 

        # Plot the first heatmap
        axs[0].imshow(data, origin='upper', extent=extent,
                      transform=ccrs.PlateCarree() ,cmap='viridis',
                      vmin = vmin, vmax = vmax)
        axs[0].add_feature(cfeature.BORDERS, linestyle='-', linewidth=1, edgecolor='black')
        axs[0].add_feature(cfeature.COASTLINE, linestyle='-', linewidth=1, edgecolor='black')
        axs[0].gridlines(visible=False)
        axs[0].set_xticks([])
        axs[0].set_yticks([])
        axs[0].set_aspect(1.55)
        axs[0].set_title('prediction')


        # Create an overlay axes for placing rectangles in grid coordinates
        overlay_ax = axs[0].inset_axes([0, 0, 1, 1], transform=axs[0].transAxes, zorder=10)
        overlay_ax.set_xlim(0, data.shape[1])
        overlay_ax.set_ylim(0, data.shape[0])
        overlay_ax.axis('off')

        # Plot red frames around specific points based on grid coordinates
        for loc in x_ctxt:
            # Extract the grid row and column indices
            row_idx = loc[0]
            col_idx = loc[1]
            
            # Create a red rectangle based on grid coordinates
            rect = patches.Rectangle((col_idx, data.shape[0] - row_idx - 1), 1, 1, linewidth=1, edgecolor='red', facecolor='none')
            overlay_ax.add_patch(rect)
        

        # Plot the second heatmap
        axs[1].imshow(target, origin='upper', extent=extent, 
                      transform=ccrs.PlateCarree(), cmap='viridis', 
                      vmin = vmin, vmax = vmax)
        axs[1].add_feature(cfeature.BORDERS, linestyle='-', linewidth=1, edgecolor='black')
        axs[1].add_feature(cfeature.COASTLINE, linestyle='-', linewidth=1, edgecolor='black')
        axs[1].gridlines(visible=False)
        axs[1].set_xticks([])
        axs[1].set_yticks([])
        axs[1].set_aspect(1.55)
        axs[1].set_title('real')

        if save is not None:

            i = save
            dir = "plots"

            name = f"era5_posterior_{i}"
            pathname = os.path.join(dir, name)
            plt.savefig(pathname)

            return
        
        # Show the plot
        plt.show()

    else:

        # Coordinates (longitude and latitude) for the data
        lat = np.linspace(47.3, 55.1, data.shape[1])  # Latitude range for Germany
        lon = np.linspace(5.9, 15.0, data.shape[0])  # Longitude range for Germany
        

        # Create the figure and axis with a simple PlateCarree projection
        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={'projection': ccrs.PlateCarree()}, dpi = 600)

        # Plot the heatmap using imshow with correct extent
        extent = [lon.min(), lon.max(), lat.min(), lat.max()]
        ax.imshow(data, origin='upper', extent=extent, transform=ccrs.PlateCarree(), cmap='viridis')

        # Overlay the outline of Germany
        ax.add_feature(cfeature.BORDERS, linestyle='-', linewidth=1, edgecolor='black')
        ax.add_feature(cfeature.COASTLINE, linestyle='-', linewidth=1, edgecolor='black')

        # Remove gridlines and ticks
        ax.gridlines(visible=False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect(1.55)

        if save is not None:
            i = save
            dir = "plots"

            name = f"era5{i}"
            pathname = os.path.join(dir, name)
            plt.savefig(pathname)

            return
        
        # Show the plot
        plt.show()

        




def plot_heatmap(data):
    """
    Plots a heatmap from a 2D array.

    Parameters:
    data (2D array-like): The data to be plotted as a heatmap.
    """
    plt.figure(figsize=(5, 5))
    sns.heatmap(data, annot=False, fmt=".2f", cmap='viridis')
    plt.title('Heatmap')
    plt.show()



print("checkpoint 5")

batch_init = Batch(
    x_target=jnp.zeros((config.batch_size, 10, config.input_dim)),
    y_target=jnp.zeros((config.batch_size, 10, config.output_dim)),
    x_context=jnp.zeros((config.batch_size, 10, config.input_dim)),
    y_context=jnp.zeros((config.batch_size, 10, config.output_dim)),
    mask_context=jnp.zeros((config.batch_size, 10)),
    mask_target=jnp.zeros((config.batch_size, 10)),
)
