import torch
from convcnp.utils import device

import os 

import matplotlib.pyplot as plt

def to_numpy(x):
    """Convert a PyTorch tensor to NumPy."""
    return x.squeeze().detach().cpu().numpy()


def plot_model_task(model, task, idx, ran = None, legend = False, save = None, name = None):


    # a task consists of many data sets, idx chooses which data set to plot
    # change to just take in and then plot one function

    num_functions = task['x_context'].shape[0]
    
    # Create a fixed set of outputs to predict at when plotting.
    if ran == None:
        x_test = torch.linspace(-2., 2., 200)[None, :, None].to(device) 
    else:

        x_test = torch.linspace(ran[0], ran[1], 200)[None, :, None].to(device) 

    # Make predictions with the model.
    model.eval()
    with torch.no_grad():
        y_mean, y_std = model(task['x_context'], task['y_context'], x_test.repeat(num_functions, 1, 1))
    
    # Plot the task and the model predictions.
    x_context, y_context = to_numpy(task['x_context'][idx]), to_numpy(task['y_context'][idx])
    # x_target, y_target = to_numpy(task['x_target'][idx]), to_numpy(task['y_target'][idx])
    y_mean, y_std = to_numpy(y_mean[idx]), to_numpy(y_std[idx])
    
    # Plot context and target sets.
    plt.scatter(x_context, y_context, label='Context Set', color='black')
    # plt.scatter(x_target, y_target, label='Target Set', color='red')
    
    # Plot model predictions.
    plt.plot(to_numpy(x_test[0]), y_mean, label='Model Output', color="C0")
    plt.fill_between(to_numpy(x_test[0]),
                     y_mean + 2 * y_std,
                     y_mean - 2 * y_std,
                     color='tab:blue', alpha=0.2)
    


    if legend:
        plt.legend()

    
    if save is not None:

        dir = "plots"
        name = f"{name}_{save}"

        path = os.path.join(dir, name)

        plt.savefig(path, dpi = 600)

    plt.show()