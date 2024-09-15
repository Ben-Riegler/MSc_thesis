import argparse
import os

import numpy as np
import stheno.torch as stheno
import torch
import matplotlib.pyplot as plt

import convcnp.data
from convcnp.architectures import SimpleConv, UNet
from convcnp.cnp import RegressionANP as ANP
from convcnp.cnp import RegressionCNP as CNP
from convcnp.experiment import (
    report_loss,
    generate_root,
    WorkingDirectory,
    save_checkpoint
)
from convcnp.set_conv import ConvCNP
from convcnp.utils import device, gaussian_logpdf

def to_numpy(x):
    """Convert a PyTorch tensor to NumPy."""
    return x.squeeze().detach().cpu().numpy()

PLOT_FREQ = 10


def validate(data, model, report_freq=None):
    """Compute the validation loss."""
    model.eval()
    likelihoods = []
    with torch.no_grad():
        for step, task in enumerate(data):
            num_target = task['y_target'].shape[1]
            y_mean, y_std = \
                model(task['x_context'], task['y_context'], task['x_target'])
            obj = \
                 gaussian_logpdf(task['y_target'], y_mean, y_std,
                                 'batched_mean')
            likelihoods.append(obj.item() / num_target)
            if report_freq:
                avg_ll = np.array(likelihoods).mean()
                report_loss('Validation', avg_ll, step, report_freq)
    avg_ll = np.array(likelihoods).mean()
    return avg_ll


def train(data, model, opt, report_freq):
    """Perform a training epoch."""
    model.train()
    losses = []
    for step, task in enumerate(data):
        y_mean, y_std = model(task['x_context'], task['y_context'],
                              task['x_target'])
        obj = -gaussian_logpdf(task['y_target'], y_mean, y_std, 'batched_mean')

        # Optimization
        obj.backward()
        opt.step()
        opt.zero_grad()

        # Track training progress
        losses.append(obj.item())
        avg_loss = np.array(losses).mean()
        report_loss('Training', avg_loss, step, report_freq)
    return avg_loss


# Parse arguments given to the script.
parser = argparse.ArgumentParser()
parser.add_argument('data',
                    choices=['eq',
                             'matern',
                             'noisy-mixture',
                             'weakly-periodic',
                             "step",
                             "sin",
                             'sawtooth'],
                    help='Data set to train the CNP on. ')
parser.add_argument('model',
                    choices=['convcnp', 'convcnpxl', 'cnp', 'anp'],
                    help='Choice of model. ')
parser.add_argument('--root',
                    help='Experiment root, which is the directory from which '
                         'the experiment will run. If it is not given, '
                         'a directory will be automatically created.')
parser.add_argument('--train',
                    action='store_true',
                    help='Perform training. If this is not specified, '
                         'the model will be attempted to be loaded from the '
                         'experiment root.')
parser.add_argument('--epochs',
                    default=100,
                    type=int,
                    help='Number of epochs to train for.')
parser.add_argument('--learning_rate',
                    default=1e-3,
                    type=float,
                    help='Learning rate.')
parser.add_argument('--weight_decay',
                    default=1e-5,
                    type=float,
                    help='Weight decay.')
args = parser.parse_args()

# Load working directory.
if args.root:
    wd = WorkingDirectory(root=args.root)
else:
    experiment_name = f'{args.model}-{args.data}'
    wd = WorkingDirectory(root=generate_root(experiment_name))

# Load data generator.
if args.data == 'sawtooth':
    gen = convcnp.data.SawtoothGenerator()
    gen_val = convcnp.data.SawtoothGenerator(num_tasks=60)
    gen_test = convcnp.data.SawtoothGenerator(num_tasks=2048)

if args.data == 'step':
    gen = convcnp.data.StepGenerator()
    gen_val = convcnp.data.StepGenerator(num_tasks=60)
    gen_test = convcnp.data.StepGenerator(num_tasks=2048)

elif args.data == "sin":
    gen = convcnp.data.SinGenerator()
    gen_val = convcnp.data.SawtoothGenerator(num_tasks=60)
    gen_test = convcnp.data.SawtoothGenerator(num_tasks=2048)

else:
    if args.data == 'eq':
        kernel = stheno.EQ().stretch(0.25)
    elif args.data == 'matern':
        kernel = stheno.Matern52().stretch(0.25)
    elif args.data == 'noisy-mixture':
        kernel = stheno.EQ().stretch(1.) + \
                 stheno.EQ().stretch(.25) + \
                 0.001 * stheno.Delta()
    elif args.data == 'weakly-periodic':
        kernel = stheno.EQ().stretch(0.5) * stheno.EQ().periodic(period=0.25)
    else:
        raise ValueError(f'Unknown data "{args.data}".')

    gen = convcnp.data.GPGenerator(kernel=kernel)
    gen_val = convcnp.data.GPGenerator(kernel=kernel, num_tasks=60)
    gen_test = convcnp.data.GPGenerator(kernel=kernel, num_tasks=2048)

# Load model.
if args.model == 'convcnp':
    model = ConvCNP(learn_length_scale=True,
                    points_per_unit=64,
                    architecture=SimpleConv())
elif args.model == 'convcnpxl':
    model = ConvCNP(learn_length_scale=True,
                    points_per_unit=64,
                    architecture=UNet())
elif args.model == 'cnp':
    model = CNP(latent_dim=128)
elif args.model == 'anp':
    model = ANP(latent_dim=128)
else:
    raise ValueError(f'Unknown model {args.model}.')


# Create a fixed set of outputs to predict at when plotting.
x_test = torch.linspace(-2., 2., 200)[None, :, None].to(device)


def plot_model_task(model, task, idx, legend):
    num_functions = task['x_context'].shape[0]
    
    # Make predictions with the model.
    model.eval()
    with torch.no_grad():
        y_mean, y_std = model(task['x_context'], task['y_context'], x_test.repeat(num_functions, 1, 1))
    
    # Plot the task and the model predictions.
    x_context, y_context = to_numpy(task['x_context'][idx]), to_numpy(task['y_context'][idx])
    x_target, y_target = to_numpy(task['x_target'][idx]), to_numpy(task['y_target'][idx])
    y_mean, y_std = to_numpy(y_mean[idx]), to_numpy(y_std[idx])
    
    # Plot context and target sets.
    plt.scatter(x_context, y_context, label='Context Set', color='black')
    plt.scatter(x_target, y_target, label='Target Set', color='red')
    
    # Plot model predictions.
    plt.plot(to_numpy(x_test[0]), y_mean, label='Model Output', color='blue')
    plt.fill_between(to_numpy(x_test[0]),
                     y_mean + 2 * y_std,
                     y_mean - 2 * y_std,
                     color='tab:blue', alpha=0.2)
    if legend:
        plt.legend()


model.to(device)

# Perform training.
opt = torch.optim.Adam(model.parameters(),
                       args.learning_rate,
                       weight_decay=args.weight_decay)

if args.train:
    # Run the training loop, maintaining the best objective value.
    best_obj = -np.inf
    for epoch in range(args.epochs):
        print('\nEpoch: {}/{}'.format(epoch + 1, args.epochs))

        # Compute training objective.
        train_obj = train(gen, model, opt, report_freq=50)
        report_loss('Training', train_obj, 'epoch')

        # Compute validation objective.
        val_obj = validate(gen_val, model, report_freq=20)
        report_loss('Validation', val_obj, 'epoch')

        # Update the best objective value and checkpoint the model.
        is_best = False
        if val_obj > best_obj:
            best_obj = val_obj
            is_best = True

        save_checkpoint(wd,
                        {'epoch': epoch + 1,
                         'state_dict': model.state_dict(),
                         'best_acc_top1': best_obj,
                         'optimizer': opt.state_dict()},
                        is_best=is_best)

        # Plot model behaviour every now and again.
        if epoch % PLOT_FREQ == 0:
            print('Epoch %s: NLL %.3f' % (epoch, train_obj))
            task = gen.generate_task()
            fig = plt.figure(figsize=(24, 5))
            for i in range(3):
                plt.subplot(1, 3, i + 1)
                plot_model_task(model, task, idx=i, legend=i==2)
            
            
            if not os.path.exists(wd.file("plots")):
                dir = str(wd.file("plots"))

                os.makedirs(dir, exist_ok=True)

            path = f'plots/plot_epoch{epoch}.png'
            plt.savefig(wd.file(path))
            plt.close()
else:
    # Load saved model.
    load_dict = torch.load(wd.file('model_best.pth.tar', exists=True))
    model.load_state_dict(load_dict['state_dict'])

# Finally, test model on ~2000 tasks.
test_obj = validate(gen_test, model)
print('Model averages a log-likelihood of %s on unseen tasks.' % test_obj)
with open(wd.file('test_log_likelihood.txt'), 'w') as f:
    f.write(str(test_obj))

