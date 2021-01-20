"""
trainer.py
==========================
Generic model training as a sacred ingredient.
"""
import sys; sys.path.append('../')
import os
import time
from collections import OrderedDict
from sacred import Ingredient
from tqdm import tqdm
import numpy as np
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from ignite.engine import create_supervised_trainer, create_supervised_evaluator, Events
from ignite.metrics import Accuracy, Loss, RunningAverage
from ignite.handlers import ModelCheckpoint, EarlyStopping
from ignite.contrib.metrics.roc_auc import ROC_AUC
from experiments.nets.losses import RMSELoss

train_ingredient = Ingredient('train')


@train_ingredient.config
def config():
    optimizer_name = 'adam'
    loss_str = 'ce'
    lr = None
    max_epochs = 1000
    metrics = ['loss']
    val_metric_to_monitor = 'loss'
    epoch_per_metric = 1
    print_freq = 5
    plateau_patience = 15
    plateau_terminate = 60
    gpu_if_available = True
    gpu_idx = -1


@train_ingredient.capture
def train(model,
          train_dl,
          val_dl,
          test_dl,
          loss_str,
          optimizer_name,
          lr,
          max_epochs,
          metrics,
          val_metric_to_monitor,
          print_freq,
          epoch_per_metric,
          plateau_patience,
          plateau_terminate,
          gpu_if_available,
          gpu_idx,
          custom_metrics=None,
          save_dir=None):
    """Simple model training framework setup with ignite.

    This builds and runs a standard training process using the ignite framework. Given train/val/test dataloaders,
    attaches specified metrics and runs over the training set with LR scheduling, early stopping, and model
    check-pointing all built in.

    Args:
        model (nn.Module): A network built in standard PyTorch.
        train_dl (DataLoader): Train data.
        val_dl (DataLoader): Val data.
        test_dl (DataLoader): Test data.
        optimizer_name (str): Name of the optimizer to use.
        lr (float): The initial value of the learning rate.
        loss_str (function): The loss function.
        max_epochs (int): Max epochs to run the algorithm for.
        metrics (list): A list of metric strings to be monitored.
        val_metric_to_monitor (str): The metric to monitor for LR scheduling and early stopping.
        print_freq (int): Frequency of printing train/val results to console.
        epoch_per_metric (int): Number of epochs before next computation of val metrics.
        plateau_patience (int): Number of epochs with no improvement before LR reduction.
        plateau_terminate (int): Number of epochs with no improvement before stopping.
        gpu_if_available (bool): Run on the gpu if one exists.
        gpu_idx (int): The index of the gpu to run on.
        custom_metrics (dict): Dictionary of custom metrics.
        save_dir (str): Location to save the model checkpoints.

    Returns:
        (results:dict, validation_history:dict): The results of the best model and the full training history.
    """
    device = set_device(gpu_if_available, gpu_idx=gpu_idx)
    loss_fn = set_loss(loss_str)
    lr = set_lr(train_dl) if lr is None else lr
    optimizer = setup_optimizer(model, optimizer_name, lr)

    # Choose metrics given the string list
    binary = True if isinstance(loss_fn, torch.nn.BCEWithLogitsLoss) else False
    metrics, train_metrics, val_metrics = setup_metrics(metrics, loss_fn, binary=binary, custom_metrics=custom_metrics)

    # Build engines
    trainer_output_tfm = lambda x, y, y_pred, loss: (loss.item(), y, y_pred)
    trainer = create_supervised_trainer(model, optimizer, loss_fn, device=device, output_transform=trainer_output_tfm)
    evaluator = create_supervised_evaluator(model, device=device, metrics=val_metrics)

    # Attach running average metrics to trainer
    for name, metric in train_metrics.items():
        metric.attach(trainer, name)

    # Progress bar
    pbar = tqdm(range(max_epochs))

    # Validation loop
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_metrics(engine):
        epoch = engine.state.epoch
        pbar.update(1)

        if (epoch % epoch_per_metric == 0) or (epoch == 0):
            evaluator.run(val_dl, max_epochs=1)

            add_metrics_to_dict(trainer.state.metrics, validation_history, '.train')
            add_metrics_to_dict(evaluator.state.metrics, validation_history, '.val')

            if (epoch % print_freq == 0) or (epoch == 0):
                print_val_results(epoch, validation_history, pbar=pbar)

    # Score to monitor for early stopping and check-pointing
    sign = -1 if val_metric_to_monitor is 'loss' else 1
    score_function = lambda engine: engine.state.metrics[val_metric_to_monitor] * sign

    # LR scheduling (monitors validation loss), early stopping and check-pointing
    scheduler = ReduceLROnPlateau(optimizer, patience=plateau_patience, threshold=1e-6, min_lr=1e-7)
    evaluator.add_event_handler(Events.EPOCH_COMPLETED, lambda engine: scheduler.step(engine.state.metrics['loss']))

    # Early stopping
    stopping = EarlyStopping(patience=plateau_terminate, score_function=score_function, trainer=trainer)
    evaluator.add_event_handler(Events.EPOCH_COMPLETED, stopping)

    # Checkpoint
    save_best_model = ModelCheckpoint(save_dir, '', score_function=score_function)
    evaluator.add_event_handler(Events.EPOCH_COMPLETED, save_best_model, {'best_model': model})

    # History
    validation_history = OrderedDict()
    for type in ('train', 'val'):
        for name in metrics:
            validation_history[name + '.' + type] = []

    # Train the model
    start, start_memory = time.time(), get_memory(device, reset=True)
    trainer.run(train_dl, max_epochs=max_epochs)
    elapsed = time.time() - start
    memory_usage = get_memory(device) - start_memory

    # Score on test
    model.load_state_dict(torch.load(save_best_model.last_checkpoint))
    evaluator.run(test_dl, max_epochs=1)

    # Final model results
    results = OrderedDict(**{
        'elapsed_time': elapsed,
        'memory_usage': memory_usage
    })

    # Best metric/value
    func = np.argmax if sign == 1 else np.argmin
    best_idx = func(validation_history[val_metric_to_monitor + '.val'])
    for key, value in validation_history.items():
        results[key] = value[best_idx]

    for metric, value in evaluator.state.metrics.items():
        results[metric + '.test'] = value

    print_final_results(results)

    return model, results, validation_history


def get_freest_gpu():
    """ GPU with most available memory. """
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    print(memory_available)
    return np.argmax(memory_available)


def set_device(gpu_if_available, gpu_idx=None):
    """Sets the cuda device to run on.

    This will only set if `gpu_is_available` is marked True. If `gpu_idx` is set, then will run on that gpu index, else
    will find the free-est gpu in terms of available memory and run on that.

    Args:
        gpu_if_available (bool): Set True to allow a gpu to be selected.
        gpu_idx (int): The index of the GPU to run on.

    Returns:
        torch.device: The GPU device.
    """
    device = torch.device('cpu')
    if torch.cuda.is_available():
        if gpu_if_available:
            if gpu_idx > -1:
                device = torch.device('cuda:{}'.format(gpu_idx))
            else:
                device = torch.device('cuda:{}'.format(get_freest_gpu()))
    return device


def set_loss(loss_str):
    if loss_str == 'ce':
        loss_fn = nn.CrossEntropyLoss()
    if loss_str == 'bce':
        loss_fn = nn.BCEWithLogitsLoss()
    elif loss_str == 'mse':
        loss_fn = nn.MSELoss()
    elif loss_str == 'rmse':
        loss_fn = RMSELoss()
    # if pvar_loss:
    #     loss_fn = PvarLossWrapper(loss_fn, lmbda=0.1)
    return loss_fn


def setup_optimizer(model, optimizer_name, lr, final_layer_scaling=10):
    """Sets up the optimizer according to a given name.

    If the model is a NeuralRDE, will multiply the learning rate of the final layer.

    Args:
        model (nn.Module): A PyTorch model.
        optimizer_name (str): Name from ['adam', 'sgd']
        lr (float): The main value of the lr.
        final_layer_scaling (float): Final layer lr becomes `lr * final_layer_scaling`.

    Returns:
        optimizer: A PyTorch optimizer.
    """
    optimizers = {
        'adam': optim.Adam,
        'sgd': optim.SGD
    }

    # Any param that has name starting 'final_linear' gets a multiplied lr by `final_layer_scaling`
    ps = []
    for name, param in model.named_parameters():
        lr_ = lr if not name.startswith('final_linear') else lr * final_layer_scaling
        ps.append({"params": param, "lr": lr_})
    optimizer = optimizers[optimizer_name](ps)

    return optimizer


def set_lr(train_dl):
    """ Sets the learning rate to `0.01 * 32 / batch_size`. """
    batch_size = train_dl.batch_sampler.batch_size if hasattr(train_dl, 'batch_sampler') else train_dl.batch_size
    return 0.01 * 32 / batch_size


def setup_metrics(metric_names, loss_fn, binary=False, custom_metrics=None):
    """ Choose metrics to monitor given a list of metric strings. """
    sigmoid_predict = lambda x: torch.round(torch.sigmoid(x))
    acc_train_tfm = lambda x: (x[2], x[1])
    acc_val_tfm = lambda x: x
    if binary:
        acc_train_tfm = lambda x: (torch.round(torch.sigmoid(x[2])), x[1])
        acc_val_tfm = lambda x: (sigmoid_predict(x[0]), x[1])

    train_metrics = {
        'acc': RunningAverage(Accuracy(output_transform=acc_train_tfm)),
        'loss': RunningAverage(output_transform=lambda x: x[0]),
        'auc': RunningAverage(ROC_AUC(output_transform=acc_train_tfm))
    }

    val_metrics = {
        'acc': Accuracy(output_transform=acc_val_tfm),
        'loss': Loss(loss_fn),
        'auc': RunningAverage(ROC_AUC(output_transform=acc_val_tfm))
    }

    assert [x in train_metrics.keys() for x in metric_names]

    # Subset the metrics
    train_metrics = {
        name: train_metrics[name] for name in metric_names
    }
    val_metrics = {
        name: val_metrics[name] for name in metric_names
    }

    # Add any custom metrics
    if custom_metrics is not None:
        for name, metric_class in custom_metrics.items():
            train_metrics[name] = RunningAverage(metric_class(output_transform=acc_train_tfm))
            val_metrics[name] = metric_class(output_transform=acc_val_tfm)
        metric_names += custom_metrics.keys()

    return metric_names, train_metrics, val_metrics


def add_metrics_to_dict(metrics, history, dot_str):
    """Adds metrics to a dict with the same keys.

    Args:
        metrics (dict): A dictionary of keys and lists, where the key is the metric name and the list is storing metrics
            over the epochs.
        history (dict): The history dict that contains the same keys as in metrics.
        dot_str (str): Metrics are labelled 'metric_name.{train/val}'.

    Returns:
        None
    """
    for name, value in metrics.items():
        history[name + dot_str].append(value)


def get_memory(device, reset=False, in_mb=True):
    """ Gets the GPU usage. """
    if device is None:
        return float('nan')
    if device.type == 'cuda':
        if reset:
            torch.cuda.reset_max_memory_allocated(device)
        bytes = torch.cuda.max_memory_allocated(device)
        if in_mb:
            bytes = bytes / 1024 / 1024
        return bytes
    else:
        return float('nan')


def print_val_results(epoch, history, pbar=None):
    """ Prints output of the validation loop to the console. """
    print_string = 'EPOCH: {}'.format(epoch)

    for name, value in history.items():
        if '.' in name:
            n_split = name.split('.')
            name = n_split[1].title() + ' ' + n_split[0]

        if 'acc' in name:
            val_str = '{:.1f}%'.format(value[-1] * 100)
        else:
            val_str = '{:.3f}'.format(value[-1])

        print_string += ' | {}: {}'.format(name, val_str)

    pbar.write(print_string) if pbar is not None else print(print_string)


def print_final_results(history):
    """ Prints final results to the console """
    print_string = '\nFinal Results: '
    for name, value in history.items():
        if '.' in name:
            n_split = name.split('.')
            name = n_split[1].title() + ' ' + n_split[0]

        if 'acc' in name:
            val_str = '{:.1f}%'.format(value * 100)
        else:
            val_str = '{:.3f}'.format(value)

        print_string += ' | {}: {}'.format(name, val_str)

    print(print_string)


