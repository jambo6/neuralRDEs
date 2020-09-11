"""
common.py
=======================
Training functions common to all experiment runs.
"""
import numpy as np
from experiments.nets.rnn import RNN, GRU
from experiments.nets.neural import ODE_RNN
from experiments.nets.grud import GRUD
from ncdes import NeuralRDE


def get_model(model_type,
              input_dim,
              hidden_dim,
              output_dim,
              hidden_hidden_multiplier=1,
              num_layers=1,
              initial_dim=None,
              return_sequences=False,
              tune_params=None,
              adjoint=False,
              solver='rk4'):
    """Gets the desired model according to the specified parameters.

    NOTE: Here we give an optional `total_params' argument. If this is specified, the `hidden_dim' argument is ignored
    and is instead tuned to make the number of learnable params as close to `total_params' as possible.
    """
    def model_getter(hidden_dim, hidden_hidden_dim=None):
        if model_type == 'nrde':
            model = NeuralRDE(
                initial_dim, input_dim, hidden_dim, output_dim, hidden_hidden_dim=hidden_hidden_dim,
                num_layers=num_layers, return_sequences=return_sequences, adjoint=adjoint, solver=solver
            )
        elif model_type == 'logsig-rnn':
            model = RNN(input_dim, hidden_dim, output_dim, return_sequences=return_sequences)
        elif model_type == 'rnn':
            model = RNN(input_dim, hidden_dim, output_dim, return_sequences=return_sequences)
        elif model_type in ['gru', 'gru-dt']:
            model = GRU(input_dim, hidden_dim, output_dim, return_sequences=return_sequences)
        elif model_type == 'grud':
            model = GRUD(input_dim, hidden_dim, output_dim, return_sequences=return_sequences)
        elif model_type == 'odernn':
            model = ODE_RNN(input_dim, hidden_dim, output_dim, gru=False, return_sequences=return_sequences)
        else:
            raise ValueError('model_type:{} not implemented.'.format(model_type))
        return model

    # Now we have a number of different situations. If tune_params *only* is set, then assume hidden_hidden is equal to
    # hidden, and num_layers is 1. If tune_params not set, all others must be set and we run with those. Finally if
    # params *and* tune params are set then reduce hidden and hidden_hidden until params are the same.
    all_params_set = all([isinstance(x, int) for x in [hidden_dim, hidden_hidden_multiplier, num_layers]])
    tune_set = False if tune_params is None else True
    if tune_set:
        # Tune the hidden_hidden_dim for the case of an nrde else hidden_dim
        if model_type == 'nrde':
            builder = lambda x: model_getter(hidden_dim, x)
        else:
            builder = lambda x: model_getter(x)
        model = tune_total_params(builder, tune_params)
    elif not tune_set and all_params_set:
        model = model_getter(hidden_dim, hidden_hidden_multiplier * hidden_dim)
    else:
        raise ValueError("`tune_params` or all of [hidden_dim, hidden_hidden_multiplier, num_layers] must be set.")

    # Get info on final number of params
    n_params = get_num_params(model)
    print(model, '\n', 'Num model parameters: {}'.format(n_params))

    return model, n_params


def get_num_params(model):
    """ Gets the number of trainable parameters in a pytorch model. """
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


def tune_total_params(builder, total_params):
    """Tunes the size of a given parameter to make number of model parameters as close to desired as possible.

    Given a lambda function that takes one parameter corresponding to a model parameter values and when called
    initialises a PyTorch model, this method will tune the value of the parameter

    Example:
        >>> builder = lambda hidden_dim: RNN(input_dim, hidden_dim, output_dim)
        >>> model = tune_total_params(builder, total_params=100)

    Args:
        builder (lambda function): A lambda function of the form `lambda param: model(*, param, *)'
        total_params (int): The desired number of total parameters

    Returns:
        The model initialised with the the given param.
    """
    min_params = get_num_params(builder(1))
    assert min_params <= total_params, 'Other params must be made smaller to ensure total params < {}. ' \
                                       'Min number of params is {}.'.format(total_params, min_params)

    n_params, hidden_dim = 0, 1
    while n_params < total_params:
        model = builder(hidden_dim)
        n_params = get_num_params(model)
        hidden_dim += 1

    # Revert back one if it is closer
    if hidden_dim - 2 > 0:
        prev_build = builder(hidden_dim-2)
        if total_params - get_num_params(prev_build) < total_params - n_params:
            model = prev_build

    return model


