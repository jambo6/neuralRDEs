"""
get_data.py
===============================
Reusable sacred code for pro the data.
"""
from sacred import Ingredient
import torch
from torch.utils.data import DataLoader
from experiments.utils.get_data import train_test_val_split, original_idx_split
from experiments.utils.get_model import get_model
from experiments.utils.get_data import *
from experiments.utils.messy import remove_random_rows, impute_missing, drop_nan_imputation
from experiments.nets.grud import prepare_gru_variant_data
from ncdes.data.scalers import TrickScaler
from ncdes.data.dataset import FixedCDEDataset, FlexibleCDEDataset, SubsampleDataset
from ncdes.data.intervals import FixedIntervalSampler, RandomSampler, BatchIntervalSampler
from ncdes.data.functions import linear_interpolation

data_ingredient = Ingredient('data')


@data_ingredient.config
def config():
    # Location
    ds_folder = ''
    ds_name = 'EigenWorms'
    missing_rate = None
    include_observational_intensity = False
    imputation_method = 'ffill'
    # Norm
    scaling = 'stdsc'
    # Train/test/val split
    train_test_val_seed = 0
    val_frac = 0.15
    test_frac = 0.15
    # DataLoader
    batch_size = 1024
    sampler_name = 'fixed'
    sampler_args = {}
    # Model
    adjoint = False
    solver = 'rk4'


@data_ingredient.capture
def ready_all_data_and_model(_run,
                             model_type,
                             depth,
                             step,
                             hidden_dim,
                             hidden_hidden_multiplier,
                             num_layers,
                             tune_params,
                             adjoint,
                             solver,
                             ignore_model=False):
    """ Helper function for performing all data and model getting steps. """
    # Get the raw data
    train_data, val_data, test_data, output_dim, return_sequences = process_data(model_type=model_type)

    if 'folded' in model_type:
        def perform_fold(controls):
            L = controls.size(1)
            zeros = torch.zeros(controls.size(0), step - L % step, controls.size(2))
            new_controls = torch.cat([controls, zeros], 1)
            folded = new_controls.reshape(new_controls.size(0), int(new_controls.size(1) / step), -1)
            return folded
        train_data[0] = perform_fold(train_data[0])
        val_data[0] = perform_fold(val_data[0])
        test_data[0] = perform_fold(test_data[0])
        step = 1
        model_type = model_type.split('_')[0]
        assert model_type in ['nrde', 'rnn', 'gru', 'odernn']

    # Setup as datasets
    train_ds, train_sampler = build_dataset(model_type=model_type, data=train_data, depth=depth, step=step)
    val_ds, val_sampler = build_dataset(model_type=model_type, data=val_data, depth=depth, step=step)
    test_ds, test_sampler = build_dataset(model_type=model_type, data=test_data, depth=depth, step=step)

    # Setup as datasets
    train_dl = build_dataloader(train_ds, sampler=train_sampler)
    val_dl = build_dataloader(val_ds, sampler=val_sampler)
    test_dl = build_dataloader(test_ds, sampler=test_sampler)

    # Some params
    input_dim = train_ds.input_dim
    initial_dim = train_ds.initial_dim if model_type == 'nrde' else None

    # Get model
    model = None
    if not ignore_model:
        model, n_params = get_model(
            model_type, input_dim, hidden_dim, output_dim, hidden_hidden_multiplier=hidden_hidden_multiplier,
            num_layers=num_layers, initial_dim=initial_dim, return_sequences=return_sequences,
            tune_params=tune_params, adjoint=adjoint, solver=solver
        )
        _run.log_scalar('num_params', n_params)
        _run.log_scalar('true_hidden_dim', model.hidden_dim)
        _run.log_scalar('model_summary', repr(model))

    return model, train_dl, val_dl, test_dl


@data_ingredient.capture
def process_data(ds_folder, ds_name, missing_rate, include_observational_intensity, imputation_method,
                 train_test_val_seed, val_frac, test_frac, scaling, model_type=None):
    """ Performs various data preprocessing operations, such as ffill, train/val/test/split and scaling. """
    if ds_folder == 'FBM':
        controls, responses, output_dim, return_sequences, original_idxs = get_fbm_data(ds_name=ds_name)
    elif ds_folder == 'TSR':
        controls, responses, output_dim, return_sequences, original_idxs = get_tsr_data(ds_name=ds_name)
    elif ds_folder in ['UEA', 'SpeechCommands']:
        controls, responses, output_dim, return_sequences, original_idxs = get_classification_data(ds_name, ds_folder)
    elif ds_folder == 'PhysioNet':
        if ds_name == 'Mortality2012':
            controls, responses, output_dim, return_sequences, original_idxs = get_physionet2012_data()
        elif ds_name == 'Sepsis2019':
            controls, responses, output_dim, return_sequences, original_idxs = get_physionet2019_data()
    elif ds_folder == 'UJIPenChars2':
        controls, responses, output_dim, return_sequences, original_idxs = get_uji_data()

    else:
        raise NotImplementedError('No other getters yet implemented.')

    # Slightly hacky way of noting if this is a regression problem
    is_regression = True if len(responses.unique()) > len(responses) / 4 else False

    # Create messy data and perform relevant imputation
    if missing_rate is not None:
        controls = remove_random_rows(controls, missing_rate)

    # Additional channels for variants
    controls = prepare_additional_channels(controls, model_type, include_observational_intensity)

    # Imputation scheme
    controls = impute_data(controls, imputation_method)

    # Perform train/val/test split
    stratify_idx = 1 if all([return_sequences, ~is_regression]) else None
    tensors = (controls, responses)
    if original_idxs is None:
        train_data, val_data, test_data = train_test_val_split(
            tensors, val_frac, test_frac,
            stratify_idx=stratify_idx,
            seed=train_test_val_seed
        )
    else:
        train_data, val_data, test_data = original_idx_split(
            tensors, original_idxs, val_frac=val_frac, stratify_idx=stratify_idx, seed=train_test_val_seed
        )

    # Scaling if set
    train_data, val_data, test_data = scale_data(scaling, train_data, val_data, test_data)

    # Reshape the GRU-variant data
    train_data, val_data, test_data = ensure_3d(train_data, val_data, test_data)

    # return train_data, test_data, val_data, output_dim, return_sequences
    return train_data, val_data, test_data, output_dim, return_sequences


@data_ingredient.capture
def build_dataset(model_type, sampler_name, data, depth, step):
    """ Sets up the relevant dataset type dependent on model_type. """
    # First get the associated sampler
    ds_length = data[0].size(1)
    sampler = get_sampler(sampler_name, ds_length, sampler_args={'step': step})

    # Now build the dataset
    if model_type in ['logsig-rnn', 'nrde']:
        if sampler_name == 'fixed':
            # Takes a sampler as a dataset arg, then no sampler goes into the DataLoader
            dataset = FixedCDEDataset(*data, sampler=sampler, depth=depth)
            sampler = None
        else:
            dataset = FlexibleCDEDataset(*data, depth)
    elif model_type in ['gru', 'rnn', 'odernn', 'gru-dt', 'gru-d']:
        dataset = SubsampleDataset(*data)
    else:
        raise NotImplementedError('model_type:{} not implemented.'.format(model_type))

    return dataset, sampler


@data_ingredient.capture
def build_dataloader(dataset, sampler, batch_size):
    """ Simple dataloader setup. """
    if sampler is not None:
        sampler = BatchIntervalSampler(dataset.n_samples, sampler, batch_size)
        dataloader = DataLoader(dataset, batch_sampler=sampler, collate_fn=interval_collate_function)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size)
    return dataloader


def interval_collate_function(batch):
    """To be applied as the `collate_function` argument when we use a BatchIntervalSampler in a DataLoader.

    PyTorch is not equipped to do dataloading over variable intervals (on top of batches). This function is a slightly
    hacky way of making this data-loading work. This is also key to making the p-var loss function work.
    """
    if len(batch) == 1:
        return batch[0]
    assert len(batch) == 2, "Can only deal with [true_batch, diff_step_compare_batch]."
    b1, b2 = batch
    (initial_0, controls_0), responses_0 = b1
    (initial_1, controls_1), responses_1 = b2
    initial = torch.cat((initial_0, initial_1))
    size_diff = controls_0.size(1) - controls_1.size(1)
    controls_1_new = torch.cat((controls_1, controls_1[:, [-1], :].repeat(1, size_diff, 1)), dim=1)
    controls = torch.cat((controls_0, controls_1_new), 0)
    responses = responses_0.repeat(2)
    return [(initial, controls), responses]


def prepare_additional_channels(controls, model_type, include_observational_intensity):
    """ Additional channels for OI and gru variants. """
    if model_type in ['gru-d', 'gru-dt']:
        controls = prepare_gru_variant_data(controls, variant=model_type, intensity=include_observational_intensity)
    elif include_observational_intensity:
         intensity = (~torch.isnan(controls)).float()
         if model_type == ['nrde', 'logsig-rnn']:
             intensity = intensity.cumsum(axis=1)
         controls = torch.stack((controls, intensity)).transpose(0, 1)
    return controls


def impute_data(controls, imputation_method):
    """ Data imputation schemes. """
    slice_ = 0 if controls.dim() == 4 else slice(None)
    if imputation_method == 'nanfill':
        controls[:, slice_] = drop_nan_imputation(controls[:, slice_])
    elif imputation_method in ['ffill', 'mean']:
        controls[:, slice_], _ = impute_missing(controls[:, slice_], imputation_method)
    elif imputation_method == 'linear':
        controls[:, slice_] = linear_interpolation(controls[:, slice_])

    # If there are still missing values (usually all nan columns), fill them with zeros.
    # TODO sort this out a bit
    if torch.isnan(controls[:, slice_]).sum() > 0:
        controls[:, slice_][torch.isnan(controls[:, slice_])] = 0

    return controls


def scale_data(scaling, train_data, val_data, test_data, controls_idx=0):
    """ 3D tensor scaling on train/test/val. """
    scaler = TrickScaler(scaling=scaling)
    slice_ = 0 if train_data[controls_idx].dim() == 4 else slice(None)
    # Scaling
    train_data[controls_idx][:, :, 1:] = scaler.fit_transform(train_data[controls_idx][:, :, 1:])
    val_data[controls_idx][:, :, 1:] = scaler.transform(val_data[controls_idx][:, :, 1:])
    test_data[controls_idx][:, :, 1:] = scaler.transform(test_data[controls_idx][:, :, 1:])
    # train_data[controls_idx][:, slice_] = scaler.fit_transform(train_data[controls_idx][:, slice_])
    # val_data[controls_idx][:, slice_] = scaler.transform(val_data[controls_idx][:, slice_])
    # test_data[controls_idx][:, slice_] = scaler.transform(test_data[controls_idx][:, slice_])
    return train_data, val_data, test_data


def ensure_3d(train_data, val_data, test_data, controls_idx=0):
    """ For 4d data of shape [N, D, L, C], reshapes onto [N, L, D * C]. """
    if train_data[controls_idx].dim() == 4:
        _, D, L, C = train_data[controls_idx].size()
        reshaper = lambda x: x.transpose(1, 2).reshape(x.size(0), L, C * D)

        train_data[controls_idx] = reshaper(train_data[controls_idx])
        val_data[controls_idx] = reshaper(val_data[controls_idx])
        test_data[controls_idx] = reshaper(test_data[controls_idx])

    return train_data, val_data, test_data


def get_sampler(name, ds_length, sampler_args):
    if name == 'fixed':
        sampler = FixedIntervalSampler(ds_length, **sampler_args)
    elif name == 'random':
        sampler = RandomSampler(ds_length, **sampler_args)
    else:
        raise NotImplementedError('Only FixedIntervalSampler is currently implemented.')
    return sampler


