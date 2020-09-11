"""
get_data.py
======================
Functions for loading the data in raw format.
"""
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from experiments.utils.functions import load_pickle
from ncdes.data.functions import torch_ffill
from ncdes.data.scalers import TrickScaler
DATA_DIR = '../data'


def get_classification_data(ds_name, ds_folder=''):
    """ For getting data in the standard form where we have `data.pkl`, `labels.pkl`, `original_idxs.pkl` file. """
    # Load raw
    folder_str = '' if ds_folder == '' else '/' + ds_folder
    loc = DATA_DIR + '/processed' + folder_str + '/' + ds_name
    controls = load_pickle(loc + '/data.pkl')
    responses = load_pickle(loc + '/labels.pkl').long().view(-1)

    # Ensure responses start at 0
    responses = responses - responses.min()

    # Some params
    output_dim = len(torch.unique(responses))
    return_sequences = False

    # Make times and add to the controls
    num_samples, length = controls.size()[0:2]
    times = torch.linspace(0, 1, length).repeat(num_samples, 1).unsqueeze(-1)
    controls = torch.cat((times, controls), dim=2)

    return controls, responses, output_dim, return_sequences, None


def get_physionet2012_data(contained_value_fraction=0.25):
    # Load
    loc = DATA_DIR + '/processed/Physionet/Mortality2012'
    controls = load_pickle(loc + '/data.pkl')
    responses = load_pickle(loc + '/labels.pkl').float().view(-1, 1)
    original_idxs = load_pickle(loc + '/original_idxs.pkl')
    column_names = load_pickle(loc + '/all_columns.pkl')
    contained_values = load_pickle(loc + '/contained_values.pkl').values[:-1]


    # Remove features that contain < x% of values
    idxs = np.argwhere(contained_values > contained_value_fraction).reshape(-1)
    static_features = controls[:, :, [x for x in range(controls.size(2)) if x not in idxs]]
    controls = controls[:, :, idxs]

    # # For getting the names of the static features
    # demographics = ['Age', 'Gender', 'Height', 'ICUType', 'Weight']
    # variable_columns = [column_names[x] for x in idxs if column_names[x] not in demographics]

    # Params
    output_dim = 1
    classification = True

    # Time is first idx
    times = controls[:, :, [0]]
    times = torch_ffill(times)
    controls[:, :, [0]] = times

    return controls, responses, output_dim, classification, original_idxs


def get_physionet2019_data(max_length=72):
    # Load
    loc = DATA_DIR + '/processed/Physionet2019'
    all_data = load_pickle(loc + '/data.pkl')
    responses = load_pickle(loc + '/labels.pkl').float().view(-1, 1)
    column_names = load_pickle(loc + '/column_names.pkl')

    # Reduce length
    all_data = all_data[:, :max_length]

    FEATURE_TYPES = {
        'vitals': ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp'],
        'laboratory': [
            'EtCO2', 'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN', 'Alkalinephos', 'Calcium',
            'Chloride', 'Creatinine', 'Bilirubin_direct', 'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium',
            'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC', 'Fibrinogen', 'Platelets'
        ],
        'demographics': ['Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime', 'Hospital'],
    }

    # Find where the cols exist
    vitals_idxs = index_getter(column_names, FEATURE_TYPES['vitals'])
    static_idxs = index_getter(column_names, FEATURE_TYPES['laboratory'] + FEATURE_TYPES['demographics'])

    # Subset
    static_data = all_data[:, :, static_idxs]
    controls = all_data[:, :, vitals_idxs]

    # Add time
    times = torch.linspace(0, 1, controls.size(1)).repeat(controls.size(0)).view(-1, max_length, 1)
    controls = torch.cat((times, controls), dim=2)

    # Forward fill the static data, anything not filled let be zero, and consider only the terminal value.
    static_data = torch_ffill(static_data)
    static_data = static_data[:, -1, :]
    static_data[torch.isnan(static_data)] = 0

    # Params
    output_dim = 1
    return_sequences = True

    return controls, responses, output_dim, return_sequences, None


def get_tsr_data(ds_name):
    folder = DATA_DIR + '/processed/TSR/' + ds_name
    controls = load_pickle(folder + '/data.pkl')
    responses = load_pickle(folder + '/labels.pkl')
    responses = responses.unsqueeze(1)

    # Add time
    times = torch.linspace(0, 1, controls.size(1)).repeat(controls.size(0)).view(-1, controls.size(1), 1)
    controls = torch.cat((times, controls), dim=2)

    original_idxs = [list(x.numpy().reshape(-1)) for x in load_pickle(folder + '/original_idxs.pkl')]
    original_idxs = None
    return_sequences = False
    output_dim = 1
    return controls, responses, output_dim, return_sequences, original_idxs


def get_fbm_data(ds_name):
    folder = DATA_DIR + '/processed/FBM'
    times, controls, responses = load_pickle(folder + '/{}'.format(ds_name if '.pkl' in ds_name else ds_name + '.pkl'))
    output_dim = responses.size(2)
    return_sequences = True
    times = times.unsqueeze(-1)
    controls = torch.cat((times, controls), dim=2)
    return controls, responses, output_dim, return_sequences, None


def get_uji_data(min_num_samples=50, ffill=True, normalise='mmslocal', irregular_times=False):
    # Data
    folder = '../../data/processed/UJIPenChars2/UJIPenChars2'
    controls = load_pickle(folder + '/data.pkl')
    responses = load_pickle(folder + '/labels.pkl')
    char_labels = load_pickle(folder + '/alphabetic_labels.pkl')

    # Choose time definition
    if irregular_times:
        num_nan = (~torch.isnan(controls[:, :, 0])).sum(axis=1)
        times = pad_sequence([torch.linspace(0, 1, x) for x in num_nan], padding_value=float('nan')).T.unsqueeze(-1)
    else:
        times = torch.linspace(0, 1, controls.size(1)).repeat(controls.size(0)).view(-1, controls.size(1), 1)
    controls = torch.cat((times, controls), dim=2)

    # Preprocess
    if normalise == 'mmsglobal':
        controls = TrickScaler(scaling='mms').fit_transform(controls)
    elif normalise == 'mmslocal':
        maxs = torch.Tensor(np.nanmax(controls, axis=1))
        mins = torch.Tensor(np.nanmin(controls, axis=1))
        controls = ((controls.transpose(0, 1) - mins) / (maxs - mins)).transpose(0, 1)
    if ffill:
        controls = torch_ffill(controls)

    # Remove anything with < 50 samples
    unique, counts = torch.unique(responses, return_counts=True)
    remove_labels = unique[counts < min_num_samples]
    mask = torch.Tensor([False if x in remove_labels else True for x in responses]).to(bool)
    controls, responses, char_labels = controls[mask], responses[mask], char_labels[mask]

    output_dim = 1
    return_sequences = False
    original_idxs = None

    return controls, responses, output_dim, return_sequences, original_idxs


def train_test_val_split(tensors,
                         val_frac=0.15,
                         test_frac=0.15,
                         stratify_idx=None,
                         shuffle=True,
                         seed=None):
    """Train test split method for an arbitrary number of tensors.

    Args:
        tensors (list): A list of torch tensors.
        val_frac (float): The fraction to use as validation data.
        test_frac (float): The fraction to use as test data.
        stratify_idx (int): The index of the `tensors` variable to use as stratification labels.
        shuffle (bool): Set True to shuffle first.
        seed (int): Random seed.

    Returns:
        tuple: A tuple containing three lists corresponding to the train/val/test split of `tensors`.
    """
    num_samples = tensors[0].size(0)
    assert [t.size(0) == num_samples for t in tensors]

    if seed is not None:
        np.random.seed(seed)

    # Stratification labels
    stratify = None
    if isinstance(stratify_idx, int):
        stratify = tensors[stratify_idx] if tensors[stratify_idx].dim() <= 2 else None

    # Get train/val and test data split
    train_val_test_split = train_test_split(*tensors, stratify=stratify, shuffle=shuffle, test_size=test_frac)
    train_val_data, test_data = train_val_test_split[0::2], train_val_test_split[1::2]

    # Now split train val data
    stratify = None if stratify is not isinstance(stratify_idx, int) else train_val_data[stratify_idx]
    val_frac_of_train = val_frac / (1 - test_frac)
    train_val_split = train_test_split(*train_val_data, stratify=stratify, shuffle=shuffle, test_size=val_frac_of_train)
    train_data, val_data = train_val_split[0::2], train_val_split[1::2]

    return train_data, val_data, test_data


def original_idx_split(tensors, original_idxs, val_frac=0.15, stratify_idx=None, shuffle=True, seed=None):
    """Splits the data according to some original, pre-defined splits.

    Args:
        tensors (list): A list of tensors to split.
        original_idxs (list): A list of indexes. If of length 3 assumed to be [train idxs, val idxs, test_idxs], if of
            length 2 assumed to be [train + val idxs, test idxs].
        val_frac (float): The proportion of the training set to use as validation data. Ignored if `original_idxs` if of
            length 2.
        stratify_idx (int): The tensor index on which to stratify the data, again works only in length 2 case.
        shuffle (bool): Set True to shuffle first.

    Returns:
        tuple: A tuple containing three lists corresponding to the train/val/test split of `tensors`.
    """
    if seed is not None:
        np.random.seed(seed)

    assert any([original_idxs is None, len(original_idxs) == 2, len(original_idxs) == 3])

    if len(original_idxs) == 3:
        train_data, val_data, test_data = [[tensor[idxs] for tensor in tensors] for idxs in original_idxs]
    else:
        train_val_data, test_data = [[tensor[idxs] for tensor in tensors] for idxs in original_idxs]
        stratify = None if not isinstance(stratify_idx, int) else tensors[stratify_idx]
        train_val_split = train_test_split(*train_val_data, stratify=stratify, shuffle=shuffle, test_size=val_frac)
        train_data, val_data = train_val_split[0::2], train_val_split[1::2]

    return train_data, val_data, test_data


def index_getter(full_list, idx_items):
    """Boolean mask for the location of the idx_items inside the full list.

    Args:
        full_list (list): A full list of items.
        idx_items (list/str): List of items you want the indexes of.

    Returns:
        list: Boolean list with True at the specified column locations.
    """
    # Turn strings to list format
    if isinstance(idx_items, str):
        idx_items = [idx_items]

    # Check that idx_items exist in full_list
    diff_cols = [c for c in idx_items if c not in full_list]
    assert len(diff_cols) == 0, "The following cols do not exist in the dataset: {}".format(diff_cols)

    # Actual masking
    col_mask = [c in idx_items for c in full_list]

    return col_mask


