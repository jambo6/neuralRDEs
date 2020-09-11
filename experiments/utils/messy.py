"""
messy.py
=========================================
Methods for 'messing' up the tensor data.
"""
import torch
import numpy as np
from ncdes.data.functions import torch_ffill


def remove_random_rows(data, missing_rate, seed=0):
    """Removes rows randomly at a rate of `missing_rate` for each element of the batch.

    Args:
        data (torch.Tensor): Input data of shape [N, L, C]
        missing_rate (float): The rate at which to drop data in (0, 1).
        seed (int): Numpy random seed.

    Returns:
        torch.Tensor: Shape [N, L, C] but rows have been randomly turned to nan's in the L dimension.
    """
    if seed is not None:
        np.random.seed(seed)

    num_samples, length = data.size()[0:2]
    num_keep = int(length * missing_rate)
    idx_remove = np.stack([np.random.permutation(length) for _ in range(num_samples)])[:, :num_keep]

    for i in range(len(data)):
        data[i, idx_remove[i]] = float('nan')

    return data


def impute_missing(data, method):
    """Imputation of missing data into a 3D tensor

    The two methods here are 'ffill' and 'mean'.
        mean - Fills nan values with the feature means.
        ffill - Forward fills the tensor. Any initial nan values (that cannot be ffilled) are then filled with the
            feature means.

    Args:
        data (torch.Tensor):
        method (str): One of 'ffill' or 'mean'.

    Returns:
        (torch.Tensor, torch.Tensor): The forward filled data and a masking tensor.
    """
    # First get the masking tensor
    mask = (~torch.isnan(data)).int()

    # If ffill then we ffill first
    if method == 'ffill':
        data = torch_ffill(data)

    # Now impute with the column means
    N, L, C = data.size()
    data_reshaped = data.reshape(-1, C).numpy()
    col_mean = np.nanmean(data_reshaped, axis=0)
    inds = np.where(np.isnan(data_reshaped))
    data_reshaped[inds] = np.take(col_mean, inds[1])
    data_filled = torch.Tensor(data_reshaped).view(N, L, C)

    return data_filled, mask


def drop_nan_imputation(controls):
    """The method of imputation where we simply drop all interior values.

    If the data is filled with nan values, this function first removes all rows that are fully nan (that is, no
    measurement/new data was taken). It then forward fills any remaining entries and pads with the final value repeated
    to make everything fit into a tensor. The output of this is then a tensor such that for each batch element, values
    are only recorded when new data was actually imputed into the system. It also includes the corresponding times for
    which these events happened. Crucially, these times are different for different samples.
    """
    raise NotImplementedError('Needs to be fixed as `times` has been removed from the args`')
    # We need the max number of data pieces to forward fill the end to.
    max_data_pieces = (~torch.isnan(controls)).sum(axis=1).max()

    new_times, new_controls = [], []
    for times_, controls_ in zip(times, controls):
        # Keep any row with at least one piece
        mask = (~torch.isnan(controls_)).sum(axis=1) > 0
        new_times_ = times_[mask]
        new_controls_ = controls_[mask]
        # In cases where there are still nans (because nans are not column uniform), forward fill
        if torch.isnan(new_controls_).sum() > 0:
            new_controls_ = torch_ffill(new_controls_)
        # Now we need to make everything of the max data size so it fits in a tensor. We just copy the last element.
        num_to_fill = (max_data_pieces - len(new_times_)).item()
        if num_to_fill > 0:
            new_times_ = torch.cat((new_times_, new_times_[[-1]].repeat(num_to_fill, 1)))
            new_controls_ = torch.cat((new_controls_, new_controls_[[-1]].repeat(num_to_fill, 1)))
        new_times.append(new_times_)
        new_controls.append(new_controls_)

    new_times = torch.stack(new_times)
    new_controls = torch.stack(new_controls)

    return new_times, new_controls


if __name__ == '__main__':
    times = torch.arange(50).repeat(10, 1)
    controls = torch.randn(10, 50, 3)
    controls = remove_random_rows(controls, 0.5)
    times, controls = drop_nan_imputation(times, controls)
    # controls, mask = impute_missing(controls, 'ffill')