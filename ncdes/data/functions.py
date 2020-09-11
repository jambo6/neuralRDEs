import numpy as np
import torch


def torch_ffill(data):
    """ Forward fills in the length dim if data is shape [N, L, C]. """
    def fill2d(x):
        """ Forward fills in the L dimension if L is of shape [L, N]. """
        mask = np.isnan(x)
        idx = np.where(~mask, np.arange(mask.shape[1]), 0)
        np.maximum.accumulate(idx, axis=1, out=idx)
        out = x[np.arange(idx.shape[0])[:, None], idx]
        return out

    if isinstance(data, list):
        data_out = [torch.Tensor(fill2d(d.numpy().T).T) for d in data]
    elif data.dim() == 3:
        # Reshape to apply 2d ffill
        data_shaped = data.transpose(1, 2).reshape(-1, data.size(1)).numpy()
        data_fill = fill2d(data_shaped).reshape(-1, data.size(2), data.size(1))
        data_out = torch.Tensor(data_fill).transpose(1, 2)
    elif data.dim() == 2:
        data_out = torch.Tensor(fill2d(data.numpy().T)).T
    else:
        raise NotImplementedError('Needs implementing for different dimensions.')

    return data_out


def linear_interpolation(data, fill_start=True, fill_end=True, fill_remaining='zeros'):
    """Speedy implementation of linear interpolation for 3D tensors.

    Given a data tensor of shape [N, L, C] that is filled with nan values, and a corresponding times tensor of shape
    [N, L] the corresponds to the time the data was collected for each row, this function will linearly interpolate the
    data according to the times without the use of any for loops. This is done by forward filling and backward filling
    both the data and times, so we now have for any [batch_idx, i, j] entry the next observed value, the last observed
    value, and the times at which these happen. From here we simply do `last + obs_time * ((next - last) / time_diff)`
    to fill in any nan values.

    Args:
        data (torch.Tensor): The data of shape [N, L, C]. It is assumed that the times are in the first index of the
            data channels.
        fill_start (bool): Whether to fill initial nans with first available value.
        fill_end (bool): Whether to fill end nans with last available value.
        fill_remaining (str): Method to fill remaining values. These are the series channels that have are given no
            value for the duration of the time-series. Currently the only implemented method fills with zero.

    Returns:
        torch.Tensor: The tensor with linearly interpolated values.
    """
    # Repeat times along channels dim to associate each data-point with a time.
    full_times = data[:, :, [0]].repeat(1, 1, data.size(2))
    nan_mask = torch.isnan(data)

    # Times at positions only when the variable is observed
    full_nan_times = full_times.clone()
    full_nan_times[nan_mask] = float('nan')

    # Stack the data and times to get the bfill and fills in one call to torch_ffill
    data_to_ffill = torch.cat((data, data.flip(1), full_nan_times, full_nan_times.flip(1)), 2)
    out = torch_ffill(data_to_ffill)
    data_ffill, data_bfill, times_ffill, times_bfill = torch.chunk(out, 4, 2)
    data_bfill, times_bfill = data_bfill.flip(1), times_bfill.flip(1)

    # The linearly interpolated data to impute.
    impute = data_ffill + (full_times - times_ffill) * ((data_bfill - data_ffill) / (times_bfill - times_ffill))

    # Perform the imputation
    data[nan_mask] = impute[nan_mask]

    # Deal with start and end nans
    if fill_end:
        end_mask = torch.isnan(data)
        data[end_mask] = data_ffill[end_mask]
    if fill_start:
        start_mask = torch.isnan(data)
        data[start_mask] = data_bfill[start_mask]
    if fill_remaining == 'zeros':
        remaining_mask = torch.isnan(data)
        data[remaining_mask] = 0.

    return data



