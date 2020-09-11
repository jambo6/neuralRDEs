"""
dataset.py
=============================
Dataset classes for use with the rough differential equations setup.
"""
import torch
from torch.utils.data import Dataset
import signatory


class Path(Dataset):
    """Dataset that abstractly defines a path via signatures.

    We assume the definition of a path to be an object that you can query with and interval, and have a signature
    over the interval returned. This class takes the more classical definition of a path and converts it into an object
    that we can query with intervals to return logsignatures.

    Example:
        >>> data = torch.randn(10, 50, 3)
        >>> dataset = Path(data, depth=3)
        >>> dataset[0, [1, 3]]      # Returns the signature of data[0, 1:3]
        >>> dataset[3:5, [[0, 10], [10, 20]]]   # Returns the signatures of two intervals of batch 3:5
    """
    def __init__(self,
                 data,
                 depth=3,
                 logsig=True,
                 device=None):
        """
        Args:
            data (torch.Tensor): Standard [N, L, C] format data.
            depth (int): Signature truncation depth.
            logsig (bool): Set False to return the signature rather than the logsignature.
            device (torch.device): This will pass the data to `device` during signature computation, then pass back to
                the original device after.
        """
        self.depth = depth
        self.logsig = logsig

        # Setup Path. This computes all the necessary signatures.
        original_device = data.device
        data.to(device) if device is not None else None
        self.path_class = signatory.Path(data, depth)
        data.to(original_device)

        # The function for extracting the signature.
        self.signature = self.path_class.__getattribute__('logsignature' if logsig else 'signature')

        # Pass over some functionality
        self.size = self.path_class.size
        self.logsig_dim = self.path_class.logsignature_channels()

    def __getitem__(self, idx):
        """Index with a batch dimension and a valid interval. Returns the logsignatures of the batch over the interval.

        Indexing must be of the form self[batch_idxs, interval] where interval is a list [i, j] such that [i, j] is a
        valid interval for which to compute the logsignature over. The indexing [batch_idxs, [i, j]] will return the
        logsignature[i:j] of the specified batch_idxs. Note this means we require j > i + 1.

        Indexing is also allowed in the form self[batch_idx, [[i_1, i_2], [i_3, i_4], ...]. The aim is for this to
        emulate standard tensor behaviour but instead of indexes, we provide intervals.

        Args:
            idx: Indexer of the form [batch_idxs, [i, j]].

        Returns:
            torch.Tensor: Of shape [batch_idxs, logsig_channels].
        """
        # Ensure we have specified batch and interval
        assert len(idx) == 2, 'idx must be of the form [batch_idx, interval]. Got {}.'.format(idx)
        assert isinstance(idx[1], list), 'Interval must be of type:list got type:{}.'.format(type(idx[1]))

        # Open
        batch_idx, intervals = idx

        # Slightly different computations if multiple intervals are specified
        if not any(isinstance(l, list) for l in intervals):
            signatures = self.signature(*intervals)[batch_idx]
        else:
            signatures = torch.stack([self.signature(*interval)[batch_idx] for interval in intervals], dim=1)

        return signatures

    def __len__(self):
        return self.size(0)


class FlexibleCDEDataset:
    """A simple extension of SignatureDataset to handle labels.

    Dataset for working with NeuralCDEs (or logsig-rnns) that is flexible in the sense of allowing the choice of
    time-series partition to change. This dataset differs from 'FixedCDEDatset' in that it

    Note:
        If the problem is classification, the response should be of shape (N,) as normal.

        If however the problem is regression, we assume that a dataloader with an interval sampler is going to be
        iterated over. The returned indexes will be of the form [batch, [[i_1, i_2], [i_3, i_4], ...] and we assume that
        the labels to be predicted are at [batch, [i_2, i_4, ...]]. That is, the labels are given specifically such that
        the label to predict aligns with the data index that is used to predict it. If a different behaviour is wanted,
        a new method will have to introduced to select the correct labels.

        If the problem is regression, but we care only about the terminal value, then having the labels be of shape (N,)
        where each index denotes the terminal value of the batch, this will be fine.
    """
    def __init__(self, controls, responses, depth=2):
        """
        Args:
            controls (torch.Tensor): The control path of shape [N, L, C_control].
            responses (torch.Tensor): The response path of shape [N, L, C_response] (if regression) or shape (N,) if
                classification.
            depth (int): The truncation depth of the logsignature. Set False to disable the signature computation.
        """
        # Hold onto the initial values
        self.initial_values = controls[:, 0, :]
        self.initial_dim = self.initial_values.size(-1)

        # Repsonses and params
        self.responses = responses
        self.depth = depth

        # Ready the signatory path object
        self.signatures = Path(controls, depth=depth, logsig=True, device=controls.device)

        # Pass some functions
        self.n_samples = self.signatures.size(0)
        self.n_intervals = self.signatures.size()[1]
        self.ds_length = self.n_intervals + 1
        self.logsig_dim = self.signatures.logsig_dim
        self.input_dim = self.logsig_dim
        self.controls = self.signatures     # As the signatures are now the controls

    def __getitem__(self, idx):
        """Returns the signatures over the intervals along the batch indexes, and the labels along batch.

        idx must be specified analogously to LogSignatureDataset. That is, index must be of the form `[batch_idx,
        intervals]` where intervals is a list denoting a valid interval `[i, j]` or list of lists `[[i_1, i_2], [i_2,
        i_3], ...]`. Returns the signatures over the intervals for the given batch along with the labels for the given
        batch.

        Args:
            idx: See `src.data.utils.dataset.SignatureDataset`.

        Returns:
            torch.Tensor, torch.Tensor: Batched signatures over intervals, batched labels.
        """
        # First extract the batch and interval components
        assert len(idx) == 2, 'idx must be of the form [batch_idx, intervals]. Got {}.'.format(idx)
        batch_idx, intervals = idx

        # Get the signatures
        initial_values = self.initial_values[batch_idx]
        signatures = self.signatures[batch_idx, intervals]

        # Return the response at the end of each interval if regression, else the final class
        if self.responses.dim() < 3:
            response_out = self.responses[batch_idx]
        else:
            end_idxs = [i[-1] - 1 for i in intervals]
            response_out = self.responses[batch_idx, end_idxs]

        return (initial_values, signatures), response_out

    def __len__(self):
        return len(self.signatures)


class FixedCDEDataset:
    """Dataset for working with NRDEs where the intervals we compute over is assumed not to change.

    This differs from the FlexibleCDEDataset as it does not allow the sampling intervals to change. The signatures are
    pre-computed over the intervals and the dataset is fixed. For cases where we do not care about the intervals being
    allowed to vary, this is significantly faster than FlexibleCDEDataset as it does not require computation of the
    signatures from signatories stored signature pieces each time we index the dataset.
    """
    def __init__(self, controls, responses, sampler=None, depth=2, response_in_initial=False):
        """
        Args:
            controls (torch.Tensor): The control path of shape [N, L, C_control].
            responses (torch.Tensor): The response path of shape [N, L, C_response] (if regression) or shape (N,) if
                classification.
            sampler (IntervalSampler): An initialised sampler from src.data.utils.intervals.
            depth (int): The depth of the log-signature to truncate at.
            response_in_initial (bool): If a continuous regression problem, set True to include the response at time
                t=0 to the initial values.
        """
        # Hold onto the initial values
        if response_in_initial:
            assert responses.dim() == 3, "Can only put the response in the IC if dim == 3."
            self.initial_values = torch.cat((controls[:, 0, :], responses[:, 0, :]), dim=1)
        else:
            self.initial_values = controls[:, 0, :]
        self.initial_dim = self.initial_values.size(-1)

        # Setup intervals
        intervals, knot_idxs = sampler.intervals, sampler.knot_idxs
        self.sampler = None

        # Responses and params
        self.responses = responses[:, knot_idxs] if responses.dim() == 3 else responses
        self.depth = depth

        # Compute the signatures
        self.signatures = torch.stack(
            [signatory.logsignature(controls[:, i[0]:i[1]], depth=depth) for i in intervals], dim=1
        )

        # Some params
        self.n_samples = self.signatures.size(0)
        self.ds_length = self.signatures.size(1)
        self.input_dim = self.signatures.size(2)
        self.controls = self.signatures     # As the signatures are now the controls

        # Add some functionality
        self.size = self.signatures.size()

    def __getitem__(self, idx):
        return (self.initial_values[idx], self.signatures[idx]), self.responses[idx]

    def __len__(self):
        return len(self.responses)


class SubsampleDataset:
    """Subsamples the dataset enabling use of src.data.intervals sampling methods.

    This class was built with a desire to use the same src.data.intervals sampling methods for non-logsig networks. For
    example, to compare against a vanilla RNN the features to predict y_{t_i} are simply x_{t_i} as opposed to
    logsig_[x_{t_{i-1}}, x_{t_i}]. This class simply extracts the corresponding values from an interval sampler.
    """
    def __init__(self, controls, responses):
        """
        Args:
            controls (torch.Tensor): The control path of shape [N, L, C_control].
            responses (torch.Tensor): The response path of shape [N, L, C_response] (if regression) or shape (N,) if
                classification.
        """
        self.controls = controls
        self.responses = responses

        # Pass some functions
        self.n_samples = self.controls.size(0)
        self.ds_length = self.controls.size(-2)
        self.input_dim = self.controls.size(-1)

    def __getitem__(self, idx):
        # First extract the batch and interval components
        assert len(idx) == 2, 'idx must be of the form [batch_idx, intervals]. Got {}.'.format(idx)
        batch_idx, intervals = idx

        # Subsampled indexes
        idxs = [intervals[0][0]] + [i[-1] - 1 for i in intervals]
        inputs = self.controls[batch_idx]
        inputs = inputs[:, idxs]

        # Response dependent on whether it is classification
        if self.responses.dim() == 3:
            outputs = self.responses[batch_idx, idxs]
        else:
            outputs = self.responses[batch_idx]

        return inputs, outputs

    def __len__(self):
        return len(self.controls)

