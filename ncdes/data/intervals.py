"""
intervals.py
=============================
A collection of methods to return intervals over which to compute signatures.
"""
import torch
from torch.utils.data.sampler import SequentialSampler, BatchSampler
from torch.utils.data import DataLoader


class BatchIntervalSampler(BatchSampler):
    """A copy of PyTorch's BatchSampler with some additional code to handle interval sampling.

    This allows for a method to sample from the batch and another method to sample the intervals. A standard default
    of a SequentialSampler is used to sample from the batch if None is specified.
    """
    def __init__(self,
                 n_samples,
                 interval_sampler,
                 batch_size,
                 sampler=None,
                 include_pvar_batch=False,
                 drop_last=False):
        self.n_samples = n_samples
        self.interval_sampler = interval_sampler
        self.batch_size = batch_size
        self.include_pvar_batch = include_pvar_batch
        self.drop_last = drop_last

        # Set batch sampling to be sequential if not set
        if sampler is None:
            data_source = torch.arange(n_samples)
            self.sampler = SequentialSampler(data_source=data_source)

    def convert_to_interval_batch(self, batch, intervals):
        if intervals is not None:
            if self.include_pvar_batch:
                batch = [[batch, intervals], [batch, [intervals[i] for i in range(0, len(intervals), 2)]]]
            else:
                batch = [[batch, intervals]]
        return batch

    def __iter__(self):
        batch = []
        intervals = self.interval_sampler() if self.interval_sampler is not None else None
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                batch = self.convert_to_interval_batch(batch, intervals)
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            batch = self.convert_to_interval_batch(batch, intervals)
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size


class FixedIntervalSampler:
    """Samples using a fixed step size until no more can be computed.

    Given raw data of shape [N, L, C], and a step size S, this method first extracts the intervals
        [[0, S], ..., [S, 2 * S], [S, 3 * S], ...]
    until the final interval is less than size S. The `__iter__` method simply gets the next batch_idx along with this
    fixed window partition.
    """
    def __init__(self, ds_length, step, from_start=True, include_end=False):
        """
        Args:
            ds_length (int): The length of the dataset (number of time-points).
            step (int): The step size to take.
            from_start (bool): See `fixed_step_intervals` docs.
            include_end (bool): See `fixed_step_intervals` docs.
        """
        assert step <= ds_length, "Step cannot be larger than the length of the dataset. Dataset length is {} given " \
                                  "step size is {}.".format(ds_length, step)
        self.ds_length = ds_length
        self.step = step
        self.from_start = from_start
        self.include_end = include_end

        # Setup the intervals
        self.intervals = fixed_step_partition(ds_length, step, from_start=from_start, include_end=include_end)

        # Get the indexes of the end points, start points and knots
        self.start_idxs = [i[0] for i in self.intervals]
        self.end_idxs = [i[-1] - 1 for i in self.intervals]
        self.knot_idxs = [self.intervals[0][0]] + self.end_idxs

    def __call__(self):
        return self.intervals


class RandomSampler:
    """Samples the intervals in a semi-random fashion by choosing random indexes but approximately keeping to the step.

    Suppose we have a dataset of length L, given a `min_step_frac` and a `max_step_frac` this will run over the dataset
    with steps inside `[min_step_frac * L, max_step_frac * L]`. So the steps are random but it should on average keep
    approximately to the step size.
    """
    def __init__(self, ds_length, step, min_step_frac=0.2, max_step_frac=2):
        assert step >= 2, "Random sampler does not work for `step < 2`."
        self.ds_length = ds_length
        self.step = step
        self.min_step_frac = min_step_frac
        self.max_step_frac = max_step_frac
        self.min_step = int(step * min_step_frac)
        self.max_step = int(step * max_step_frac)

    def __call__(self):
        # Little hack to make sure the step to the end is ok
        end_ok = False
        while not end_ok:
            # Make random steps
            randints = torch.randint(self.min_step + 1, self.max_step, (self.ds_length,))
            steps = randints.cumsum(0)
            # Remove anything > ds_len
            steps = steps[steps <= self.ds_length]
            steps[-1] = self.ds_length
            # Make the intervals
            start_idx = torch.cat([torch.randint(0, 1, (1,)), steps[:-1]])
            end_idx = steps + 1
            intervals = torch.cat((start_idx.view(-1, 1), end_idx.view(-1, 1)), 1).tolist()
            # Check the final step size is within the specified bound.
            final_step_size = intervals[-1][1] - intervals[-1][0]
            if all([final_step_size >= self.min_step + 1, final_step_size <= self.max_step]):
                end_ok = True
        return intervals


def fixed_step_partition(length, step, from_start=True, include_end=False):
    """Computes sub-interval index positions over some length with a fixed step.

    Args:
        length (int): The length of the full interval.
        step (int): The step size for each sub-interval.
        from_start (bool): Set True to index increasing from 0, False to index backwards from length + 1.
        include_end (bool): Set True to include the end (or start if from_start=False) of the interval even if it is
            shorter than `step`.

    Returns:
        list: A list of increasing intervals [[i_1, i_2], [i_3, i_4], ...]
    """
    if from_start:
        if include_end:
            # Go to length - 1 to ensure the final interval will be at least length 2 (else not a valid interval)
            intervals = [[i, i + step + 1] for i in range(0, length, step)]
            intervals[-1][-1] = length + 1
        else:
            intervals = [[i, i + step + 1] for i in range(0, length - step, step)]
    else:
        if include_end:
            intervals = [[i - step - 1, i] for i in range(length + 1, 1, -step)][::-1]
            intervals[0][0] = 0
        else:
            intervals = [[i - step - 1, i] for i in range(length + 1, step, -step)][::-1]
    return intervals


def create_interval_dataloader(dataset, sampler, batch_size):
    """Creates a NRDE dataloader given a dataset, sampler, and batch_size.

    This is needed to setup the correct way to sample intervals alongside batches.

    Args:
        dataset (torch.Dataset): Standard torch dataset.
        sampler (IntervalSampler): An interval sampler as defined in this file.
        batch_size (int): The size of the batch.

    Returns:
        torch.DataLoader: A dataloader that also samples the intervals.

    """
    def collate_fn(batch):
        return batch[0]
    # Create a sampler
    batch_sampler = BatchIntervalSampler(dataset.n_samples, sampler, batch_size)
    dataloader = DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=collate_fn)
    return dataloader