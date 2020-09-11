"""
runs.py
===============================
Contains scripts to run different experiments.
"""
import os
import shutil
from pathlib import Path
import torch
from utils.functions import load_pickle
THIS_FOLDER = str(Path(__file__).resolve().parents[0])


def run(folder, dataset, config_name, gpu_idxs=False, test=False, purge=True):
    """Runs for standard experiments (UEA, SpeechCommands, TSR).

    Args:
        folder (str): The folder the data is contained in (in data/processed).
        dataset (str): The dataset name in the folder.
        config_name (str): The name of the corresponding config entry.
        n_jobs (int): Number of parallel processes. Set to -1 to work out the number of GPUs and run over that many.
        test (bool): Set True to enter test mode (runs for 1 epoch).
        purge (bool): Set True to remove any non completed runs from the save folder.
        gpu_nums (list): Provide a list of GPU indexes to use if you do not wish to do all.

    Returns:
        None
    """
    assert folder in ['SpeechCommands', 'UEA', 'TSR'], "Not implemented for folder:{}.".format(folder)
    assert any([isinstance(gpu_idxs, list), gpu_idxs == False]), "GPU indexes must be a list or False."
    dot_py_string = 'tsr' if folder == 'TSR' else 'classification'
    # Remove non-completed runs
    if purge:
        purge_all(folder, dataset, config_name)
    # Python command to run
    test_string = '-t' if test else ''
    python_string = 'python {}/{}.py -f {} -ds {} -c {} {}' \
                    ''.format(THIS_FOLDER, dot_py_string, folder, dataset, config_name, test_string)
    # Build the GNU parallel command
    if isinstance(gpu_idxs, list):
        assert len(gpu_idxs) < get_num_gpus()
        command = _set_gpu_run_command(python_string, gpu_idxs)
    else:
        command = python_string
    # Print to console and run the command
    print(command)
    os.system(command)


def _set_gpu_run_command(python_string, gpu_idxs):
    """Command to run a configuration over all available GPUs.

    This required some careful handling, we need to create a number of distinct commands equal to the number of
    available GPUs. Each command handles 1/num_gpus of the config entries. This way we only ever have one process
    running on each GPU. If it is not done this way then this is not enforced.

    Args:
        python_string (str): The python command string.
        gpu_idxs (int): Indexes of GPUs to run on.

    Returns:
        None
    """
    # Pass to the gpu argument both an index and an identifier
    num_gpus = len(gpu_idxs)
    gpu_arg = ' '.join([str(x) for x in gpu_idxs])
    igpu_arg = ' '.join([str(i) for i in range(num_gpus)])
    # Build and run the parallel command.
    command = "parallel -j {} --link --bar '{} -gpu {{1}} -igpu {{2}} -ngpus {}' ::: {} ::: {}" \
              "".format(num_gpus, python_string, num_gpus, gpu_arg, igpu_arg)
    return command


def purge_all(folder, dataset, config):
    """ Remove all experiments that do not have a marked completed state. """
    folder = './models/{}/{}/{}'.format(folder, dataset, config)
    if not os.path.exists(folder):
        return None
    for run_num in [x for x in os.listdir(folder) if not x.startswith('_')]:
        state_path = folder + '/{}/other/completion_state.pkl'.format(run_num)
        remove_func = lambda: shutil.rmtree(folder + '/{}/'.format(run_num))
        if not os.path.exists(state_path):
            remove_func()
        elif load_pickle(state_path) is False:
            remove_func()


def get_num_gpus():
    """ Get the number of GPUs so we can split a run over them. """
    return torch.cuda.device_count()


def run_all(gpu_idxs=None):
    # run args
    args = [('UEA', 'EigenWorms'), ('TSR', 'BIDMC32RR'), ('TSR', 'BIDMC32HR'), ('TSR', 'BIDMC32SpO2')]

    # Cheeky for loop
    for i, (folder, dataset) in enumerate(args):
        # Optimize hyperparameters
        run(folder, dataset, 'hyperopt')
        # Main run with optimized hyperparameters
        run(folder, dataset, 'main')



