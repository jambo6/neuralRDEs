"""
setup.py
================================================
Various method used in setting up and running sacred experiments.
"""
import os
import sys; sys.path.append('../../')
import traceback
import logging
import shutil
import random
import numpy as np
import torch
from pprint import pprint
from sacred.observers import FileStorageObserver
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm
from .extract import get_hyperparams, load_configs
from .functions import timeit
from .functions import load_pickle, save_pickle


def set_logger(ex, level='WARNING'):
    """ Sets the sacred experiment logger. """
    logger = logging.getLogger('logger')
    logger.setLevel(getattr(logging, level))
    ex.logger = logger


def create_fso(ex, directory, remove_folder=False):
    """
    Creates a file storage observer for a given experiment in the specified directory.

    Check sacred docs for a full explanation but this just sets up the folder to save the information from the runs of
    the given experiment.

    NOTE: This is currently setup to delete/replace the specified folder if it already exists. This should be changed if
    it is not the desired behaviour.

    Args:
        ex (sacred.Experiment): The sacred experiment to be tracked.
        directory (str): The directory to 'watch' the experiment and save information to.

    Returns:
        None
    """
    if remove_folder:
        if os.path.exists(directory) and os.path.isdir(directory):
            shutil.rmtree(directory)
    ex.observers.append(FileStorageObserver(directory))


@timeit
def basic_gridsearch(ex,
                     grid,
                     param_drops=None,
                     verbose=2,
                     handle_completed_state=True,
                     logging='WARNING'):
    """Basic gridsearch for a sacred experiment.

    Given an experiment and a parameter grid, this will iterate over all possible combinations of parameters specified
    in the grid. In an iteration, the experiment configuration is updated and the experiment is run.

    Args:
        ex (sacred.Experiment): A sacred experiment.
        grid (dict, list): Parameter grid, analogous setup to as with sklearn gridsearches. If a list is specified then
            it will assume it is from a restart and continue as normal.
        param_drops (list): A list of functions that take one argument that acts on params and drops certain options
            from being run.
        verbose (int): Output verbosity level.
        handle_completed_state (bool): Set True to examine whether the parameters have already been run and additionally
            if that run was marked as completed. NOTE: You need to be careful with this option. It will only work if the
            completion state is being set at the end of an experiment run. If it is not being set then it will always
            delete and rerun.
        logging (str): The logging level.
        seed (int): Random state to set. Set None for a random seed, this will be stored in the run config.

    Examples:
        Param drops
        ---------------
        The param drops argument is used to remove certain parameter combinations from the parameter grid that would
        otherwise have been run. For example:
        >>> param_drops = [
        >>>    lambda params: (params['model_type'] in ['rnn', 'gru']) and (params['depth'] > 1)
        >>> ]
        >>> basic_gridsearch(ex, grid, param_drops=param_drops)
        will run the gridesarch as normal, but in any instances where model_type was 'rnn' or 'gru' and 'depth' was > 1
        the gridsearch will skip these options.

    Returns:
        None
    """
    # Set logging
    set_logger(ex, level=logging)

    # Setup the grid
    assert any([isinstance(grid, dict), isinstance(grid, list)])
    if isinstance(grid, dict):
        param_grid = list(ParameterGrid(grid))
    else:
        param_grid = grid

    # Perform the param drops
    if param_drops is not None:
        param_grid = [
            p for p in param_grid if not any([drop_bool(p) for drop_bool in param_drops])
        ]

    # Get num iters
    grid_len = len(param_grid)

    for i, params in tqdm(enumerate(param_grid)):
        # Print info
        if verbose > 0:
            print('\n\n\nCONFIGURATION {} of {}\n'.format(i + 1, grid_len) + '-' * 100)
            pprint(params)
            print('-' * 100)

        # Set the random seed
        handle_seed(params)

        # Skip if done
        if handle_completed_state:
            if check_run_existence(ex, params):
                continue

        # Update the ingredient configuration
        params = update_ingredient_params(ex, params)

        # Update configuration and run
        try:
            # Update with hyperopt if specified
            # This is in the try except in case hyperparams cannot be found, error is caught and printed
            if 'hyperopt_metric' in params:
                metric = params['hyperopt_metric']
                assert isinstance(metric, str), "Hyperopt metric must be a string."
                hyperparams = get_hyperparams(ex, params['model_type'], metric=metric)
                params.update(hyperparams)

            # Run and mark completed if successful
            ex.run(config_updates=params, info={})
            save_pickle(True, ex.current_run.save_dir + '/other/completion_state.pkl')
        except Exception as e:
            handle_error(ex.current_run, e, print_error=True)


def purge_uncompleted_runs(save_dir):
    """Removes runs that do not have completion state marked as True.

    Note: This will also remove any runs for which the completion state does not exist at all, so use with care.
    """
    for f in os.listdir(save_dir):
        loc = save_dir + '/' + f + '/other'
        if not os.path.exists(loc + '/completion_state.pkl'):
            shutil.rmtree(loc + '/')
        else:
            completed_state = load_pickle(loc + '/completion_state.pkl')
            if not completed_state:
                shutil.rmtree(loc + '/')


def check_run_existence(ex, params):
    """ Checks if the parameters for a given run already exist. """
    save_dir = ex.observers[0].basedir

    # Skip if this is a fresh install
    if not os.path.exists(save_dir):
        return False

    # If directory exists, load in all configurations
    if os.path.exists(save_dir):
        configs, run_nums = load_configs(save_dir)
    else:
        return False

    # Remove additional keys that are made by sacred
    for i in range(len(configs)):
        c = nested_dict_to_dunder(configs[i])
        configs[i] = {k: v for k, v in c.items() if k in params.keys()}

    # If the model was completed, then do not run it again. If it was not completed then remove it and start afresh.
    completed_bool = False
    if params in configs:
        print('Skipping {} as already completed.'.format(params))
        completed_bool = True

    return completed_bool


def nested_dict_to_dunder(d):
    """ Converts a dict with 1-level nested keys to a dunder dictionary. """
    d_copy = d.copy()
    for key, value in d.items():
        if isinstance(d[key], dict):
            for k_inner, v_inner in d[key].items():
                d_copy[key + '__' + k_inner] = v_inner
    return d_copy


def set_completion_state(_run, state):
    """ Sets the completion state as True or False for help with reruns. """
    assert isinstance(state, bool)
    save_pickle(state, _run.save_dir + '/other/completion_state.pkl')


def handle_error(_run, e, print_error=True):
    """Handling of the error if ex.main fails to complete successfully.

    Args:
        _run (sacred._run): A sacred run object.
        e (Exception): An error message.
        print_error (bool): Set True to print the error to the console.

    Returns:
        None: Saves the error to the sacred metrics and notes the error in a text file in `/other/`.
    """
    if _run is not None:
        # Save error information
        err_name = 'error'
        _run.log_scalar(err_name, str(e))

        # Make other folder if not exists then save error as a txt file
        other_loc = _run.save_dir + '/other'
        if not os.path.exists(other_loc):
            os.makedirs(other_loc)
        with open(other_loc + '/{}.txt'.format(err_name), 'w') as f:
            f.write(str(e))

    # Print error to console
    if print_error:
        print(traceback.format_exc())


def handle_seed(params):
    """ Sets the seed for anything with randomness. """
    if 'seed' in params:
        seed = params['seed']
    else:
        seed = np.random.randint(2**16)
        print('Setting random seed {} as unspecified.'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    return seed


def attach_states(_run, save_dir):
    """ Attaches the save directory to the run for use in the experiment later on, and gives a completion state. """
    _run.save_dir = save_dir + '/' + _run._id
    save_pickle(False, _run.save_dir + '/other/completion_state.pkl')


def get_type_params(params, type):
    """
    Extracts the parameters of a specific type when the format is given like with sklearn pipelines. That is:
        params = {
            'type1__param_name': [params1],
            'type2__param_name': [params2]
        }
    """
    type_params = {
        __name.split('__')[1]: param_list for __name, param_list in params.items() if __name.split('__')[0] == type
    }
    return type_params


def update_ingredient_params(ex, params):
    """ Updates various configuration parameters """
    ingredients = {i.path: i for i in ex.ingredients}

    # Assert all __ args have an associated ingredient name.
    dunders = [k.split('__')[0] for k in params.keys() if '__' in k]
    for d in dunders:
        assert d in ingredients.keys(), "{} does not have an associated ingredient. Must be one of: {}." \
                                        "".format(d, ingredients.keys())

        # Update the params
        update_params = get_type_params(params, d)
        if update_params != {}:
            ingredients[d].add_config(**update_params)

        for update_param in update_params.keys():
            del params[d + '__' + update_param]

    return params


def config_parallelisation(config, igpu, ngpus):
    """Sets up parallelisation over the configurations if specified.

    Given a configuration with multiple run combinations, this function converts to a list of configurations and takes
    the configuration of that list according to config_idx. This allows us to run a command such as:
        `parallel -j 4 --bar python uea.py -idx {} ::: {0.50}`
    which will run the uea experiment and grab a different bit of the configuration each time. Provided the total number
    of configurations is < 50 then this will complete everything (just ensure that {i..j} covers more than the config.

    Args:
        config (dict): Dictionary of all configurations.
        gpu (int): The index of the GPU to run on.
        igpu (int): The unique location identifier of the GPU as a position in ngpus.
        ngpus (int): The total number of GPUs that are being run on.

    Returns:
    """
    if ngpus < 2:
        pass
    elif ngpus >= 2:
        config_list = list(ParameterGrid(param_grid=config))
        config = [config_list[i] for i in list(range(igpu, len(config_list), ngpus))]
    return config


def handle_resume(save_dir, resume, remove):
    """ Resumes an experiment only if the resume flag is hit to ensure the user knows what is happening. """
    if os.path.exists(save_dir):
        if remove:
            shutil.rmtree(save_dir)
        if not resume:
            raise Exception("Runs already exist at: {}. \nPass the resume (-r) flag to confirm you are aware of this "
                            "and wish to proceed. Note, you should run purge on this first unless runs are currently"
                            "in the process of completing."
                            .format(save_dir))


