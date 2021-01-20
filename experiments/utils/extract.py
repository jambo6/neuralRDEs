"""
extract.py
===================================
Method for extracting data from runs.
"""
import os
from pathlib import Path
import warnings
import pandas as pd
from .functions import load_json


def create_run_frame(ex_dir):
    """Creates a DataFrame from the run saves.

    Args:
        ex_dir (str): The experiment directory.

    Returns:
        pd.DataFrame: A pandas dataframe containing all results from the run.
    """
    assert os.path.exists(ex_dir), "No run folder exists at: {}".format(ex_dir)
    run_nums = get_run_nums(ex_dir)

    frames = []
    for run_num in run_nums:
        loc = ex_dir + '/' + run_num
        try:
            config = extract_config(loc)
            metrics = extract_metrics(loc)
            # Expand inner dicts
            config = convert_dunder_config(config)
            # Some configs break if we dont do this
            config = {str(k): str(v) for k, v in config.items()}
            # Create a config and metrics frame and concat them
            df_config = pd.DataFrame.from_dict(config, orient='index').T
            df_metrics = pd.DataFrame.from_dict(metrics, orient='index').T
            df = pd.concat([df_config, df_metrics], axis=1)
            df.index = [int(run_num)]
            frames.append(df)
        except Exception as e:
            print('Could not load metrics at: {}. Failed with error:\n\t"{}"'.format(loc, e))

    # Concat for a full frame
    df = pd.concat(frames, sort=False)
    df.sort_index(inplace=True)

    # Make numeric cols when possible
    df = df.apply(lambda x: pd.to_numeric(x, errors='ignore'), axis=1)

    return df


def get_run_nums(ex_dir):
    """ Extracts the run folder names from an experiment directory. """
    not_in = ['_sources', '.ipynb_checkpoints']
    run_nums = [x for x in os.listdir(ex_dir) if x not in not_in]
    return run_nums


def extract_config(loc):
    """ Extracts the configuration from the directory. """
    config = load_json(loc + '/config.json')
    return config


def extract_metrics(loc):
    """ Extracts the metrics from the directory. """
    metrics = load_json(loc + '/metrics.json')

    # Strip of non-necessary entries
    metrics = {key: value['values'] for key, value in metrics.items()}

    return metrics


def convert_dunder_config(cfg):
    """ Converts an ingredients config into a single config with dunder columns. """
    cfg_new = cfg.copy()
    for k, v in cfg.items():
        if isinstance(v, dict):
            for k_inner, v_inner in v.items():
                cfg_new[k + '__' + k_inner] = v_inner
            del cfg_new[k]
    return cfg_new


def load_configs(ex_dir):
    """ Loads all configuration files into a list from the given experiment directory. """
    configs = []
    run_nums = get_run_nums(ex_dir)
    for run_num in run_nums:
        loc = ex_dir + '/' + run_num
        try:
            configs.append(extract_config(loc))
        except:
            warnings.warn("Cannot load config in {}. Consider deleting.".format(loc), Warning)
    return configs, run_nums


def get_hyperparams(ex,
                    model_type,
                    metric='loss',
                    lower_is_better=None
                    ):
    """Extracts the best hyper-parameters from the saved runs.

    This assumes the hyperopt values to load are in the same parent directory as the experiment but in a hyperopt
    folder. This also assumes the keys are ['hidden_dim', 'hidden_hidden_dim', 'num_layers'].

    Args:
        ex (sacred.Experiment): The current sacred experiment object.
        model_type (str): The model to extract the hyperparams for.
        metric (str): The metric to use i.e. 'acc', 'loss'.
        lower_is_better (bool): Whether to use large or small value of the metric. Leave as None to imply it from the
            metric string.

    Returns:
        list: The hyperparameter keys to extract.
    """
    # Setup
    save_dir = str(Path(ex.observers[0].basedir).parent / 'hyperopt')
    save_dir = save_dir + '-odernn' if model_type == 'odernn_folded' else save_dir
    assert os.path.exists(save_dir), "Cannot find hyper-parameter run at {}.".format(save_dir)

    # Load the results
    frame = create_run_frame(save_dir)
    val_metric = metric + '.val'    # Only consider val as a metric
    assert model_type in frame['model_type'].unique(), "Search not yet run for model_type:{}.".format(model_type)
    assert val_metric in frame.columns, "Metric not found, try one of {}.".format([x for x in frame.columns if '.val' in x])
    model_results = frame[frame['model_type'] == model_type]

    # Sort ascending
    if not isinstance(lower_is_better, bool):
        if val_metric == 'loss.val':
            lower_is_better = True
        elif val_metric == 'acc.val':
            lower_is_better = False
        else:
            raise NotImplementedError("Can't imply `lower_is_better`, specify manually.")

    # Max over the val scores
    model_results = model_results[~model_results[val_metric].isna()]
    best_model = model_results.sort_values(val_metric, ascending=lower_is_better).iloc[0]

    # Get params
    hyperparam_keys = ['hidden_dim', 'hidden_hidden_multiplier', 'num_layers', 'num_params']
    hyperparams = {key: int(best_model[key]) for key in hyperparam_keys}

    # Make num_params -> tune_params
    # hyperparams['tune_params'] = hyperparams['num_params']
    del hyperparams['num_params']

    return hyperparams


if __name__ == '__main__':
    save_dir = '../models/test/UEA/BasicMotions'
    partition = create_run_frame(save_dir + '/partition')
