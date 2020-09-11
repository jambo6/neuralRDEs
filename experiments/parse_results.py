"""
parse_results.py
======================================
For parsing the output of the experiments.
"""
from pathlib import Path
import argparse
from experiments.utils.extract import create_run_frame
ROOT_DIR = str(Path(__file__).resolve().parents[1])


def parse_results(folder, dataset, config_name, sort_key='val', average_over=['depth', 'step'], havok=False,
                  print_frame=True, pretty_std=True):
    """Converts the result to a DataFrame and prints important information to the console.

    Args:
        folder (str): The main folder containing the results.
        dataset (str): The dataset subfolder.
        config_name (str): The name of the configuration.
        sort_key (str): The metric sort key ('test', or 'val').
        average_over (list): If a list will groupby the selected columns and take a mean.
        havok (str): Whether to prepend havok to the ex_dir string.
        print_frame (bool): Set True to print out the frame.
        pretty_std (bool): Set True to print mean +/- std, else will return just the mean frame.

    Returns:
        None: Just prints the DataFrame formatted results to the console.
    """
    # Create the frame
    havok_str = 'havok/' if havok else ''
    ex_dir = ROOT_DIR + '/experiments/models/{}{}/{}/{}'.format(havok_str, folder, dataset, config_name)
    frame = create_run_frame(ex_dir)
    # Assume a metric
    metric = 'acc' if 'acc.val' in frame.columns else 'loss'
    ascending = True if metric == 'loss' else False
    standard_columns = [
        'step', 'depth', '{}.test'.format(metric), '{}.train'.format(metric), '{}.val'.format(metric),
        'num_params', 'true_hidden_dim', 'data__sampler_name', 'memory_usage', 'elapsed_time'
    ]
    frame = frame[standard_columns]
    # Mean and stddevs if average over is specified.
    if average_over:
        stds = frame.groupby(['depth', 'step'], as_index=True).std()
        means = frame.groupby(['depth', 'step'], as_index=True).mean().round(3)
        # sort
        frame = means.sort_values('{}.{}'.format(metric, sort_key), ascending=ascending)
        if pretty_std:
            stds = stds.loc[frame.index]
            # means +/- stds for acc/loss cols
            std_cols = [metric + '.' + x for x in ['test', 'val', 'train']]
            frame[std_cols] = means[std_cols].astype(str) + ' +/- ' + stds[std_cols].round(3).astype(str)
    frame = frame.sort_values('{}.{}'.format(metric, sort_key), ascending=ascending)
    if print_frame:
        print(frame)
    return frame


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', help='Folder that holds the data.', type=str)
    parser.add_argument('dataset', help='The name of the dataset to run.', type=str)
    parser.add_argument('config', help='The configuration key.', type=str)
    parser.add_argument('-s', '--sort_key', help='Sort key (usually test or val)', default='val', type=str)
    parser.add_argument('-avg', '--average_over', help='The groupby columns.', action='store_true')
    args = parser.parse_args()

    parse_results(args.folder, args.dataset, args.config, sort_key=args.sort_key, average_over=args.average_over)
