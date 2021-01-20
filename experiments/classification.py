import argparse
from sacred import Experiment
from utils.setup import create_fso, basic_gridsearch, config_parallelisation, handle_resume, attach_states
from ingredients.trainer import train_ingredient, train
from ingredients.prepare_data import data_ingredient, ready_all_data_and_model
from utils.functions import save_pickle
from configurations import configs

# CLI's for paralellisation
parser = argparse.ArgumentParser()
parser.add_argument('-f', '--folder', help='Folder that holds the data.', default='UEA')
parser.add_argument('-ds', '--dataset', help='The name of the dataset to run.', default='EigenWorms')
parser.add_argument('-c', '--config', help='The config argument.', default='test')
parser.add_argument('-rm', '--remove_folder', help='Removes the folder if exists and restarts.', action='store_true')
# parser.add_argument('-rm', '--remove_folder', help='Removes the folder if exists and restarts.', action='store_false')
parser.add_argument('-t', '--test', help='Set in a small epoch test mode.', action='store_true')
parser.add_argument('-gpu', '--gpu_idx', help='The index of the GPU to run on.', default=0, type=int)
parser.add_argument('-igpu', '--igpu', help='Integer identifier of the GPU from ngpus.', default=0, type=int)
parser.add_argument('-ngpus', '--num_gpus', help='The total number of GPUs', default=0, type=int)
args = parser.parse_args()

# Save folder
test_str = '/test' if args.test else ''
SAVE_DIR = './models{}/{}/{}/{}'.format(test_str, args.folder, args.dataset, args.config)
handle_resume(SAVE_DIR, True, args.remove_folder)

# Setup configuration parallelisation
configuration = configs[args.folder][args.dataset][args.config]
configuration = config_parallelisation(configuration, args.igpu, args.num_gpus)

# Setup experiment
ingredients = [data_ingredient, train_ingredient]
ex = Experiment(args.folder + '.' + args.dataset, ingredients=ingredients)


# Data config
@data_ingredient.config
def update_cfg():
    # Location
    ds_folder = args.folder
    ds_name = args.dataset
    # Dataset
    solver = 'rk4'
    adjoint = False
    # Dataloader
    batch_size = 32


# Ingredient config
@train_ingredient.config
def update_cfg():
    optimizer_name = 'adam'
    loss_fn = 'ce'
    max_epochs = 1000 if not args.test else 1
    metrics = ['loss', 'acc']
    val_metric_to_monitor = 'acc'
    gpu_idx = args.gpu_idx


# Main configuration
@ex.config
def config():
    # Dataset
    model_type = None
    depth = 1
    step = 1
    # Model
    hyperopt_metric = None
    hidden_dim = None
    hidden_hidden_multiplier = None
    num_layers = 1
    tune_params = None


@ex.main
def main(_run,
         model_type,
         depth,
         step,
         hidden_dim,
         hidden_hidden_multiplier,
         num_layers,
         tune_params):
    # For saving additional run info and tracking completion state
    attach_states(_run, save_dir=SAVE_DIR)

    # Get dataloaders and model in one step
    model, train_dl, val_dl, test_dl = ready_all_data_and_model(
        _run, model_type, depth, step, hidden_dim, hidden_hidden_multiplier, num_layers, tune_params
    )

    # Train model
    model, results, history = train(
        model, train_dl, val_dl, test_dl, save_dir=_run.save_dir + '/model'
    )

    # Save best values to metrics and history to file
    for name, value in results.items():
        _run.log_scalar(name, value)
    save_pickle(history, _run.save_dir + '/validation_history.pkl')


if __name__ == '__main__':
    # Create FSO (this creates a folder to log information into).
    create_fso(ex, SAVE_DIR, remove_folder=False)

    # Remove some parameter combinations
    param_drops = [
        lambda params: (params['model_type'] not in ['nrde', 'logsig-rnn']) and (params['depth'] > 1),
        lambda params: (params['model_type'] in ['nrde', 'logsig-rnn'] and params['depth'] > 1 and params['step'] == 1)
    ]

    # Gridsearch it
    basic_gridsearch(ex, configuration, param_drops=param_drops)
