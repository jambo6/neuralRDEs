import os
import pickle
import json
import time


def save_pickle(obj, filename, protocol=4, create_folder=True):
    """ Basic pickle/dill dumping.

    Given a python object and a filename, the method will save the object under that filename.

    Args:
        obj (python object): The object to be saved.
        filename (str): Location to save the file.
        protocol (int): Pickling protocol (see pickle docs).
        create_folder (bool): Set True to create the folder if it does not already exist.

    Returns:
        None
    """
    if create_folder:
        _create_folder_if_not_exist(filename)

    # Save
    with open(filename, 'wb') as file:
        pickle.dump(obj, file, protocol=protocol)


def load_pickle(filename):
    """ Basic dill/pickle load function.

    Args:
        filename (str): Location of the object.

    Returns:
        python object: The loaded object.
    """
    with open(filename, 'rb') as file:
        obj = pickle.load(file)
    return obj


def _create_folder_if_not_exist(filename):
    """ Makes a folder if the folder component of the filename does not already exist. """
    os.makedirs(os.path.dirname(filename), exist_ok=True)


def save_json(obj, filename, create_folder=True):
    """ Save file with json. """
    if create_folder:
        _create_folder_if_not_exist(filename)

    # Save
    with open(filename, 'wb') as file:
        json.dump(obj, file)


def load_json(filename):
    """ Load file with json. """
    with open(filename) as file:
        obj = json.load(file)
    return obj


def timeit(method):
    """ Get the time it takes for a method to run.

    Args:
        method (function): The function to time.

    Returns:
        Method wrapped with an operation to time it.
    """
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r \n  %2.2f ms' % (method, (te - ts) * 1000))
        return result
    return timed


