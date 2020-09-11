import os
import tarfile
import urllib.response
import zipfile
import pickle


def mkdir_if_not_exists(loc, file=False):
    """Makes a directory if it doesn't already exist. If loc is specified as a file, ensure the file=True option is set.

    Args:
        loc (str): The file/folder for which the folder location needs to be created.
        file (bool): Set true if supplying a file (then will get the dirstring and make the dir).

    Returns:
        None
    """
    loc_ = os.path.dirname(loc) if file else loc
    if not os.path.exists(loc):
        os.makedirs(loc_, exist_ok=True)


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
        mkdir_if_not_exists(filename, file=True)

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


def download_url(url, loc):
    """ Downloads a url file to a specified location. """
    if not os.path.exists(loc):
        urllib.request.urlretrieve(url, loc)


def download_zip(base_folder, zipname, zipurl, unzip=True):
    """Function for downloading and unzipping zip files from urls.

    Args:
        base_folder (str): The folder it will be downloaded into.
        zipname (str): The filename of the zip. Will be saved as '{}/{}.zip'.format(base_folder, zipname).
        zipurl (str): The url of the zip download.
        unzip (bool): Set True to unzip after download

    Returns:
        None
    """
    assert os.path.isdir(base_folder), "Please make a folder at {} to store the data.".format(base_folder)

    # If the folder is empty, download into it. If not, clean the folder out. This download is designed to be the first
    # step in this process.
    if len(os.listdir(base_folder)) == 0:
        zip_loc = base_folder + '/{}.zip'.format(zipname)
        if not os.path.exists(zip_loc):
            download_url(zipurl, zip_loc)
    else:
        print('Files already exist in {}. Please remove if you require a fresh download.'.format(base_folder))

    # Perform unzipping into the same directory.
    if unzip:
        unzip(zip_loc, base_folder)


def unzip(file, loc):
    """ Unzips a zip file to a specified location. """
    with zipfile.ZipFile(file, 'r') as zip_ref:
        zip_ref.extractall(loc)


def extract_compressed_file(path, extract_loc):
    """Extracts a .zip or .tar.gz file to a specified directory.

    Args:
        path (str): The location of the compressed filed to extract.
        extract_loc (str): The location to extract to

    Returns:
        None
    """
    if path.endswith('.zip'):
        opener, mode = zipfile.ZipFile, 'r'
    elif path.endswith('.tar.gz') or path.endswith('.tgz'):
        opener, mode = tarfile.open, 'r:gz'
    else:
        raise NotImplementedError('Not implemented for file type: {}. Must end in .zip or .tar.gz.'.format(path))

    mkdir_if_not_exists(extract_loc)

    file = opener(path, mode)
    file.extractall(extract_loc)
    file.close()

