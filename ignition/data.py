from .imports import *


def to_onehot(y, num_classes=None):
    """Converts label indices to one-hot encoded vectors.
    
    Parameters
    ----------
    y: list or numpy array 
        Contains the labels as integers from 0 to num_classes.
    num_classes: int, optional
        The total number of classes.
    
    Returns
    -------
    Numpy array with one-hot encoded vectors.
    """
    y = np.array(y, dtype=np.int)
    
    if not num_classes:
        num_classes = np.max(y) + 1
    
    num_examples = y.shape[0]
    onehot = np.zeros((num_examples, num_classes))
    onehot[np.arange(num_examples), y] = 1.
    return onehot


def from_onehot(y):
    """Converts one-hot encoded probabilities to label indices.
    
    Parameters
    ----------
    y: Tensor

    Returns
    -------
    Tensor containing the label indices.
    """
    _, labels_pred = torch.max(y, 1)
    return labels_pred


def label2class(y, class_names):
    """Converts label indices to class names.
    
    Parameters
    ----------
    y: Tensor or numpy array
    class_names: list
    
    Returns
    -------
    List containing the class name for each prediction.
    """
    return list(map(lambda x: class_names[x], y))


def data_loader_sample_count(data_loader, max_steps=None):
    """How many examples the DataLoader object will iterate through."""
    if isinstance(data_loader, torch.utils.data.DataLoader):
        num_samples = len(data_loader.sampler)
    elif hasattr(data_loader, "num_examples"):
        num_samples = data_loader.num_examples()
    else:
        num_samples = len(data_loader)
    
    if hasattr(data_loader, "batch_size"):
        if getattr(data_loader, "drop_last", False):
            num_samples = (num_samples // data_loader.batch_size) * data_loader.batch_size
        if max_steps:
            num_samples = min(num_samples, max_steps * data_loader.batch_size)
    return num_samples


class SingleTensorDataset(Dataset):
    """Dataset wrapping a data tensor but no targets."""
    def __init__(self, data_tensor):
        self.data_tensor = data_tensor

    def __getitem__(self, index):
        return self.data_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)

