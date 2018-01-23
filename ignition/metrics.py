from .imports import *
from .data import *
from .utils import *


def accuracy_metric(y_pred, y_true, multi_label=False):
    """Computes the accuracy metric over a tensor with predictions.
    
    To compute accuracy of binary classification, both y_pred and y_true
    should be a 1D tensor of shape (num_examples, ).

    To compute accuracy of multi-class classification, y_pred should be
    a 2D tensor of shape (num_examples, num_classes) containing one-hot
    encoded categories; y_true is 1D with size (num_examples, ).

    To compute accuracy of multi-label classification, both y_pred and
    y_true should be a 2D tensor of shape (num_examples, num_labels) and 
    the multi_label parameter should be True.
    
    NOTE: For multi-label classification, this function computes "subset" 
    accuracy / the exact match ratio, where each row in y_pred must match
    each row in y_true exactly.

    Parameters
    ----------
    y_pred: Tensor or Variable
        The predicted label indices or one-hot encoded categories.
    y_true: Tensor or Variable
        The ground-truth label indices.
    multi_label: bool (optional)
        True if this is a multi-label classification, False if it is
        binary or multi-class.
    
    Returns
    -------
    float
        The accuracy over this batch.
    """
    assert(y_pred.dim() == 1 or y_pred.dim() == 2)
    assert(len(y_pred) == len(y_true))
    
    if y_pred.dim() > 1:
        if multi_label:
            assert(y_true.dim() == y_pred.dim())
        else:
            assert(y_true.dim() == 1)
            y_pred = from_onehot(y_pred)

    if isinstance(y_pred, Variable):    
        y_pred = y_pred.data
    if isinstance(y_true, Variable):    
        y_true = y_true.data       

    if multi_label:
        return np.mean(np.all(y_pred == y_true, axis=1))
    else:
        return (y_pred == y_true).sum() / len(y_true)

