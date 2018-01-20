from .imports import *
from .data import *
from .utils import *


def accuracy_metric(y_pred, y_true, multi_label=False):
    """Computes the accuracy metric over a tensor with predictions.
    
    To compute accuracy of binary classification, y_pred should be a 1D
    tensor of shape (num_examples, ).
    
    To compute accuracy of multi-class classification, y_pred should be
    a 2D tensor of shape (num_examples, num_classes).
    
    To compute accuracy of multi-label classification, y_pred should be
    a 2D tensor of shape (num_examples, num_labels) and the multi_label
    parameter should be True.
    
    Parameters
    ----------
    y_pred: Tensor or Variable
        One-dimensional tensor with predicted label indices, or 2D tensor
        with one-hot encoded categories.
    y_true: Tensor or Variable
        One-dimensional tensor with ground-truth label indices.
    multi_label: bool (optional)
        True if this is a multi-label classification, False if it is
        multi-class.
    
    Returns
    -------
    float
        The accuracy over this batch.
    """
    assert(y_pred.dim() == 1 or y_pred.dim() == 2)
    assert(y_true.dim() == 1)
    assert(len(y_pred) == len(y_true))
    
    if y_pred.dim() > 1 and not multi_label:
        y_pred = from_onehot(y_pred)
    if isinstance(y_pred, Variable):    
        y_pred = y_pred.data
    if isinstance(y_true, Variable):    
        y_true = y_true.data       

    return (y_pred == y_true).sum() / len(y_true)

