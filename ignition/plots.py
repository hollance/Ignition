from .imports import *
from .utils import *
from sklearn import metrics


def imshow(img, untransform=None, figsize=None):
    if untransform:
        img = untransform(img)

    if torch.is_tensor(img):
        img = np.transpose(img.numpy(), (1, 2, 0))

    fig = plt.figure(figsize=figsize)
    plt.grid(False)
    plt.imshow(img)
    plt.show()


def plot_images_with_titles(images, titles, nrows=1):
    ncols = len(images) // nrows
    fig = plt.figure(figsize=(max(ncols*2, 2), max(nrows*2, 2)))
    for i in range(len(images)):
        ax = fig.add_subplot(nrows, ncols, i+1)
        ax.set_frame_on(False)
        ax.set_axis_off()
        ax.set_title(titles[i], fontsize=14)
        ax.imshow(images[i])
    plt.show()
    

def plot_images_from_indices_with_titles(idxs, dataset, titles, nrows, untransform=None):
    """Plots a grid of images and titles.
    
    Parameters
    ----------
    idxs: numpy array
        The indices of the images in the dataset
    dataset: numpy array or a Dataset object
        Contains the actual image data. If a numpy array, the images must have
        shape (H, W, 3). If a Dataset object, the images must be (3, H, W).
    nrows: int
        Size of the image grid.
    untransform: callable, optional
        Use this when the dataset is a PyTorch Dataset object.
    """
    images = []
    for i, idx in enumerate(idxs):
        img = dataset[idx]
        if isinstance(img, tuple):
            img = img[0]
        if untransform:
            img = untransform(img)
        if torch.is_tensor(img):
            img = np.transpose(img.numpy(), (1, 2, 0))
        images.append(img)
    plot_images_with_titles(images, titles, nrows)    

    
def plot_predictions(predictions, target_names=None, figsize=(18, 4), width=0.8):
    predictions = to_numpy(predictions)
    fig = plt.figure(figsize=figsize)
    if target_names:
        plt.bar(range(0, len(target_names)), predictions, width)
        plt.xticks(range(0, len(target_names)), target_names)
    else:
        plt.bar(range(0, predictions.shape[0]), predictions, width)
    plt.show()


def print_top5(predictions, target_names):
    predictions = to_numpy(predictions)
    top_ixs = predictions.argsort()[::-1][:5]
    for i in top_ixs:
        print("%.5f %s" % (predictions[i], target_names[i]))


def confusion_matrix(y_true, y_pred, target_names=None, figsize=(12, 12), cmap=None):
    conf = metrics.confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(conf, target_names, figsize, cmap)


def plot_confusion_matrix(conf, target_names=None, figsize=(12, 12), cmap=None):
    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(conf, annot=True, annot_kws={"size": 16}, fmt="d",
                cmap=cmap, ax=ax,
                xticklabels=target_names if target_names else "auto",
                yticklabels=target_names if target_names else "auto")

    ax.set_title("confusion matrix", fontsize=16)
    ax.set_xlabel("predicted label", fontsize=16)
    ax.set_ylabel("true label", fontsize=16)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=16)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=16)

    ax.grid(False)
    plt.show()


def plot_target_counts(y, target_names):
    """Plots a histogram of how often each class is predicted.
    
    Parameters
    ----------
    y: Tensor or numpy array
        The class indices.
    target_names: list
        The names of the classes.
    """
    y_labels = list(map(lambda x: target_names[x], y))
    fig = plt.figure(figsize=(12, 6))
    sns.countplot(y_labels)

    
def precision_recall(y_true, y_pred, target_names=None):
    print(metrics.classification_report(y_true, y_pred, target_names=target_names))    
    
    
def _plot_images_with_probabilities(idxs, probs, labels, dataset, classes, nrows, untransform=None):
    """Plots a grid of images with their predicted probability and
    class label as the title.
    
    Parameters
    ----------
    probs: numpy array
        The probability to show for each image.
    labels: numpy array
        The class index to use for each image's class name.
    classes: list
        The names of the classes.
    """
    titles = []
    for i, idx in enumerate(idxs):
        titles.append("%.4f %s" % (probs[i], classes[labels[idx]]))
    plot_images_from_indices_with_titles(idxs, dataset, titles, nrows, untransform)


def _gather(class_idx, mask, y):
    """Grabs the indices for the rows from y that match the mask,
    as well as the predicted probabilities for the specified class."""
    idxs = np.where(mask)[0]
    probs = y[idxs, class_idx]
    return idxs, probs


def _most(class_idx, k, y, idxs, probs):
    """Grabs the k rows with the highest probability."""
    idxs = idxs[np.argsort(probs)[-k:]][::-1]
    probs = y[idxs, class_idx]    
    return idxs, probs


def _least(class_idx, k, y, idxs, probs):
    """Grabs the k rows with the lowest probability."""
    idxs = idxs[np.argsort(probs)[:k]][::-1]
    probs = y[idxs, class_idx]
    return idxs, probs


def _random(class_idx, k, y, idxs, _):
    """Grabs k rows at random."""
    idxs = np.random.choice(idxs, k, replace=False)
    probs = y[idxs, class_idx]
    return idxs, probs


def correct_examples(class_idx, y_pred, y_true):
    """Returns the indices of the examples for the given class
    that were predicted correctly, as well as the corresponding
    probabilities.
    
    Parameters
    ----------
    class_idx: int
        The index of the class to look at.
    y_pred: numpy array
        Array of shape (num_examples, num_classes) containing the 
        predicted probabilities.
    y_true: numpy array
        Array of shape (num_examples) with the true label indices
        (not one-hot encoded).
    """
    y_pred_idx = np.argmax(y_pred, axis=1)
    mask = (y_pred_idx == class_idx) & (y_true == class_idx)
    return _gather(class_idx, mask, y_pred)


def incorrect_examples_by_precision(class_idx, y_pred, y_true):
    """Returns the indices of the examples that were incorrectly
    predicted as being from the specified class, as well as the 
    corresponding probabilities."""
    y_pred_idx = np.argmax(y_pred, axis=1)
    mask = (y_pred_idx == class_idx) & (y_true != class_idx)
    return _gather(class_idx, mask, y_pred)


def incorrect_examples_by_recall(class_idx, y_pred, y_true):
    """Returns the indices of the examples that were incorrectly
    predicted as being from a different class, as well as the 
    corresponding probabilities."""
    y_pred_idx = np.argmax(y_pred, axis=1)
    mask = (y_pred_idx != class_idx) & (y_true == class_idx)
    return _gather(class_idx, mask, y_pred)


def random_correct(class_idx, y_pred, y_true, k=10):
    """Returns k random correct predictions for a class."""
    return _random(class_idx, k, y_pred, *correct_examples(class_idx, y_pred, y_true))


def most_confident_correct(class_idx, y_pred, y_true, k=10):
    """Returns the most confident correct predictions for a class."""
    return _most(class_idx, k, y_pred, *correct_examples(class_idx, y_pred, y_true))


def least_confident_correct(class_idx, y_pred, y_true, k=10):
    """Returns the least confident correct predictions for a class."""
    return _least(class_idx, k, y_pred, *correct_examples(class_idx, y_pred, y_true))


def random_incorrect(class_idx, y_pred, y_true, k=10):
    """Returns k random incorrect predictions for a class."""
    return _random(class_idx, k, y_pred, *incorrect_examples_by_precision(class_idx, y_pred, y_true))


def most_confident_incorrect_by_precision(class_idx, y_pred, y_true, k=10):
    """Returns the most confident wrong predictions for a class."""
    return _most(class_idx, k, y_pred, *incorrect_examples_by_precision(class_idx, y_pred, y_true))


def least_confident_incorrect_by_precision(class_idx, y_pred, y_true, k=10):
    """Returns the least confident wrong predictions for a class."""
    return _least(class_idx, k, y_pred, *incorrect_examples_by_precision(class_idx, y_pred, y_true))


def most_confident_incorrect_by_recall(class_idx, y_pred, y_true, k=10):
    """Returns the most confident wrong predictions for a class."""
    return _most(class_idx, k, y_pred, *incorrect_examples_by_recall(class_idx, y_pred, y_true))


def least_confident_incorrect_by_recall(class_idx, y_pred, y_true, k=10):
    """Returns the least confident wrong predictions for a class."""
    return _least(class_idx, k, y_pred, *incorrect_examples_by_recall(class_idx, y_pred, y_true))


def most_uncertain(y_pred, y_true, k=10):
    """Returns the examples with the lowest probabilities across all classes."""
    y_pred_max = y_pred.max(axis=1)
    idxs = np.argsort(y_pred_max)[:k]
    probs = y_pred_max[idxs]
    return idxs, probs


def plot_random_correct(class_idx, y_pred, y_true, dataset, classes, 
                        k=10, nrows=1, untransform=None):
    _plot_images_with_probabilities(*random_correct(class_idx, y_pred, y_true, k), 
                                    y_true, dataset, classes, nrows, untransform)


def plot_random_incorrect(class_idx, y_pred, y_true, dataset, classes, 
                          k=10, nrows=1, untransform=None):
    _plot_images_with_probabilities(*random_incorrect(class_idx, y_pred, y_true, k), 
                                    y_true, dataset, classes, nrows, untransform)


def plot_most_confident_correct(class_idx, y_pred, y_true, dataset, classes, 
                                k=10, nrows=1, untransform=None):
    """Plots the images for the most confident correct predictions for a class.
    
    Parameters
    ----------
    class_idx: int
        The index of the class to look at.
    y_pred: numpy array
        Array of shape (num_examples, num_classes) containing the 
        predicted probabilities.
    y_true: numpy array
        Array of shape (num_examples) with the true label indices
        (not one-hot encoded).
    dataset: numpy array or a Dataset object
        Contains the actual image data. If a numpy array, the images must have
        shape (H, W, 3). If a Dataset object, the images must be (3, H, W).
    classes: list
        The names of the classes.
    k: int
        How many images to show.
    nrows: int
        Size of the image grid.
    untransform: callable, optional
        Use this when the dataset is a PyTorch Dataset object.
    """    
    _plot_images_with_probabilities(*most_confident_correct(class_idx, y_pred, y_true, k), 
                                    y_true, dataset, classes, nrows, untransform)
    

def plot_least_confident_correct(class_idx, y_pred, y_true, dataset, classes, 
                                 k=10, nrows=1, untransform=None):
    _plot_images_with_probabilities(*least_confident_correct(class_idx, y_pred, y_true, k), 
                                    y_true, dataset, classes, nrows, untransform)
    

def plot_most_confident_incorrect_by_precision(class_idx, y_pred, y_true, dataset, classes, 
                                               k=10, nrows=1, untransform=None):
    _plot_images_with_probabilities(*most_confident_incorrect_by_precision(class_idx, y_pred, y_true, k), 
                                    y_true, dataset, classes, nrows, untransform)


def plot_least_confident_incorrect_by_precision(class_idx, y_pred, y_true, dataset, classes, 
                                                k=10, nrows=1, untransform=None):
    _plot_images_with_probabilities(*least_confident_incorrect_by_precision(class_idx, y_pred, y_true, k), 
                                    y_true, dataset, classes, nrows, untransform)


def plot_most_confident_incorrect_by_recall(class_idx, y_pred, y_true, dataset, classes, 
                                            k=10, nrows=1, untransform=None):
    y_pred_idx = np.argmax(y_pred, axis=1)
    _plot_images_with_probabilities(*most_confident_incorrect_by_recall(class_idx, y_pred, y_true, k), 
                                    y_pred_idx, dataset, classes, nrows, untransform)


def plot_least_confident_incorrect_by_recall(class_idx, y_pred, y_true, dataset, classes, 
                                             k=10, nrows=1, untransform=None):
    y_pred_idx = np.argmax(y_pred, axis=1)
    _plot_images_with_probabilities(*least_confident_incorrect_by_recall(class_idx, y_pred, y_true, k),
                                    y_pred_idx, dataset, classes, nrows, untransform)


def plot_most_uncertain(y_pred, y_true, dataset, classes, 
                        k=10, nrows=1, untransform=None):
    y_pred_idx = np.argmax(y_pred, axis=1)
    _plot_images_with_probabilities(*most_uncertain(y_pred, y_true, k), 
                                    y_pred_idx, dataset, classes, nrows, untransform)

