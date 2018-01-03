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
    ax.set_title("confusion matrix", fontsize=16)
    ax.set_xticks(range(conf.shape[1]))
    ax.set_yticks(range(conf.shape[0]))
    ax.set_xlabel("predicted label", fontsize=16)
    ax.set_ylabel("true label", fontsize=16)
    if target_names:
        ax.set_xticklabels(target_names, rotation=90)
        ax.set_yticklabels(target_names)
    plt.tick_params(axis="both", which="major", labelsize=12)
    plt.tick_params(axis="both", which="minor", labelsize=10)     
    ax.grid(False)
    im = ax.imshow(conf, interpolation="nearest", cmap=cmap)
    fig.colorbar(im)
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
