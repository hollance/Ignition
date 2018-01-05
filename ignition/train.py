from .imports import *
from .data import *
from .metrics import *
from .summary import *
from .utils import *


def predict_on_batch(model, x):
    """Returns the predictions for a single batch of examples.
    
    NOTE: Before you call this, make sure to do `model.train(False)`.
    
    Parameters
    ----------
    model: nn.Module
        The model.
    x: Tensor or numpy array
        Must have size (batch_size, in_channels, height, width).

    Returns
    -------
    Tensor containing the predicted probabilities.
    """
    return model(make_var(x, volatile=True)).data


def predict(model, pred_fn, data_loader, verbose=False):
    """Returns the predictions for the examples in the given dataset.

    Parameters
    ----------
    model: nn.Module
        The model.
    pred_fn: callable
        Function that performs the prediction over a single batch.
        Takes two arguments: model and x, a Tensor of inputs.
        Returns a Tensor (not a Variable) of size (batch_size, ...) 
    data_loader: torch.utils.data.DataLoader 
        Provides the dataset.
    verbose: bool (optional)
        Whether to show a progress bar (default is False)
    
    Returns
    -------
    Tensor containing the predictions.
    """
    model.train(False)
    offset = 0
    
    if verbose:
        progress_bar = ProgressBar(len(data_loader))
        
    for batch_idx, data in enumerate(data_loader):
        # The data loader can return inputs and targets, or just inputs.
        inputs = data[0] if type(data) is list else data

        batch_pred = pred_fn(model, inputs)
        batch_size = batch_pred.size(0)

        # Allocate the tensor that holds the predictions.
        if batch_idx == 0:
            num_samples = data_loader_sample_count(data_loader)
            y_size = list(batch_pred.size())
            y_size[0] = num_samples
            y_pred = torch.zeros(torch.Size(y_size))

        y_pred[offset:offset + batch_size, ...] = batch_pred
        offset += batch_size

        if verbose:
            progress_bar.update(batch_idx)
     
    if verbose:
        progress_bar.end()

    return y_pred


def evaluate_on_batch(model, x, y, loss_fn=None, metrics=["loss", "acc"]):
    """Evaluates the model on a single batch of examples.
    
    This is a evaluation function for a basic classifier. For more complex models,
    you should write your own evaluation function.    
    
    NOTE: Before you call this, make sure to do `model.train(False)`.
    
    Parameters
    ----------
    model: nn.Module
        Needed to make the predictions.
    x: Tensor or numpy array 
        Must have size (batch_size, in_channels, height, width)
    y: Tensor or numpy array 
        Contains the label indices (not one-hot encoded)
    loss_fn: optional
        The loss function used to compute the loss. Required when the
        metrics include "loss".
    metrics: list
        Which metrics to compute over the batch.
    
    Returns
    -------
    dict
        The computed metrics for this batch.
    """
    
    outputs = model(make_var(x, volatile=True))
    y_true = make_var(y, dtype=np.int, volatile=True)
    
    results = {}
    if "loss" in metrics:
        results["loss"] = loss_fn(outputs, y_true).data[0]
    if "acc" in metrics:
        results["acc"] = accuracy_metric(outputs, y_true)
    return results


def evaluate(model, eval_fn, data_loader, max_steps=None, verbose=True):
    """Computes the loss and accuracy on the given dataset in test mode.
    
    Example of how to use the evaluation function:
    ```
    def my_eval_fn(model, x, y):
        return evaluate_on_batch(model, x, y, crossentropy_loss, ["acc"])

    evaluate(model, my_eval_fn, testloader)
    ```
    Note how `my_eval_fn` captures the loss function and anything else it needs.

    Parameters
    ----------
    model: nn.Module
        The model to evaluate.
    eval_fn: callable 
        The function that evaluates a single batch. This function takes the
        following arguments:
            `model_to_eval`: the value from self.model
            `x`: a batch of inputs
            `y`: a batch of targets (optional)
        It should return a dictionary with any metrics you want to capture, 
        such as the loss and accuracy.
    data_loader: torch.utils.data.DataLoader 
        Provides the dataset.
    max_steps: int (optional)
        If not None, evaluate at most this many iterations.
    verbose: bool
        Whether to show a progress bar (default is True).
    
    Returns
    -------
    dict
        The computed metrics for the dataset.    
    """
    model.train(False)

    total_steps = 0
    total_examples = 0
    running_loss = 0.
    correct = 0
    results = {}
    
    if verbose:
        pbar_size = len(data_loader)
        if max_steps: pbar_size = min(max_steps, pbar_size)
        progress_bar = ProgressBar(pbar_size)

    for batch_idx, data in enumerate(data_loader):
        if isinstance(data, list):
            batch_size = data[0].size(0)
            results = eval_fn(model, *data)
        else:
            batch_size = data.size(0)
            results = eval_fn(model, data)
            
        total_steps += 1
        total_examples += batch_size
        
        if "loss" in results:
            running_loss += results["loss"]
            results["loss"] = running_loss / total_steps

        # Note: because the last batch may be smaller, we can't compute the 
        # average acc by adding up the accuracies for the batches. Instead,
        # keep track of the number of correct and divide by total examples.
        if "acc" in results:
            correct += results["acc"] * batch_size
            results["acc"] = correct / total_examples

        if verbose:
            msg = ", ".join(map(lambda x: "%s: %.5f" % x, results.items()))
            progress_bar.update(batch_idx, msg)

        if max_steps and total_steps >= max_steps:
            break

    if verbose:
        elapsed = time.time() - progress_bar.start_time
        progress_bar.end("%d steps - %ds - %s" % (total_steps, elapsed, msg))
        
    return results


def fit_on_batch(model, x, y, loss_fn, optimizer, metrics=["loss", "acc"]):
    """Trains the model on a single batch of examples.

    This is a training function for a basic classifier. For more complex models,
    you should write your own training function.

    NOTE: Before you call this, make sure to do `model.train(True)`.

    Parameters
    ----------
    model: nn.Module
        The model to train.
    x: Tensor or numpy array 
        Must have size (batch_size, in_channels, height, width).
    y: Tensor or numpy array 
        Contains the label indices (not one-hot encoded).
    loss_fn: 
        The loss function to use.
    optimizer: 
        The SGD optimizer to use.
    metrics: list
        Which metrics to compute over the batch.

    Returns
    -------
    dict
        The computed metrics for this batch.
    """
    optimizer.zero_grad()

    # Forward pass
    outputs = model(make_var(x))

    # Compute loss
    y_true = make_var(y, dtype=np.int, volatile=True)
    loss = loss_fn(outputs, y_true)
    
    # Backward pass
    loss.backward()
    optimizer.step()

    # Additional metrics
    results = {}
    if "loss" in metrics:
        results["loss"] = loss.data[0]
    if "acc" in metrics:
        results["acc"] = accuracy_metric(outputs, y_true)
    return results


class Trainer:
    """Encapsulates everything that's needed to train a model.
    
    Example of how to use the train function:
    ```
    def my_train_fn(model, x, y):
        return fit_on_batch(model, x, y, crossentropy_loss, optimizer)

    trainer = Trainer(model, my_train_fn, trainloader)
    trainer.fit(epochs=3)
    ```

    Attributes
    ----------
    callbacks: list 
        The active Callback objects.
    hyper: dictionary 
        Any hyperparameters that you want to modify with callbacks and pass
        into the train_fn function.
    """

    def __init__(self, model, train_fn, train_loader, eval_fn=None, val_loader=None, history=None, verbose=True):
        """Creates a new trainer.
        
        Parameters
        ----------
        model: nn.Module
            The model to train. Typically the train_fn should already capture model
            but you need to pass it to Trainer anyway for evaluation and Callbacks.
        train_fn: callable 
            The function that trains the model on a single batch of examples.
            It takes the following arguments:
                `model_to_train`: the value from self.model
                `x`: a batch of inputs
                `y`: a batch of targets (optional)
            If any hyperparameters are given to fit(), then the train_fn must also 
            accept these as arguments.                
            The train function should return a dictionary with any metrics you want
            to capture, typically the loss and accuracy.
        train_loader: torch.utils.data.DataLoader 
            Provides the training dataset.
        eval_fn: callable, optional
            Function that evaluates a single batch. See `evaluate()` for what this
            evaluation function should look like.
        val_loader: torch.utils.data.DataLoader, optional
            Provides the validation dataset.
        history: a History object, optional
            For when you want to resume training.
        verbose: bool
            Whether to show a progress bar (default is True).
        """
        self.model = model
        self.train_fn = train_fn
        self.train_loader = train_loader
        self.eval_fn = eval_fn
        self.val_loader = val_loader
        self.verbose = verbose
        self.history = history if history else History()
        self.hyper = {}
        self.callbacks = []

    def fit(self, epochs, max_steps=None, print_every=10):
        """Trains the model.
        
        Parameters
        ----------
        epochs: int
            How often to loop through the dataset.
        max_steps: int (optional)
            If not None, run each epoch for at most this many iterations.
        print_every: int (optional, default is 10)
            After how many batches to update the progress bar.
        """
        callback_dict = {"trainer": self, "model": self.model, "hyper": self.hyper}
        apply_on_all(self.callbacks, "on_train_begin", callback_dict)
        
        total_epochs = self.history.epochs() + epochs

        if self.verbose:
            msg = "Train epochs %d-%d on %d examples" % (self.history.epochs() + 1, total_epochs, 
                                                         data_loader_sample_count(self.train_loader, max_steps))
            if self.val_loader:
                msg += ", validate on %d examples" % data_loader_sample_count(self.val_loader)
            print(msg)

        have_header = False
        column_names = []
        hyper_keys = list(self.hyper.keys())

        for epoch in range(epochs):
            callback_dict["epoch"] = self.history.epochs()
            apply_on_all(self.callbacks, "on_epoch_begin", callback_dict)

            total_steps = 0
            total_examples = 0
            correct = 0
            running_loss = 0.
            results = {}
            
            self.model.train(True)

            if self.verbose:
                pbar_size = len(self.train_loader)
                if max_steps: pbar_size = min(max_steps, pbar_size)
                progress_bar = ProgressBar(pbar_size)

            for batch_idx, data in enumerate(self.train_loader):
                callback_dict["batch"] = batch_idx
                apply_on_all(self.callbacks, "on_batch_begin", callback_dict)

                if isinstance(data, list):
                    batch_size = data[0].size(0) 
                    batch_results = self.train_fn(self.model, *data, **self.hyper)
                else:
                    batch_size = data.size(0)                
                    batch_results = self.train_fn(self.model, data, **self.hyper)

                total_steps += 1
                total_examples += batch_size

                if "loss" in batch_results:
                    running_loss += batch_results["loss"]
                    results["loss"] = running_loss / total_steps
                if "acc" in batch_results:
                    correct += batch_results["acc"] * batch_size
                    results["acc"] = correct / total_examples

                if self.verbose and batch_idx % print_every == 0:
                    msg = "train " + ", ".join(map(lambda x: "%s: %.5f" % x, results.items()))
                    progress_bar.update(batch_idx, msg)

                apply_on_all(self.callbacks, "on_batch_end", callback_dict)

                if max_steps and total_steps >= max_steps:
                    break

            column_values = []
            for metric_name, metric_value in results.items():
                self.history.add("train_" + metric_name, metric_value)
                callback_dict["train_" + metric_name] = metric_value
                column_values.append(metric_value)
                if not have_header: column_names.append("tr " + metric_name)

            if self.val_loader:
                if self.verbose: 
                    progress_bar.update(batch_idx, msg + " - evaluating... 🤖 ")

                results = evaluate(self.model, self.eval_fn, self.val_loader, verbose=False)
                
                for metric_name, metric_value in results.items():
                    self.history.add("val_" + metric_name, metric_value)
                    callback_dict["val_" + metric_name] = metric_value
                    column_values.append(metric_value)
                    if not have_header: column_names.append("val " + metric_name)
                    
            if self.verbose:
                elapsed = time.time() - progress_bar.start_time

                column_values = [ a_slash_b(self.history.epochs(), total_epochs),
                                  total_steps, "%ds" % elapsed ] + column_values
                for key in hyper_keys:
                    column_values.append(self.hyper[key])
                
                if not have_header:
                    column_names = ["epoch", "steps", "time"] + column_names + hyper_keys
                    w1 = { "epoch": 7, "time": 5, "tr loss": 8, "tr acc": 7 } 
                    w2 = dict.fromkeys(hyper_keys, 8)
                    fmt = dict.fromkeys(hyper_keys, "%.5f")
                    table = Table(column_names, widths={**w1, **w2}, formats=fmt)
                    progress_bar.end(table.header())
                    print(table.row(column_values))
                    have_header = True
                else:
                    progress_bar.end(table.row(column_values))

            apply_on_all(self.callbacks, "on_epoch_end", callback_dict)

        apply_on_all(self.callbacks, "on_train_end", callback_dict)
        

class Callback():
    """Training callbacks must extend this abstract base class."""
    def __init__(self):
        pass

    def on_train_begin(self, info_dict):
        pass
    
    def on_epoch_begin(self, info_dict):
        pass

    def on_batch_begin(self, info_dict):
        pass

    def on_batch_end(self, info_dict):
        pass

    def on_epoch_end(self, info_dict):
        pass

    def on_train_end(self, info_dict):
        pass

    
class SaveModel(Callback):
    """Saves the model parameters after every epoch.
    
    Parameters
    ----------
    filename: string
        Name of the output file. You can use the placeholders: epoch, val_loss, 
        val_acc. For example, "mymodel_{epoch:d}_{val_loss:.4f}_{val_acc:.4f}"
    include_history: bool, optional (default is False)
        If true, also saves the optimizer state and training history.
    save_every: int, optional (default is 1)
        How often to save the model. Default is every epoch.
    always_save_better: bool, optional (default is False)
        If true, then save the model whenever the val_acc improves. Suggestion:
        set this to False during early epochs, but to True during later epochs
        so that only the models with the best score are kept.
    verbose: bool, optional (default is False)
        Print out a message when the model is saved.
    """
    def __init__(self, filename, include_history=False, save_every=1, always_save_better=False, verbose=False):
        self.filename = filename
        self.include_history = include_history
        self.save_every = save_every
        self.epochs_since_last_save = 0
        self.always_save_better = always_save_better
        self.best_val_acc = -np.Inf
        self.verbose = verbose
        
    def on_epoch_end(self, info_dict):
        val_acc = info_dict.get("val_acc")
        if val_acc and val_acc > self.best_val_acc:        
            if self.always_save_better:
                if self.verbose:
                    print("😊 val_acc improved from %.4f to %.4f" % (self.best_val_acc, val_acc))
                self.epochs_since_last_save = self.save_every  # force save
            self.best_val_acc = val_acc
        
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save < self.save_every: return
        self.epochs_since_last_save = 0

        metrics = { "epoch": info_dict["epoch"] + 1 }
        if "val_loss" in info_dict: metrics["val_loss"] = info_dict["val_loss"]
        if "val_acc" in info_dict: metrics["val_acc"] = info_dict["val_acc"]
        filename = self.filename.format(**metrics)

        if self.verbose:
            print("💾 Saving model to %s" % (filename))

        if self.include_history:
            save_checkpoint(filename, info_dict["trainer"], optimizer=None)
        else:
            torch.save(info_dict["model"].state_dict(), filename)


class LinearDecay(Callback):
    """Linearly decays a hyperparameter over a number of epochs.
    
    Parameters
    ----------
    name: string, the name of the hyperparameter
    start_value: the value at epoch 0
    end_value: the value at epoch max_epochs
    max_epochs: after this many epochs, the hyperparameter will have end_value
    optimizer: for changing the learning rate
    """
    def __init__(self, name, start_value, end_value, max_epochs, optimizer=None):
        self.name = name
        self.start_value = start_value
        self.end_value = end_value
        self.max_epochs = max_epochs
        self.optimizer = optimizer
        
    def on_epoch_begin(self, info_dict):
        new_value = self.end_value + (self.start_value - self.end_value) \
                    * (self.max_epochs - info_dict["epoch"]) / self.max_epochs
        info_dict["hyper"][self.name] = new_value
        
        if self.name == "lr" and self.optimizer:
            set_lr(self.optimizer, new_value)


class History():
    """Stores the history of a training session, such as the loss and other metrics.

    This object can be pickled.
    """
    def __init__(self):
        self.clear()

    def clear(self):
        self.metrics = {}

    def epochs(self):
        if "train_loss" in self.metrics:
            return len(self.metrics["train_loss"])
        else:
            return 0

    def best_epoch(self):
        m = [ "val_acc", "val_loss", "train_acc", "train_loss" ]
        f = [ np.argmax, np.argmin,  np.argmax,   np.argmin    ]

        for i, metric_name in enumerate(m):
            if metric_name in self.metrics:
                epoch = f[i](self.metrics[metric_name])
                return { "epoch": epoch, metric_name: self.metrics[metric_name][epoch] }

        return None

    def add(self, metric_name, value):
        if metric_name in self.metrics:
            self.metrics[metric_name].append(value)
        else:
            self.metrics[metric_name] = [value]

    def plot_loss(self, figsize=(12, 8)):
        self._plot(*self._metrics_of_type("_loss"), figsize, "loss", "upper right")

    def plot_accuracy(self, figsize=(12, 8)):
        self._plot(*self._metrics_of_type("_acc"), figsize, "accuracy", "lower right")

    def _metrics_of_type(self, type_name):
        data = []
        names = []
        for metric_name, metric_data in self.metrics.items():
            if metric_name.endswith(type_name):
                data.append(metric_data)
                names.append(metric_name[:-len(type_name)])
        return data, names

    def _plot(self, data, names, figsize, label, loc):
        fig = plt.figure(figsize=figsize)
        for d in data: plt.plot(d)
        plt.ylabel(label, fontsize=16)
        plt.xlabel("epoch", fontsize=16)
        plt.tick_params(axis="both", which="major", labelsize=12)
        plt.tick_params(axis="both", which="minor", labelsize=10)
        plt.legend(names, loc=loc, fontsize=16)

        
def set_lr(optimizer, lr):
    """Use this to manually change the learning rate."""
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def save_checkpoint(filename, trainer, optimizer):
    checkpoint = { "history": trainer.history,
                   "hyper": trainer.hyper,
                   "model": trainer.model.state_dict() }
    if optimizer:
        checkpoint["optimizer"] = optimizer.state_dict()

    torch.save(checkpoint, filename)
