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
        Image tensors should have size (batch_size, in_channels, height, width).

    Returns
    -------
    Tensor containing the predicted probabilities.
    """
    if type(x) != Variable:
        x = make_var(x, volatile=True)
    return model(x).data


def predict(model, pred_fn, data_loader, batch_axis=0, max_steps=None, verbose=False):
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
    batch_axis: int (optional)
        Which axis the mini-batches are on.
    max_steps: int (optional)
        If not None, run for at most this many iterations.
    verbose: bool (optional)
        Whether to show a progress bar (default is False)
    
    Returns
    -------
    Tensor containing the predictions.
    """
    model.train(False)
    offset = 0
    total_steps = 0
    
    if verbose:
        pbar_size = len(data_loader)
        if max_steps: pbar_size = min(max_steps, pbar_size)
        progress_bar = ProgressBar(pbar_size)
        num_samples = data_loader_sample_count(data_loader, max_steps)
        print("Predict on %d examples" % num_samples)
        
    for batch_idx, data in enumerate(data_loader):
        # The data loader can return inputs and targets, or just inputs.
        inputs = data[0] if type(data) in [list, tuple] else data

        batch_pred = pred_fn(model, inputs)
        batch_size = batch_pred.size(batch_axis)

        # Allocate the tensor that holds the predictions. We need to do this
        # after the first batch because we don't know the full size and data
        # type of the predictions tensor until then.
        if batch_idx == 0:
            y_size = list(batch_pred.size())
            y_size[0] = min(num_samples, max_steps*batch_size) if max_steps else num_samples
            y_pred = batch_pred.new(torch.Size(y_size))

        y_pred[offset:offset + batch_size, ...] = batch_pred
        offset += batch_size

        total_steps += 1
        
        if verbose:
            progress_bar.update(batch_idx)
     
        if max_steps and total_steps >= max_steps:
            break
    
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
        Image tensors should have size (batch_size, in_channels, height, width).
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


def evaluate(model, eval_fn, data_loader, batch_axis=0, max_steps=None, verbose=True, print_every=10):
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
    batch_axis: int (optional)
        Which axis the mini-batches are on. Usually this is 0 but for language
        data it is 1.
    max_steps: int (optional)
        If not None, evaluate at most this many iterations.
    verbose: bool
        Whether to show a progress bar (default is True).
    print_every: int (optional, default is 10)
        After how many batches to update the progress bar.
    
    Returns
    -------
    dict
        The computed metrics for the dataset.    
    """
    model.train(False)

    total_steps = 0
    total_examples = 0
    results = {}
    running_metrics = defaultdict(float)
    
    def format_msg():
        return ", ".join(map(lambda x: "%s: %.5f" % x, results.items()))

    if verbose:
        pbar_size = len(data_loader)
        if max_steps: pbar_size = min(max_steps, pbar_size)
        progress_bar = ProgressBar(pbar_size)
        num_samples = data_loader_sample_count(data_loader, max_steps)
        print("Evaluate on %d examples" % num_samples)

    for batch_idx, data in enumerate(data_loader):
        # The data can be: a single Tensor, a tuple of (inputs, targets),
        # or nested tuples ((inputs, sequence_lengths), targets).
        if isinstance(data, list) or isinstance(data, tuple):
            data_ = data[0][0] if type(data[0]) in [list, tuple] else data[0] 
            batch_size = data_.size(batch_axis)
            results = eval_fn(model, *data)
        else:
            batch_size = data.size(batch_axis)
            results = eval_fn(model, data)
            
        total_steps += 1
        total_examples += batch_size

        # Note: because the last batch may be smaller, we always multiply
        # the metrics by the batch_size again and divide by the total number
        # of examples, not by the total number of batches.
        for metric_name, metric_value in results.items():
            running_metrics[metric_name] += metric_value * batch_size
            results[metric_name] = running_metrics[metric_name] / total_examples
            
        if verbose and batch_idx % print_every == 0:
            progress_bar.update(batch_idx, format_msg())

        if max_steps and total_steps >= max_steps:
            break

    if verbose:
        elapsed = time.time() - progress_bar.start_time
        progress_bar.end("%d steps - %ds - %s" % (total_steps, elapsed, format_msg()))
        
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
        Image tensors should have size (batch_size, in_channels, height, width).
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

    def __init__(self, model, train_fn, train_loader, eval_fn=None, val_loader=None, 
                 history=None, batch_axis=0, verbose=True):
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
        batch_axis: int (optional)
            Which axis the mini-batches are on. Usually this is 0 but for language
            data it is 1.
        verbose: bool
            Whether to show a progress bar (default is True).
        """
        self.model = model
        self.train_fn = train_fn
        self.train_loader = train_loader
        self.eval_fn = eval_fn
        self.val_loader = val_loader
        self.history = history if history else History()
        self.batch_axis = batch_axis
        self.verbose = verbose
        self.hyper = {}
        self.callbacks = []

    def fit(self, epochs, max_steps=None, max_eval_steps=None, print_every=10):
        """Trains the model.
        
        Parameters
        ----------
        epochs: int
            How often to loop through the dataset.
        max_steps: int (optional)
            If not None, run each epoch for at most this many iterations.
        max_eval_steps: int (optional)
            If not None, run the validation for at most this many iterations.
        print_every: int (optional, default is 10)
            After how many batches to update the progress bar.
        """
        self.should_stop = False

        total_epochs = self.history.epochs() + epochs
        steps_per_epoch = len(self.train_loader)

        callback_dict = {
            "trainer": self, 
            "model": self.model, 
            "hyper": self.hyper, 
            "steps_per_epoch": steps_per_epoch,
        }
        apply_on_all(self.callbacks, "on_train_begin", callback_dict)
        
        if self.verbose:
            msg = "Train epochs %d-%d on %d examples" % (self.history.epochs() + 1, total_epochs, 
                                                         data_loader_sample_count(self.train_loader, max_steps))
            if self.val_loader:
                msg += ", validate on %d examples" % data_loader_sample_count(self.val_loader, max_eval_steps)
            print(msg)

        have_header = False
        column_names = []
        hyper_keys = list(self.hyper.keys())
        iteration = 0
        
        for epoch in range(epochs):
            callback_dict["epoch"] = self.history.epochs()
            apply_on_all(self.callbacks, "on_epoch_begin", callback_dict)

            total_steps = 0
            total_examples = 0
            results = {}
            running_metrics = defaultdict(float)
            
            self.model.train(True)

            if self.verbose:
                pbar_size = steps_per_epoch
                if max_steps: pbar_size = min(max_steps, pbar_size)
                progress_bar = ProgressBar(pbar_size)

            for batch_idx, data in enumerate(self.train_loader):
                callback_dict["batch"] = batch_idx
                callback_dict["iteration"] = iteration
                apply_on_all(self.callbacks, "on_batch_begin", callback_dict)
                iteration += 1

                if isinstance(data, list) or isinstance(data, tuple):
                    data_ = data[0][0] if type(data[0]) in [list, tuple] else data[0] 
                    batch_size = data_.size(self.batch_axis)
                    batch_results = self.train_fn(self.model, *data, **self.hyper)
                else:
                    batch_size = data.size(self.batch_axis)                
                    batch_results = self.train_fn(self.model, data, **self.hyper)

                total_steps += 1
                total_examples += batch_size

                for metric_name, metric_value in batch_results.items():
                    running_metrics[metric_name] += metric_value * batch_size
                    results[metric_name] = running_metrics[metric_name] / total_examples

                if self.verbose and batch_idx % print_every == 0:
                    msg = "train " + ", ".join(map(lambda x: "%s: %.5f" % x, results.items()))
                    progress_bar.update(batch_idx, msg)

                for metric_name, metric_value in results.items():
                    callback_dict["batch_" + metric_name] = metric_value
                    
                apply_on_all(self.callbacks, "on_batch_end", callback_dict)

                if self.should_stop or (max_steps and total_steps >= max_steps):
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

                apply_on_all(self.callbacks, "on_eval_begin", callback_dict)

                results = evaluate(self.model, self.eval_fn, self.val_loader, 
                                   batch_axis=self.batch_axis, 
                                   max_steps=max_eval_steps, verbose=False)
                
                for metric_name, metric_value in results.items():
                    self.history.add("val_" + metric_name, metric_value)
                    callback_dict["val_" + metric_name] = metric_value
                    column_values.append(metric_value)
                    if not have_header: column_names.append("val " + metric_name)

                apply_on_all(self.callbacks, "on_eval_end", callback_dict)

            if self.verbose:
                elapsed = time.time() - progress_bar.start_time

                column_values = [ a_slash_b(self.history.epochs(), total_epochs),
                                  total_steps, "%ds" % elapsed ] + column_values
                for key in hyper_keys:
                    column_values.append(self.hyper[key])
                
                if not have_header:
                    column_names = ["epoch", "steps", "time"] + column_names + hyper_keys
                    w1 = { "epoch": 7, "steps": 5, "time": 5 } 
                    w2 = dict.fromkeys(hyper_keys, 8)
                    fmt = dict.fromkeys(hyper_keys, "%.5f")
                    table = Table(column_names, widths={**w1, **w2}, formats=fmt)
                    progress_bar.end(table.header())
                    print(table.row(column_values))
                    have_header = True
                else:
                    progress_bar.end(table.row(column_values))

            apply_on_all(self.callbacks, "on_epoch_end", callback_dict)

            if self.should_stop:
                break

        apply_on_all(self.callbacks, "on_train_end", callback_dict)
        

class Callback():
    """Training callbacks must extend this abstract base class."""
    def __init__(self): pass
    def on_train_begin(self, info_dict): pass
    def on_epoch_begin(self, info_dict): pass
    def on_batch_begin(self, info_dict): pass
    def on_batch_end(self, info_dict): pass
    def on_eval_begin(self, info_dict): pass
    def on_eval_end(self, info_dict): pass
    def on_epoch_end(self, info_dict): pass
    def on_train_end(self, info_dict): pass

    
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
    metric_name: string, optional (default is "val_acc")
        The metric to use when always_save_better is True.
    smaller_is_better: bool, optional (default is False)
        True if a smaller value for the chosen metric means improvement.
    verbose: bool, optional (default is False)
        Print out a message when the model is saved.
    """
    def __init__(self, filename, include_history=False, save_every=1, 
                 always_save_better=False, metric_name="val_acc", smaller_is_better=False,
                 verbose=False):
        self.filename = filename
        self.include_history = include_history
        self.save_every = save_every
        self.epochs_since_last_save = 0
        self.always_save_better = always_save_better
        self.metric_name = metric_name
        self.smaller_is_better = smaller_is_better
        self.best = np.Inf if smaller_is_better else -np.Inf
        self.verbose = verbose

    def on_epoch_end(self, info_dict):
        metric = info_dict.get(self.metric_name)
        if metric:
            f = np.less if self.smaller_is_better else np.greater
            if f(metric, self.best):
                if self.always_save_better:
                    if self.verbose:
                        print("😊 %s improved from %.4f to %.4f" % (self.metric_name, self.best, metric))
                    self.epochs_since_last_save = self.save_every  # force save
                self.best = metric

        self.epochs_since_last_save += 1
        if self.epochs_since_last_save < self.save_every: return
        self.epochs_since_last_save = 0

        metrics = dict(info_dict)
        metrics["epoch"] += 1
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
    """
    def __init__(self, name, start_value, end_value, max_epochs):
        self.name = name
        self.start_value = start_value
        self.end_value = end_value
        self.max_epochs = max_epochs
        
    def on_epoch_begin(self, info_dict):
        new_value = self.end_value + (self.start_value - self.end_value) \
                    * (self.max_epochs - info_dict["epoch"]) / self.max_epochs
        info_dict["hyper"][self.name] = new_value

            
class LRSchedule(Callback):
    """Wrapper around PyTorch's learning rate scheduler, which lets you
    adjust the learning rate based on the number of epochs.
    
    Parameters
    ----------
    scheduler: torch.optim.lr_scheduler object
    """
    def __init__(self, scheduler):
        self.scheduler = scheduler
        
    def on_epoch_begin(self, info_dict):
        self.scheduler.step()
        history = info_dict["trainer"].history
        history.add("lr", self.scheduler.get_lr())


class LRFinder(Callback):
    """Increments the learning rate on every batch until the loss starts
    increasing again. Use this to determine a good learning rate to start
    training the model with.
    
    Based on code and ideas from https://github.com/fastai/fastai
    
    Parameters
    ----------
    optimizer: torch.optim object
    start_lr: float
        The learning rate to start with (should be quite small).
    end_lr: float
        The maximum learning rate to try (should be large-ish).
    steps: int
        How many batches to evaluate. One epoch is usually enough.
    """
    def __init__(self, optimizer, start_lr=1e-5, end_lr=10, steps=100):
        self.optimizer = optimizer
        self.steps = steps
        self.values = np.logspace(np.log10(start_lr), np.log10(end_lr), steps)
        
    def on_train_begin(self, info_dict):
        self.best_loss = 1e9
        self.loss_history = []
        self.lr_history = []

    def on_batch_begin(self, info_dict):
        lr = self.values[info_dict["iteration"]]
        set_lr(self.optimizer, lr)
        self.lr_history.append(lr)
        
    def on_batch_end(self, info_dict):
        loss = info_dict["batch_loss"]
        iteration = info_dict["iteration"]

        # Note: in the last couple of batches the loss may explode,
        # which is why we don't plot those.
        self.loss_history.append(loss)
        
        if math.isnan(loss) or loss > self.best_loss*4 or iteration >= self.steps - 1:
            info_dict["trainer"].should_stop = True
            return
        
        if loss < self.best_loss and iteration > 10:
            self.best_loss = loss
            
    def plot(self, figsize=(12, 8)):
        fig = plt.figure(figsize=figsize)
        plt.ylabel("loss", fontsize=16)
        plt.xlabel("learning rate (log scale)", fontsize=16)
        plt.xscale("log")
        plt.plot(self.lr_history[10:-5], self.loss_history[10:-5])
        plt.show()

        
class CosineAnneal(Callback):
    """Cosine annealing for the learning rate, with restarts.
    
    The learning rate is varied between lr_max and lr_min over cycle_len
    epochs.
    
    Note: The validation score may temporarily be worse in the first part
    of the cycle (where the learning rate is high). This is why you should
    always train for a round number of cycles. For example, if cycle_len=1
    and cycle_mult=2 then train for 1, 3, 7, 15, 31 etc epochs.
    
    It's allowed to change the cycle_len and cycle_mult parameters before
    the next training run, but make sure you do this after the last cycle
    has completely finished (or else there will be abrupt changes in the LR).
    
    Based on the paper 'SGDR: Stochastic Gradient Descent with Warm Restarts',
    arXiv:1608.03983 and code from https://github.com/fastai/fastai
    
    Parameters
    ----------
    optimizer: torch.optim object
    lr_min: float
        The lowest learning rate.
    lr_max: float
        The highest learning rate.
    cycle_len: int
        How many epochs there are in one cycle.
    cycle_mult: int
        After each complete cycle, the cycle_len is multiplied by this number.
        This makes the learning rate anneal at a slower pace over time.
    """
    def __init__(self, optimizer, lr_min, lr_max, cycle_len, cycle_mult=1):
        self.optimizer = optimizer
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.cycle_len = cycle_len
        self.cycle_mult = cycle_mult
        self.cycles_completed = 0
        self.cycle_iter = 0
        self.cycle_width = 1

    def on_batch_begin(self, info_dict):
        steps_per_epoch = info_dict["steps_per_epoch"]
        steps = steps_per_epoch * self.cycle_len * self.cycle_width

        # Use a low learning rate for the very first batches.
        if info_dict["iteration"] < steps_per_epoch/20:
            lr = self.lr_max / 100.
        else:
            lr = self.lr_min + 0.5*(self.lr_max - self.lr_min) * \
                                   (1. + np.cos(self.cycle_iter*np.pi / steps))

        set_lr(self.optimizer, lr)
        history = info_dict["trainer"].history
        history.add("lr", lr)

        self.cycle_iter += 1
        if self.cycle_iter >= steps:
            self.cycle_iter = 0
            self.cycle_width *= self.cycle_mult
            self.cycles_completed += 1
            # TODO: save the model here for snapshot ensembles
            
            
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

    def best_epoch(self, metric_name=None, smaller_is_better=False):
        if metric_name:
            m = [ metric_name ]
            f = [ np.argmin if smaller_is_better else np.argmax ]
        else:
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
        self._plot(*self._metrics_of_type("_loss"), figsize, ylabel="loss", legend="upper right")

    def plot_accuracy(self, figsize=(12, 8)):
        self._plot(*self._metrics_of_type("_acc"), figsize, ylabel="accuracy", legend="lower right")

    def plot(self, metric_name, figsize=(12, 8), xlabel=None, ylabel=None):
        self._plot(*self._metrics_of_type(metric_name), figsize, xlabel=xlabel or "", ylabel=ylabel or metric_name)

    def _metrics_of_type(self, type_name):
        data = []
        names = []
        for metric_name, metric_data in self.metrics.items():
            if metric_name.endswith(type_name):
                data.append(metric_data)
                metric_name = metric_name[:-len(type_name)]
                if metric_name.endswith("_"):
                    metric_name = metric_name[:-1]
                names.append(metric_name)
        return data, names

    def _plot(self, data, names, figsize, xlabel="epoch", ylabel="", legend=None):
        fig = plt.figure(figsize=figsize)
        for d in data: plt.plot(d)
        plt.ylabel(ylabel, fontsize=16)
        plt.xlabel(xlabel, fontsize=16)
        plt.tick_params(axis="both", which="major", labelsize=12)
        plt.tick_params(axis="both", which="minor", labelsize=10)
        if len(data) > 1:
            plt.legend(names, loc=legend, fontsize=16)

        
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
