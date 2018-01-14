# TODO

### SaveModel callback does not save optimizer state

Currently the optimizer is not known to `Trainer`, only to the train function. However, the `SaveModel` callback calls `save_checkpoint()` which does need the optimizer state.

Possible solutions:

- Tell `Trainer` about the optimizer anyway. Typically you're only going to be using a single optimizer and so it's OK if the training loop knows about it.

- Give `SaveModel` a list of objects to serialize: `SaveModel(..., also_save={"optimizer": optimizer})`

### Trainee class instead of train function

Maybe: don't pass train function to Trainer but instance of Trainee (or TrainingStrategy):

```
class Trainee
    self.model
    self.optimizer
    self.metrics
    self.loss_fn
    def __init__(model, optimizer, loss_fn, metrics=[...])
    def fit_on_batch(x, y)
    def evaluate_on_batch(x, y)
```

This base class will have default implementations that are enough to run a classifier.

Potential problem: If using hyperparameters then subclass Trainee and give it a `train(x, y, lamb)` method. Will that work if the base class already has a `train(x, y)` method (since in Python there is no overloading, I think)? So maybe make an empty base class and one for classification (`ClassifierStrategy`).

### Pass multiple models?

For convenience, `fit()` and `evaluate()` need to be passed the `nn.Module` object to train / evaluate. Usually you'll only train / evaluate a single model at a time. The train/eval functions can still capture other models or whatever else they need. But perhaps it makes sense to allow the user to pass a list of model objects (for GANs etc).
