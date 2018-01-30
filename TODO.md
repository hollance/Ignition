# TODO

### SaveModel callback does not save optimizer state

Currently the optimizer is not known to `Trainer`, only to the train function. However, the `SaveModel` callback calls `save_checkpoint()` which does need the optimizer state.

Possible solutions:

- Tell `Trainer` about the optimizer anyway. Typically you're only going to be using a single optimizer and so it's OK if the training loop knows about it.

- Give `SaveModel` a list of objects to serialize: `SaveModel(..., also_save={"optimizer": optimizer})`

### Pass multiple models?

For convenience, `fit()` and `evaluate()` need to be passed the `nn.Module` object to train / evaluate. Usually you'll only train / evaluate a single model at a time. The train/eval functions can still capture other models or whatever else they need. But perhaps it makes sense to allow the user to pass a list of model objects (for GANs etc).
