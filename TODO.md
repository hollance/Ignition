# TODO

### Pass multiple models?

For convenience, `fit()` and `evaluate()` need to be passed the `nn.Module` object to train / evaluate. Usually you'll only train / evaluate a single model at a time. The train/eval functions can still capture other models or whatever else they need. But perhaps it makes sense to allow the user to pass a list of model objects (for GANs etc).
