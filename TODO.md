# TODO

### Pass multiple models?

For convenience, `fit()` and `evaluate()` need to be passed the `nn.Module` object to train / evaluate. Usually you'll only train / evaluate a single model at a time. The train/eval functions can still capture other models or whatever else they need. But perhaps it makes sense to allow the user to pass a list of model objects (for GANs etc).

### Multiple outputs

`predict()` does not know how to handle a model that returns multiple outputs. The workaround is to `return torch.cat([output1, output2], dim=1)`.
