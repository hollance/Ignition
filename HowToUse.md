# How to use Ignition

## Training loop

The primary idea in Ignition is that you should be able to use the same training/evaluation loop for any kind of model. The training loop makes no assumptions about the model, the loss function(s), the optimizer, etc.

To keep the training loop independent of the model and training procedure, you need to create an *eval function* and a *train function* that process a single batch of examples. The training loop then calls these functions to do all the real work.

### The eval function

Evaluating the performance of a model on a validation set is done with the `evaluate()` function:

```python
evaluate(model, eval_fn, testloader)
```

where `testloader` is the `torch.utils.data.DataLoader` object that provides the data, and `eval_fn` is the so-called *eval function*. A typical eval function is:

```python
def eval_fn(model_to_eval, x, y):
    return evaluate_on_batch(model_to_eval, x, y, crossentropy_loss, ["loss", "acc"])
```

Notice how `eval_fn` takes 3 parameters: the model to evaluate (an `nn.Module` object), a batch of inputs `x`, and a batch of corresponding labels `y`.

Ignition provides a built-in eval function called `evaluate_on_batch()` that is suitable for basic classifiers. It uses a cross-entropy loss and returns the loss and categorical accuracy across the examples in the batch. So in the example above we just use that built-in eval function.

But you can also create more complicated eval functions. For example, the following eval function computes the MSE between the output of a teacher network and a student network (the model to evaluate):

```python
def eval_fn(model_to_eval, x):
    x = make_var(x)
    teacher_outputs = teacher.forward(x)
    student_outputs = model_to_eval.forward(x)
    loss = torch.mean(torch.sum(0.5 * (teacher_outputs - student_outputs)**2, dim=1))
    return { "loss": loss.data[0] }
```

Notice that here the `eval_fn` does not get labels. That's because the `DataLoader` we're using here does not return labels; instead, we use the output of the teacher as the (pseudo)labels.

### The train function

To train a model you also need to provide a *train function*:

```python
def train_fn(model_to_train, x, y):
    return fit_on_batch(model_to_train, x, y, crossentropy_loss, optimizer)
```

The train function takes at least two parameters: the model to train (an `nn.Module` object) and a batch of inputs `x`. If the `DataLoader` also provides labels (which is usually the case), then these are passed to the train function as well.

For a basic classifier model you can use the built-in `fit_on_batch()` function as shown above, but you can also write your own:

```python
teacher = ...
optimizer = ...

def train_fn(model_to_train, x):
    optimizer.zero_grad()
    
    x = make_var(x)
    teacher_outputs = teacher.forward(x)
    student_outputs = model_to_train.forward(x)

    loss = torch.mean(torch.sum(0.5 * (teacher_outputs - student_outputs)**2, dim=1))
    loss.backward()
    optimizer.step()
    
    return { "loss": loss.data[0] }    
```

Note that here the train function does not have labels (and therefore has no `y` parameter); the outputs of the teacher network are used as the labels for the student. (In fact, the code here is similar to that in the `eval_fn`, except that we also do a backward pass and an optimizer step.)

### Trainer

The training loop is managed by a `Trainer` object:

```python
trainer = Trainer(model, optimizer, train_fn, train_loader, eval_fn, test_loader)
trainer.fit(epochs=10)
```

Note that we just pass in the train function, the eval function, and the `DataLoader`s that provide the training and validation data.

The trainer keeps track of training history: how many epochs have elapsed, loss and accuracy curves, etc.

```python
trainer.history.plot_loss()
trainer.history.plot_accuracy()
trainer.history.best_epoch()
```

### Callbacks

You can provide callbacks to the training loop, for example to save model checkpoints:

```python
trainer.callbacks = [ SaveModel(...) ]
```

### Hyperparameters

For hyperparameters that you want to change over time, you can do the following:

```python
trainer.hyper = { "lambd": 4. }
```

Now the trainer will automatically pass this new `lambd` parameter to the train function so that you can use it as part of the training procedure:

```python
def train_fn(model_to_train, x, y, lambd):
    ...
```

The Trainer also passes this hyperparameter to the callbacks. If you wanted to decay this hyperparameter over time, for example, you could use the following callback:

```python
trainer.callbacks = [ LinearDecay("lambd", start_value=4, end_value=1, max_epochs=60) ]
```

### Metrics

Which metrics will be shown during training depends solely on the train and eval functions. The training loop will gather whatever metrics these functions produce and automatically logs them. (There is no need to tell `Trainer` which metrics to collect.)
