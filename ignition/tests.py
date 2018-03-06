from .imports import *
from .train import *
from .utils import *

def perform_tests(trainer, tests=[]):
    """Performs a number of basic tests to catch silly mistakes, so that you
    don't waste hours training a model that is broken.
    
    Inspired by https://github.com/Thenerdstation/mltest
    """
    start_time = time.time()
    
    default_tests = [
        test_trainable_parameters_changed, 
        test_frozen_parameters_not_changed,
    ]
    tests = default_tests + tests

    passed = 0
    failed = 0
    print("Running tests:")
    for i in range(len(tests)):
        #print("  %d. " % (i + 1), end="")
        print("⏱ ", end="")

        trainer._save_state()
        trainer.verbose = False
        result, msg = tests[i](trainer)
        trainer._restore_state()

        if result:
            passed += 1
            print("\r✅")
        else:
            failed += 1
            print("\r❌")
        print(msg, end="")

    if failed == 0:
        print("\nSUCCESS! 😅")
    else:
        print("\n🔥🔥🔥 %d FAILURES 🔥🔥🔥" % failed)
    print("elapsed: %.1f sec" % (time.time() - start_time))


def test_trainable_parameters_changed(trainer):
    """Performs a training step and verifies that at least one parameter 
    in every trainable layer has changed."""
    print("At least one parameter changed in the trainable layers", end="")
    passed, msg = True, ""

    trainer.fit(epochs=1, max_steps=1, max_eval_steps=0)
    
    for name, new_param in trainer.model.named_parameters():
        if new_param.requires_grad:
            old_param = trainer._saved_state["model"][name]
            if not (new_param.data != old_param).any():
                msg += "    expected changes in: %s\n" % name
                passed = False

    return passed, msg


def test_frozen_parameters_not_changed(trainer):
    """Perform a training step and verifies that no parameters in frozen
    layers changed."""
    print("No parameters changed in frozen layers", end="")
    passed, msg = True, ""

    trainer.fit(epochs=1, max_steps=1, max_eval_steps=0)
    
    for name, new_param in trainer.model.named_parameters():
        if not new_param.requires_grad:
            old_param = trainer._saved_state["model"][name]
            if not (new_param.data == old_param).all():
                msg += "    expected no changes in: %s\n" % name
                passed = False

    return passed, msg


def test_classifier_loss(eval_fn, num_classes, error_margin=0.1):
    """Verifies that the loss on an untrained model is close to -ln(1/num_classes).
    Looks at several batches from the train and validation datasets. The eval_fn 
    should return a dictionary containing 'loss'."""
    def inner(trainer):
        expected = -np.log(1/num_classes)
        print("Classifier loss is close to %g ± %g" % (expected, error_margin), end="")
        passed, msg = True, ""

        loaders = _get_loaders(trainer)
        steps = np.min([len(l[1]) for l in loaders])

        for loader_name, loader in loaders:
            results = evaluate(trainer.model, eval_fn, loader, max_steps=steps, verbose=False)
            loss = results["loss"]
            if loss < expected - error_margin or loss > expected + error_margin:
                passed = False
                msg += "    %s: loss is %g, diff: %g\n" % (loader_name, loss, np.abs(loss - expected))
        
        return passed, msg
    return inner


def test_input_range(low, high):
    """Verifies that the input data is in the given range, e.g. [-1, 1]
    for preprocessed images. Also checks that all input data is not zeros.
    Looks at a single batch from the train and validation datasets."""
    def inner(trainer):
        print("Input is between %g and %g" % (low, high), end="")
        passed, msg = True, ""

        for loader_name, loader in _get_loaders(trainer):
            batch = next(iter(loader))[0]
            passed, msg = _assert_between(batch, low, high, loader_name, passed, msg)

        return passed, msg
    return inner


def test_output_range(predict_fn, low, high):
    """Verifies that the output data is in the given range, e.g. [0, 1]
    for sigmoid activation. Also checks that all output data is not zeros. 
    Looks at a single batch from the train and validation datasets. 
    The predict_fn should grab the output that you want to check."""
    def inner(trainer):
        print("Output is between %g and %g" % (low, high), end="")
        passed, msg = True, ""
        
        for loader_name, loader in _get_loaders(trainer):
            batch = predict(trainer.model, predict_fn, loader, max_steps=1, verbose=False)
            passed, msg = _assert_between(batch, low, high, loader_name, passed, msg)

        return passed, msg
    return inner


def _assert_between(batch, low, high, name, passed, msg):
    num_too_low = (batch < low).sum()
    if num_too_low != 0:
        pct = num_too_low * 100. / batch.numel()
        msg += "    %s: %d elements out of %d too low (%.1f%%), lowest: %g\n" % \
               (name, num_too_low, batch.numel(), pct, batch.min())
        passed = False

    num_too_high = (batch > high).sum()
    if num_too_high != 0:
        pct = num_too_high * 100. / batch.numel()
        msg += "    %s: %d elements out of %d too high (%1.f%%), highest: %g\n" % \
               (name, num_too_high, batch.numel(), pct, batch.max())
        passed = False

    all_zero = (batch == 0.).all()
    if all_zero:
        msg += "    %s: all elements are zero\n" % name
        passed = False
        
    return passed, msg


def _get_loaders(trainer):
    loaders = [("train", trainer.train_loader)]
    if trainer.val_loader is not None:
        loaders += [("val", trainer.val_loader)]
    return loaders

