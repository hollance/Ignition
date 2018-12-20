from .imports import *

# Helper functions for CUDA tensors.

gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def print_cuda_info():
    print("PyTorch version:", torch.__version__)
    if torch.cuda.is_available():
        print("CUDA version:", torch.version.cuda)
        print("cuDNN version:", torch.backends.cudnn.version())
        print("Available CUDA devices:")
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            caps = torch.cuda.get_device_capability(i)
            is_current = " (current)" if torch.cuda.current_device() == i else ""
            print("%d: %s, capability %d.%d%s" % (i, name, caps[0], caps[1], is_current))
    else:
        print("CUDA not available")


def is_cuda(xs):
    """Whether the tensors in the list live on the GPU."""
    return list(map(lambda x: x.is_cuda, xs))


def to_tensor(x, dtype=np.float32, requires_grad=False, device=gpu):
    """Converts a value into a Tensor. If it is a numpy array, this uses
    torch.from_numpy(), otherwise torch.tensor() which copies the data."""
    if isinstance(x, np.ndarray): 
        x = torch.from_numpy(x.astype(dtype))
    elif type(x) != torch.Tensor:
        x = torch.tensor(x)
    x = x.to(device)
    x.requires_grad_(requires_grad)
    return x


def to_numpy(x):
    """Converts a Tensor into a numpy array (if necessary)."""
    if isinstance(x, np.ndarray): 
        return x
    if isinstance(x, torch.autograd.Variable):
        x = x.data
    return x.cpu().numpy()


def dump_tensors(gpu_only=True):
    """Prints a list of the Tensors being tracked by the garbage collector."""
    import gc, warnings
    total_size = 0

    # Deprecated objects may give warnings when we use torch.is_tensor().
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj):
                    if not gpu_only or obj.is_cuda:
                        print("%s:%s%s%s %s" % (type(obj).__name__, 
                                                " GPU" if obj.is_cuda else "",
                                                " pinned" if obj.is_pinned else "",
                                                " grad" if obj.requires_grad else "", 
                                                pretty_size(obj.size())))
                        total_size += obj.numel()
                elif hasattr(obj, "data") and torch.is_tensor(obj.data):
                    if not gpu_only or obj.is_cuda:
                        print("%s → %s:%s%s%s%s %s" % (type(obj).__name__, 
                                                       type(obj.data).__name__, 
                                                       " GPU" if obj.is_cuda else "",
                                                       " pinned" if obj.data.is_pinned else "",
                                                       " grad" if obj.requires_grad else "", 
                                                       " volatile" if obj.volatile else "",
                                                       pretty_size(obj.data.size())))
                        total_size += obj.data.numel()
            except Exception as e:
                pass        
    print("Total size:", total_size)


# Miscellaneous helper functions.

def divup(a, b):
    """Divides a by b and rounds up to the nearest integer."""
    return (a + b - 1) // b


# http://anandology.com/blog/using-iterators-and-generators/
class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.it)


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g


def save_pickled(obj, filename, protocol=pickle.HIGHEST_PROTOCOL):
    with open(filename + ".pkl", "wb") as f:
        pickle.dump(obj, f, protocol)


def load_pickled(filename):
    with open(filename + ".pkl", "rb") as f:
        return pickle.load(f)


def mkdir_if_needed(dir_name):
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name, exist_ok=True)


def apply_on_all(seq, method, *args, **kwargs):
    """Performs method on all objects from the sequence."""
    if seq:
        for obj in seq:
            getattr(obj, method)(*args, **kwargs)

            
# Neural network helpers

def unregister_hooks(handles):
    for handle in handles:
        handle.remove()

        
def flattened_size(x):
    """Computes the number of features needed to flatten a tensor so
    it can be used with a fully-connected layer."""
    return numel_from_size(x.size()[1:])


def numel_from_size(size):
    """Multiplies all the dimensions in a torch.Size object."""
    s = 1
    for i in size:
        s *= i
    return s


def pretty_size(size):
    """Pretty prints a torch.Size object"""
    assert(isinstance(size, torch.Size))
    return " × ".join(map(str, size))


def trainable_parameters(net):
    """Returns a list of all trainable parameters in a model."""
    return [p for p in net.parameters() if p.requires_grad]


def freeze_model(model):
    for p in model.parameters():
        p.requires_grad = False


def unfreeze_model(model):
    for p in model.parameters():
        p.requires_grad = True


def freeze_until(model, param_name):
    """Freezes the model's layers until the specified parameter object.
    Example of param_name is 'classifier.weight'."""
    found_name = False
    for name, params in model.named_parameters():
        if name == param_name:
            found_name = True
        params.requires_grad = found_name


def forward(model, layers, input_tensor):
    """Runs the model on the given output and captures the outputs of the
    specified layers. Useful for looking at an intermediate layer."""
    outs = []
    handles = []
    for layer in layers:
        handle = layer.register_forward_hook(lambda module, inp, out: outs.append(out))
        handles.append(handle)
    with torch.no_grad():        
        output = model.forward(input_tensor.to(gpu))
    unregister_hooks(handles)
    return outs
