from .imports import *
from .table import *
from .utils import *

def print_parameter_sizes(model):
    """Prints the shapes of all trainable parameters in the model."""
    model.train(False)

    table = Table(["Parameter", "Shape", "Count", "Train?"], widths=[30, 24, 12, 6])
    print(table.header(with_line=False))

    for name, params in model.named_parameters():
        trainable = "Yes" if params.requires_grad else "No"
        print(table.row([ name, pretty_size(params.size()), params.numel(), trainable ]))
    
    from functools import reduce
    total = reduce(lambda x, y: x + y.numel(), model.parameters(), 0)
    print("Total params: {:,}".format(total))


def print_activation_sizes(model, input_size=None, input_tensor=None):
    """Prints the input and output shapes of the modules in the model.
    
    Note: this puts the model into evaluation mode, otherwise it may change the
    running averages of any batch norm layers, etc.
    """
    model.train(False)

    sizes = []
    def grab_module_shapes(name):
        def closure(module, inp, out):
            # TODO: If something is a tuple, should really grab all the
            # sizes instead of just the first one.
            i = inp
            o = out
            while type(i) in [list, tuple]: i = i[0]
            while type(o) in [list, tuple]: o = o[0]
            unknown = torch.Size([-1])
            isize = i.size() if hasattr(i, "size") else unknown
            osize = o.size() if hasattr(o, "size") else unknown
            sizes.append((name, isize, osize))
        return closure    
    
    handles = []
    for name, module in model.named_children():
        handle = module.register_forward_hook(grab_module_shapes(name))
        handles.append(handle)    
 
    if input_tensor is not None:
        inp = make_cuda(input_tensor)
    else:
        inp = make_cuda(torch.randn(input_size))

    out = model(inp)
    unregister_hooks(handles)
    del inp, out, handles

    table = Table(["Module", "Input Size", "Output Size"], widths=[30, 24, 24])
    print(table.header(with_line=False))

    for name, in_size, out_size in sizes:
        print(table.row([ name, pretty_size(in_size), pretty_size(out_size)]))    
    

# Based on code from https://github.com/ncullen93/torchsample
from collections import OrderedDict

def model_summary(m, input_size):
    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split('.')[-1].split("'")[0]
            module_idx = len(summary)

            m_key = '%s-%i' % (class_name, module_idx+1)
            summary[m_key] = OrderedDict()
            summary[m_key]['input_shape'] = list(input[0].size())
            summary[m_key]['input_shape'][0] = -1
            summary[m_key]['output_shape'] = list(output.size())
            summary[m_key]['output_shape'][0] = -1

            params = 0
            if hasattr(module, 'weight'):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]['trainable'] = module.weight.requires_grad
            if hasattr(module, 'bias') and module.bias is not None:
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]['nb_params'] = params

        if (not isinstance(module, nn.Sequential) and
            not isinstance(module, nn.ModuleList) and
            not (module == m)):
            hooks.append(module.register_forward_hook(hook))

    summary = OrderedDict()
    hooks = []
    m.apply(register_hook)

    if isinstance(input_size[0], (list, tuple)):
        x = [make_cuda(torch.rand(1, *in_size)) for in_size in input_size]
    else:
        x = [make_cuda(torch.rand(1, *input_size))]
    m(*x)

    for h in hooks: h.remove()

    return summary


def plot_graph(model, input_size):
    model.train(False)

    # Create a dictionary of the names for all parameters.
    params_dict = {}
    for name, params in model.named_parameters():
        params_dict[name] = params
    
    # Do a forward pass of the graph.
    inp = make_cuda(torch.randn(input_size))
    out = model(inp)

    # Create the Graphviz object.
    return make_dot(out, params_dict)


# Based on code from https://github.com/szagoruyko/functional-zoo/blob/master/visualize.py
from graphviz import Digraph

def make_dot(output, params=None):
    """Produces Graphviz representation of PyTorch autograd graph

    Blue nodes are the Tensors that require grad, orange are Tensors
    saved for backward in torch.autograd.Function

    Args:
        output: output Tensor
        params: dict of (name, Parameter), optional
    """
    if params is not None:
        param_map = {id(v): k for k, v in params.items()}

    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()

    def add_nodes(obj):
        if obj not in seen:
            if torch.is_tensor(obj):
                dot.node(str(id(obj)), size_to_str(obj.size()), fillcolor='orange')
            elif hasattr(obj, 'variable'):
                u = obj.variable
                name = param_map[id(u)] + '\n' if params is not None else ''
                node_name = name + pretty_size(u.size())
                dot.node(str(id(obj)), node_name, fillcolor='lightblue')
            else:
                dot.node(str(id(obj)), str(type(obj).__name__))
            seen.add(obj)
            if hasattr(obj, 'next_functions'):
                for u in obj.next_functions:
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(obj)))
                        add_nodes(u[0])
            if hasattr(obj, 'saved_tensors'):
                for t in obj.saved_tensors:
                    dot.edge(str(id(t)), str(id(obj)))
                    add_nodes(t)

    add_nodes(output.grad_fn)
    return dot
