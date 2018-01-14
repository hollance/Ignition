from .imports import *
from .utils import *


def a_slash_b(a, b):
    """Creates the string "a/b" where a is as wide as b."""
    b_str = str(b)
    return "{a: >{fill}}/{b}".format(a=a, b=b_str, fill=len(b_str))


class ProgressBar:
    def __init__(self, total, bar_length=30, show_eta=True):
        self.total = total
        self.bar_length = bar_length
        self.show_eta = show_eta
        self.last_len = 0
    
    def update(self, current, msg=None):
        now = time.time()
        if current == 0:
            self.start_time = now

        if current > self.total:
            current = self.total

        s = a_slash_b(current + 1, self.total)
        s += self._bar(current)
        
        elapsed = now - self.start_time
        if self.show_eta:
            time_per_unit = elapsed / current if current > 0 else 0
            eta = time_per_unit * (self.total - current)
            s += "ETA " + self._format_time(eta)
        else:
            s += self._format_time(elapsed)
        
        if msg: s += " - " + msg
        self._write(s)

    def end(self, msg=None):
        if msg: self._write(msg)
        sys.stdout.write("\n")
        sys.stdout.flush()

    def _bar(self, current):
        fill_len = int(self.bar_length * (current + 1) / self.total)
        rest_len = self.bar_length - fill_len

        #lbracket = "["
        #rbracket = "]"
        #block = "="
        #head = ">"
        #dot = "."

        lbracket = ""
        rbracket = ""
        block = chr(0x2588)
        head = chr(0x2588)
        dot = chr(0x00b7)
        
        s = " " + lbracket + block * max(fill_len - 1, 0)
        if fill_len > 0: s += head if rest_len > 0 else block
        s += dot * rest_len + rbracket + " "
        return s
        
    def _write(self, s):
        # Erase any leftover contents from the previous iteration
        len_s = len(s)        
        s += " " * max(self.last_len - len_s, 0)
        self.last_len = len_s

        # Write to stdout
        s += "\r"
        sys.stdout.write(s)
        sys.stdout.flush()

    def _format_time(self, t):
        if t > 3600:
            return "%d:%02d:%02d" % (t // 3600, (t % 3600) // 60, t % 60)
        elif t > 60:
            return "%d:%02d" % (t // 60, t % 60)
        else:
            return "%ds" % t


class Table:
    """General-purpose table"""
    def __init__(self, column_names, widths={}, formats={}):
        self.column_names = column_names
        if isinstance(widths, list):
            self.widths = dict(zip(column_names, widths))
        else:
            self.widths = widths
        self.formats = formats
        self.require_header = False
        
    def header(self, with_line=False):
        columns = []
        for name in self.column_names:
            l = self._width_for(name)
            s = name.ljust(l)
            columns.append(s[:l])

        s = " | ".join(columns)
        
        if with_line:
            s += "\n" + (len(s) * "—")

        return s
        
    def row(self, values):
        if self.require_header:
            self.header()
            self.require_header = False
        
        columns = []
        for i in range(len(values)):
            name = self.column_names[i]
            value = values[i]

            s = self._format_for(name, value) % value
            l = self._width_for(name)
            
            if isinstance(values[i], str):
                s = s.ljust(l)
            else:
                s = s.rjust(l)

            columns.append(s[:l])

        return " | ".join(columns)
        
    def _format_for(self, name, value):
        if name in self.formats:
            return self.formats[name]
        elif isinstance(value, str):
            return "%s"
        elif isinstance(value, float):
            return "%.5f"
        elif isinstance(value, int):
            return "%d"
        else:
            raise ValueError("Unknown data type for column %s", name)
            
    def _width_for(self, name):
        if name in self.widths:
            return self.widths[name]
        else:
            return max(len(name), 8)


def print_parameter_sizes(model):
    """Prints the shapes of all trainable parameters in the model."""
    model.train(False)

    table = Table(["Parameter", "Size", "Count", "Train?"], widths=[30, 24, 12, 6])
    print(table.header(with_line=False))

    for name, params in model.named_parameters():
        trainable = "Yes" if params.requires_grad else "No"
        print(table.row([ name, pretty_size(params.size()), params.numel(), trainable ]))
    
    from functools import reduce
    total = reduce(lambda x, y: x + y.numel(), model.parameters(), 0)
    print("Total params: {:,}".format(total))


def print_activation_sizes(model, input_size):
    """Prints the input and output shapes of the modules in the model.
    
    Note: this puts the model into evaluation mode, otherwise it may change the
    running averages of any batch norm layers, etc.
    """
    model.train(False)

    sizes = []
    def grab_module_shapes(name):
        def closure(module, inp, out):
            sizes.append((name, inp[0].size(), out.size()))
        return closure    
    
    handles = []
    for name, module in model.named_children():
        handle = module.register_forward_hook(grab_module_shapes(name))
        handles.append(handle)    

    inp = make_var(torch.randn(input_size), volatile=True)
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
        x = [make_cuda(Variable(torch.rand(1, *in_size))) for in_size in input_size]
    else:
        x = [make_cuda(Variable(torch.rand(1, *input_size)))]
    m(*x)

    for h in hooks: h.remove()

    return summary


def plot_graph(model, input_size):
    model.train(False)

    # Create a dictionary of the names for all Variable objects.
    params_dict = {}
    for name, params in model.named_parameters():
        params_dict[name] = params
    
    # Do a forward pass of the graph.
    inp = make_var(torch.randn(input_size))
    out = model(inp)
    
    # Create the Graphviz object.
    return make_dot(out, params_dict)


# Based on code from https://github.com/szagoruyko/functional-zoo/blob/master/visualize.py
from graphviz import Digraph

def make_dot(var, params=None):
    """ Produces Graphviz representation of PyTorch autograd graph

    Blue nodes are the Variables that require grad, orange are Tensors
    saved for backward in torch.autograd.Function

    Args:
        var: output Variable
        params: dict of (name, Variable), optional
    """
    if params is not None:
        #assert isinstance(params.values()[0], Variable)
        param_map = {id(v): k for k, v in params.items()}

    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()

    def add_nodes(var):
        if var not in seen:
            if torch.is_tensor(var):
                dot.node(str(id(var)), size_to_str(var.size()), fillcolor='orange')
            elif hasattr(var, 'variable'):
                u = var.variable
                name = param_map[id(u)] + '\n' if params is not None else ''
                node_name = name + pretty_size(u.size())
                dot.node(str(id(var)), node_name, fillcolor='lightblue')
            else:
                dot.node(str(id(var)), str(type(var).__name__))
            seen.add(var)
            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(var)))
                        add_nodes(u[0])
            if hasattr(var, 'saved_tensors'):
                for t in var.saved_tensors:
                    dot.edge(str(id(t)), str(id(var)))
                    add_nodes(t)

    add_nodes(var.grad_fn)
    return dot
