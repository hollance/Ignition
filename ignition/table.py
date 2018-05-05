from .imports import *
from .utils import *

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
