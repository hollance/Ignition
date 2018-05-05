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
