from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

LOG_FILENAME = None  # Set filename before first call to log (it's currently done in the manager constructor)


def log(*args):
    """Writes to standard output and to logfile."""
    print_string = make_print_string(args)
    print(*args)
    write_to_log_file(print_string, 'logs/')


def make_print_string(args):
    st = ''
    for i in args:
        st += ' ' + str(i)
    return st.lstrip(' ') + '\n'


def write_to_log_file(print_string, directory):
    if LOG_FILENAME is not None:
        with open(directory + LOG_FILENAME, "a") as logfile:
            logfile.write(print_string)
    else:
        raise Exception("No logfile name when trying to print to logfile")
