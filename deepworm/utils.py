import sys


def print_percent(percent):
    sys.stdout.write('\r')
    sys.stdout.write("[%-20s] %d%%" % ('=' * percent, 5 * percent))
    sys.stdout.flush()