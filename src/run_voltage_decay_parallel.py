"""Parallel voltage_decay; forwards all options to the shared properties suite."""
import sys
from src.parallel.properties import main
if __name__ == '__main__':
    main(['--tasks', 'voltage_decay', *sys.argv[1:]])
