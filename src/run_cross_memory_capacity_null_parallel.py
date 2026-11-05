"""Parallel cross_memory_capacity_null; forwards all options to the shared properties suite."""
import sys
from src.parallel.properties import main
if __name__ == '__main__':
    main(['--tasks', 'cross_memory_capacity_null', *sys.argv[1:]])
