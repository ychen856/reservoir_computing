"""Parallel kernel_rank; forwards all options to the shared properties suite."""
import sys
from src.parallel.properties import main
if __name__ == '__main__':
    main(['--tasks', 'kernel_rank', *sys.argv[1:]])
