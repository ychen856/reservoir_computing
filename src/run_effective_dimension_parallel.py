"""Parallel effective_dimension; forwards all options to the shared properties suite."""
import sys
from src.parallel.properties import main
if __name__ == '__main__':
    main(['--tasks', 'effective_dimension', *sys.argv[1:]])
