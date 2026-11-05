"""Parallel generalization_rank; forwards all options to the shared properties suite."""
import sys
from src.parallel.properties import main
if __name__ == '__main__':
    main(['--tasks', 'generalization_rank', *sys.argv[1:]])
