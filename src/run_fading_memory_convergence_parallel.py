"""Parallel fading_memory_convergence; forwards all options to the shared properties suite."""
import sys
from src.parallel.properties import main
if __name__ == '__main__':
    main(['--tasks', 'fading_memory_convergence', *sys.argv[1:]])
