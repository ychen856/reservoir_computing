"""Parallel narma10; forwards all options to the shared benchmarks suite."""
import sys
from src.parallel.benchmarks import main
if __name__ == '__main__':
    main(['--tasks', 'narma10', *sys.argv[1:]])
