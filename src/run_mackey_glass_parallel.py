"""Parallel mackey_glass; forwards all options to the shared benchmarks suite."""
import sys
from src.parallel.benchmarks import main
if __name__ == '__main__':
    main(['--tasks', 'mackey_glass', *sys.argv[1:]])
