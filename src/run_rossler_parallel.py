"""Parallel rossler; forwards all options to the shared benchmarks suite."""
import sys
from src.parallel.benchmarks import main
if __name__ == '__main__':
    main(['--tasks', 'rossler', *sys.argv[1:]])
