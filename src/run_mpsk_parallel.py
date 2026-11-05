import sys
from src.parallel.applications import main
if __name__ == '__main__':
    main(['--tasks', 'mpsk', *sys.argv[1:]])
