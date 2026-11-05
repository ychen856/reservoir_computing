import sys
from src.parallel.applications import main
if __name__ == '__main__':
    main(['--tasks', 'pam4', *sys.argv[1:]])
