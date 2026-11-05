import sys
from src.parallel.applications import main
if __name__ == '__main__':
    main(['--tasks', 'pam4_nonlinear', *sys.argv[1:]])
