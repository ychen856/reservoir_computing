"""Amplitude-only entry point; all shared runner flags remain available."""
import sys
from .run_parallel_integrals import main

if __name__ == '__main__':
    if '--encodings' in sys.argv:
        raise SystemExit('Use the shared runner for --encodings; this entry point is amplitude-only.')
    if '--output' not in sys.argv:
        sys.argv.extend(['--output', 'results/parallel_amplitude_integrals'])
    sys.argv.extend(['--encodings', 'amplitude'])
    main()
