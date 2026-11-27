#!/usr/bin/env python

import importlib
import shutil
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / 'data'
LOG_DIR = ROOT_DIR / 'logs'

LABS = (
    ('lab5', ROOT_DIR / 'lab5' / 'lab5_blank.py'),
    ('lab6', ROOT_DIR / 'lab6' / 'lab6_blank.py'),
    ('lab7', ROOT_DIR / 'lab7' / 'lab7_blank.py'),
)

REQUIRED_MODULES = (
    'matplotlib',
    'numpy',
    'scipy',
    'sklearn',
    'soundfile',
    'tabulate',
    'torch',
    'torchaudio',
    'tqdm',
    'whisper',
)

REQUIRED_PATHS = (
    DATA_DIR / 'voxceleb1_dev' / 'wav',
    DATA_DIR / 'voxceleb1_test' / 'wav',
    DATA_DIR / 'voxceleb_meta' / 'iden_split.txt',
    DATA_DIR / 'voxceleb_meta' / 'veri_test2.txt',
    DATA_DIR / 'lists' / 'models.txt',
    DATA_DIR / 'musan_split',
    DATA_DIR / 'RIRS_NOISES' / 'simulated_rirs',
    DATA_DIR / 'lab3_models' / 'lab3_model_0039.pth',
)


def preflight():
    errors = []

    if shutil.which('ffmpeg') is None:
        errors.append('ffmpeg is not installed or is not available in PATH')

    for module_name in REQUIRED_MODULES:
        try:
            importlib.import_module(module_name)
        except Exception as error:
            errors.append(f'cannot import {module_name}: {error}')

    for path in REQUIRED_PATHS:
        if not path.exists():
            errors.append(f'missing path: {path}')

    try:
        import torch

        if not torch.cuda.is_available():
            errors.append('CUDA is not available; lab6 training must run on a GPU node')
    except Exception:
        pass

    if errors:
        print('Preflight failed:')
        for error in errors:
            print(f'  - {error}')
        return False

    print('Preflight passed. All dependencies, data and checkpoints are available.')
    return True


def run_lab(name, script_path):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f'{name}.log'
    print(f'Starting {name}. Log: {log_path}')

    with log_path.open('a', encoding='utf-8') as log_file:
        process = subprocess.Popen(
            [sys.executable, '-u', str(script_path)],
            cwd=ROOT_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in process.stdout:
            print(line, end='')
            log_file.write(line)
            log_file.flush()

    return_code = process.wait()
    if return_code != 0:
        raise SystemExit(
            f'{name} failed with exit code {return_code}. See {log_path}'
        )

    print(f'{name} completed successfully.')


def main():
    if not preflight():
        raise SystemExit(1)

    for name, script_path in LABS:
        run_lab(name, script_path)

    print('Labs 5, 6 and 7 completed successfully.')


if __name__ == '__main__':
    main()
