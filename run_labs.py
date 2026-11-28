#!/usr/bin/env python

import argparse
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

REQUIRED_MODULES = {
    'lab5': (
        'matplotlib', 'numpy', 'scipy', 'sklearn', 'soundfile',
        'torch', 'torchaudio', 'tqdm',
    ),
    'lab6': (
        'matplotlib', 'numpy', 'scipy', 'sklearn', 'soundfile',
        'torch', 'torchaudio', 'tqdm', 'whisper',
    ),
    'lab7': (
        'matplotlib', 'numpy', 'sklearn', 'soundfile',
        'tabulate', 'torch', 'torchaudio',
    ),
}

REQUIRED_PATHS = {
    'lab5': (
        DATA_DIR / 'voxceleb1_dev' / 'wav',
        DATA_DIR / 'voxceleb1_test' / 'wav',
        DATA_DIR / 'voxceleb_meta' / 'veri_test2.txt',
        DATA_DIR / 'lists' / 'models.txt',
    ),
    'lab6': (
        DATA_DIR / 'voxceleb1_dev' / 'wav',
        DATA_DIR / 'voxceleb1_test' / 'wav',
        DATA_DIR / 'voxceleb_meta' / 'iden_split.txt',
        DATA_DIR / 'voxceleb_meta' / 'veri_test2.txt',
        DATA_DIR / 'musan_split',
        DATA_DIR / 'RIRS_NOISES' / 'simulated_rirs',
    ),
    'lab7': (
        DATA_DIR / 'voxceleb1_test' / 'wav',
        DATA_DIR / 'lab3_models' / 'lab3_model_0039.pth',
    ),
}


def preflight(lab_names):
    errors = []

    if 'lab5' in lab_names and shutil.which('ffmpeg') is None:
        errors.append('ffmpeg is not installed or is not available in PATH')

    module_names = sorted({
        module_name
        for lab_name in lab_names
        for module_name in REQUIRED_MODULES[lab_name]
    })
    for module_name in module_names:
        try:
            importlib.import_module(module_name)
        except Exception as error:
            errors.append(f'cannot import {module_name}: {error}')

    paths = {
        path
        for lab_name in lab_names
        for path in REQUIRED_PATHS[lab_name]
    }
    for path in sorted(paths):
        if not path.exists():
            errors.append(f'missing path: {path}')

    if 'lab6' in lab_names:
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

    print(f'Preflight passed for {", ".join(lab_names)}.')
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--start-at',
        choices=tuple(name for name, _ in LABS),
        default='lab5',
        help='start with this lab and run the remaining labs in order',
    )
    args = parser.parse_args()

    start_index = next(
        index for index, (name, _) in enumerate(LABS)
        if name == args.start_at
    )
    selected_labs = LABS[start_index:]
    lab_names = [name for name, _ in selected_labs]

    if not preflight(lab_names):
        raise SystemExit(1)

    for name, script_path in selected_labs:
        run_lab(name, script_path)

    print(f'{", ".join(lab_names)} completed successfully.')


if __name__ == '__main__':
    main()
