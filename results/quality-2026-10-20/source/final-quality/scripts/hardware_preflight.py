#!/usr/bin/env python3
"""Read-only native CUDA preflight; emits auditable JSON, never installs software."""
import argparse
import csv
import ctypes
import ctypes.util
import glob
import json
import os
import platform
import shutil
import struct
import subprocess
import sys
from pathlib import Path


def run(*args):
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT, timeout=30).strip()


def elf_machine(path):
    with open(path, 'rb') as stream:
        header = stream.read(20)
    if header[:4] != b'\x7fELF':
        raise ValueError(f'Expected native ELF executable/library: {path}')
    return struct.unpack('<H' if header[5] == 1 else '>H', header[18:20])[0]


def select_gpu(rows, requested, visible):
    tokens = visible.split(',') if visible is not None else []
    if tokens and (len(tokens) != 1 or not tokens[0] or tokens[0].startswith('MIG-')):
        raise ValueError('CUDA_VISIBLE_DEVICES must identify exactly one full GPU')
    def resolve(token):
        matches = [r for r in rows if token in (r['index'], r['uuid'])]
        if len(matches) != 1:
            raise ValueError('Select an exact GPU UUID or physical nvidia-smi index')
        return matches[0]
    chosen = resolve(requested) if requested else resolve(tokens[0]) if tokens else rows[0] if len(rows) == 1 else None
    if chosen is None:
        raise ValueError('Multiple GPUs: set GPU_UUID or a single CUDA_VISIBLE_DEVICES')
    if tokens and resolve(tokens[0])['uuid'] != chosen['uuid']:
        raise ValueError('GPU_UUID conflicts with allocated CUDA_VISIBLE_DEVICES')
    if chosen['mig.mode.current'] not in ('Disabled', '[N/A]'):
        raise ValueError('MIG must be disabled or explicitly unsupported for this full-GPU protocol')
    if float(chosen['utilization.gpu']) != 0:
        raise ValueError('Selected GPU is busy; wait for an idle allocation')
    return chosen


def normalize_bus(value):
    domain, device, function = value.lower().split(':')
    return (int(domain, 16), int(device, 16), function)


def inspect():
    if sys.version_info < (3, 10):  # noqa: UP036 -- this script also guards unconfigured hosts
        raise ValueError('Python >=3.10 required')
    machine = platform.machine()
    expected = {'x86_64': 62, 'aarch64': 183}.get(machine)
    if expected is None or elf_machine(Path(sys.executable).resolve()) != expected:
        raise ValueError('Python must be native x86_64 or aarch64')
    for name in ('nvcc', 'cmake', 'c++', 'nvidia-smi', 'git'):
        if not shutil.which(name):
            raise ValueError(f'Required command missing: {name}')
    if elf_machine(Path(shutil.which('c++')).resolve()) != expected:
        raise ValueError('Host C++ compiler does not match native host architecture')
    nvcc = Path(shutil.which('nvcc')).resolve()
    if elf_machine(nvcc) != expected:
        raise ValueError('nvcc does not match native host architecture')
    fields = 'index,uuid,pci.bus_id,name,driver_version,memory.total,memory.used,utilization.gpu,compute_cap,mig.mode.current,power.limit,clocks.sm,clocks.mem'
    rows = list(csv.DictReader(run('nvidia-smi', '--query-gpu=' + fields, '--format=csv,noheader,nounits').splitlines(), fieldnames=fields.split(','), skipinitialspace=True))
    selected = select_gpu(rows, os.environ.get('GPU_UUID'), os.environ.get('CUDA_VISIBLE_DEVICES'))
    processes = run('nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits')
    if any(line.split(',')[0].strip() == selected['uuid'] for line in processes.splitlines()):
        raise ValueError('Selected GPU has an active compute process')
    os.environ['CUDA_VISIBLE_DEVICES'] = selected['uuid']
    candidates = []
    candidates += glob.glob(str(nvcc.parent.parent / 'lib64/libcudart.so'))
    candidates += glob.glob(str(nvcc.parent.parent / 'targets/*/lib/libcudart.so'))
    candidates.append(ctypes.util.find_library('cudart'))
    runtime = None
    runtime_library = None
    for candidate in filter(None, candidates):
        if '/' in candidate and elf_machine(Path(candidate).resolve()) != expected:
            continue
        try:
            runtime = ctypes.CDLL(candidate)
            runtime_library = str(Path(candidate).resolve()) if "/" in candidate else candidate
            break
        except OSError:
            pass
    if runtime is None:
        raise ValueError('Native CUDA runtime unavailable; load the matching CUDA toolkit')
    def checked(code):
        if code != 0:
            raise ValueError(f'CUDA runtime preflight failed with error {code}')
    count = ctypes.c_int()
    checked(runtime.cudaGetDeviceCount(ctypes.byref(count)))
    if count.value != 1:
        raise ValueError('CUDA runtime must expose exactly one selected GPU')
    # cudaDeviceGetUuid is not a public cudart symbol. Resolve runtime ordinal
    # zero through its PCI address, which nvidia-smi associates with this UUID.
    bus = ctypes.create_string_buffer(32)
    checked(runtime.cudaDeviceGetPCIBusId(bus, len(bus), 0))
    if normalize_bus(bus.value.decode()) != normalize_bus(selected['pci.bus_id']):
        raise ValueError('CUDA runtime device PCI address does not match selected GPU UUID')
    runtime_memory = None
    if selected['memory.total'] in ('[N/A]', 'N/A'):
        free_bytes, total_bytes = ctypes.c_size_t(), ctypes.c_size_t()
        checked(runtime.cudaMemGetInfo(ctypes.byref(free_bytes), ctypes.byref(total_bytes)))
        runtime_memory = {'source': 'cudaMemGetInfo', 'free_bytes': free_bytes.value,
                          'total_bytes': total_bytes.value,
                          'note': 'Runtime-visible shared/device capacity; not NVML memory.total or exclusive physical GPU memory'}
    runtime_version = ctypes.c_int()
    checked(runtime.cudaRuntimeGetVersion(ctypes.byref(runtime_version)))
    arch = selected['compute_cap'].replace('.', '')
    override = os.environ.get('CUDA_ARCHITECTURES')
    if override and override != arch:
        raise ValueError(f'CUDA_ARCHITECTURES must match detected device ({arch})')
    for module in ('gpu_kernel_analyzer', 'numpy', 'yaml', 'matplotlib'):
        __import__(module)
    status = run('git', 'status', '--porcelain')
    return {'host_architecture': machine, 'python': sys.executable, 'python_version': sys.version,
            'gpu': selected, 'cuda_architectures': arch, 'cuda_runtime_version': runtime_version.value, 'cuda_runtime_library': runtime_library, 'cuda_memory_capacity': runtime_memory,
            'nvcc': str(nvcc), 'nvcc_version': run('nvcc', '--version'),
            'cmake_version': run('cmake', '--version'), 'host_compiler': run('c++', '--version'),
            'git_revision': run('git', 'rev-parse', 'HEAD'), 'git_status': status,
            'git_diff': run('git', 'diff', 'HEAD', '--'),
            'untracked_files': run('git', 'ls-files', '--others', '--exclude-standard')}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    try:
        result = inspect()
        Path(args.out).write_text(json.dumps(result, indent=2) + '\n')
    except (ValueError, OSError, subprocess.SubprocessError, ImportError) as exc:
        sys.exit(f'Preflight failed: {exc}')
