"""CPU-only checks for the actual GPU selection and isolation guards."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('hardware_preflight', Path(__file__).parents[1] / 'scripts/hardware_preflight.py')
preflight = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preflight)


def gpu(index='0', uuid='GPU-first', **changes):
    return {'index': index, 'uuid': uuid, 'mig.mode.current': 'Disabled', 'utilization.gpu': '0', **changes}


def test_selects_only_allocated_uuid():
    rows = [gpu(), gpu('1', 'GPU-second')]
    assert preflight.select_gpu(rows, None, '1')['uuid'] == 'GPU-second'
    assert preflight.select_gpu(rows, 'GPU-second', 'GPU-second')['index'] == '1'


@pytest.mark.parametrize('visible', ['', '0,1', '-1', 'MIG-first', 'GPU-fir'])
def test_invalid_visibility_rejected(visible):
    with pytest.raises(ValueError):
        preflight.select_gpu([gpu()], None, visible)


def test_ambiguous_selection_and_allocation_conflict():
    rows = [gpu(), gpu('1', 'GPU-second')]
    with pytest.raises(ValueError, match='Multiple GPUs'):
        preflight.select_gpu(rows, None, None)
    with pytest.raises(ValueError, match='conflicts'):
        preflight.select_gpu(rows, 'GPU-first', '1')


@pytest.mark.parametrize('changes', [{'utilization.gpu': '1'}, {'mig.mode.current': 'Enabled'}, {'utilization.gpu': 'N/A'}])
def test_busy_mig_and_unknown_state_rejected(changes):
    with pytest.raises(ValueError):
        preflight.select_gpu([gpu(**changes)], None, None)


def test_wrong_python_architecture_fails_before_gpu_commands(monkeypatch):
    monkeypatch.setattr(preflight.platform, 'machine', lambda: 'aarch64')
    monkeypatch.setattr(preflight, 'elf_machine', lambda path: 62)
    monkeypatch.setattr(preflight, 'run', lambda *args: pytest.fail('must reject before launching commands'))
    with pytest.raises(ValueError, match='Python must be native'):
        preflight.inspect()


def test_runtime_pci_address_normalization():
    assert preflight.normalize_bus('0000:AF:00.0') == preflight.normalize_bus('00000000:af:00.0')
    assert preflight.normalize_bus('0000:af:00.0') != preflight.normalize_bus('0000:ae:00.0')


def test_external_probes_have_bounded_timeout(monkeypatch):
    def checked(*args, **kwargs):
        assert kwargs['timeout'] == 30
        return 'probe output\n'
    monkeypatch.setattr(preflight.subprocess, 'check_output', checked)
    assert preflight.run('nvidia-smi') == 'probe output'


def test_explicit_unsupported_mig_mode_allows_full_gpu():
    assert preflight.select_gpu([gpu(**{'mig.mode.current': '[N/A]'})], None, None)['uuid'] == 'GPU-first'
    with pytest.raises(ValueError, match='MIG'):
        preflight.select_gpu([gpu(**{'mig.mode.current': 'Unknown'})], None, None)
