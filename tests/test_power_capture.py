import math

import pytest

from gpu_kernel_analyzer.power import summarize_power


def trace(watts=(100, 200, 300)):
    return [dict(epoch_s=float(i), monotonic_s=float(i), power_watts=w, query_duration_s=0.01)
            for i, w in enumerate(watts)]


def test_power_integral_uses_elapsed_time_and_trapezoids():
    result = summarize_power(trace(), 0, 2)
    assert result['estimated_energy_j'] == 400
    assert result['mean_power_w'] == 200
    assert result['coverage_fraction'] == 1


def test_power_only_integrates_covered_interval():
    result = summarize_power(trace(), -0.5, 2.5)
    assert result['sample_span_s'] == 2
    assert result['estimated_energy_j'] == 400
    assert result['coverage_fraction'] == pytest.approx(2 / 3)


@pytest.mark.parametrize('problem', ['missing', 'gap', 'clock', 'invalid', 'coverage'])
def test_power_rejects_unusable_traces(problem):
    rows = trace()
    start, end = 0, 2
    if problem == 'missing':
        rows.pop()
    if problem == 'gap':
        rows[1]['epoch_s'] = 0.2
    if problem == 'clock':
        rows[1]['monotonic_s'] = 1.5
    if problem == 'invalid':
        rows[1]['power_watts'] = math.nan
    if problem == 'coverage':
        end = 4
    with pytest.raises(ValueError):
        summarize_power(rows, start, end)


def test_query_rejects_unsupported_power_and_multiple_devices(monkeypatch):
    from types import SimpleNamespace

    from gpu_kernel_analyzer.power import query_gpu

    output = 'GPU-test, H100, N/A, 100, 50, 1500, 1600, 45\n'
    monkeypatch.setattr('gpu_kernel_analyzer.power.subprocess.run', lambda *a, **k: SimpleNamespace(stdout=output))
    with pytest.raises(ValueError, match='positive power'):
        query_gpu('GPU-test')
    output = 'GPU-test, H100, 400, 100, 50, 1500, 1600, 45\n' * 2
    with pytest.raises(ValueError, match='exactly one'):
        query_gpu('GPU-test')


def test_query_is_bounded_and_preserves_physical_identity(monkeypatch):
    from types import SimpleNamespace

    from gpu_kernel_analyzer.power import query_gpu

    def run(command, **kwargs):
        assert command[1:3] == ['-i', 'GPU-test']
        assert kwargs['timeout'] == 5
        return SimpleNamespace(stdout='GPU-test, H100, 400.5, 100, 50, 1500, N/A, 45\n')

    monkeypatch.setattr('gpu_kernel_analyzer.power.subprocess.run', run)
    row = query_gpu('GPU-test')
    assert row['gpu_uuid'] == 'GPU-test'
    assert row['power_watts'] == 400.5
    assert row['memory_clock_mhz'] is None
