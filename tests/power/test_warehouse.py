"""Warehouse contract tests using the archived measured H100 capture."""
import csv
import hashlib
import json
import shutil
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from gpu_power_pipeline.warehouse import WarehouseError, ingest_capture

SOURCE = Path(__file__).resolve().parents[2] / 'results/h100-power-2026-09-22'


@pytest.fixture
def capture(tmp_path):
    target = tmp_path / 'capture'
    shutil.copytree(SOURCE, target)
    return target


def rewrite(capture, filename, transform):
    path = capture / filename
    path.write_text(transform(path.read_text()))
    manifest = json.loads((capture / 'manifest.json').read_text())
    manifest['artifacts_sha256'][filename] = hashlib.sha256(path.read_bytes()).hexdigest()
    (capture / 'manifest.json').write_text(json.dumps(manifest))


def test_real_capture_idempotence_and_sql_statistics(tmp_path):
    database = tmp_path / 'power.sqlite'
    assert ingest_capture(database, SOURCE)
    assert not ingest_capture(database, SOURCE)
    with sqlite3.connect(database) as conn:
        assert conn.execute('SELECT COUNT(*) FROM runs').fetchone()[0] == 1
        manifest = json.loads((SOURCE / 'manifest.json').read_text())
        assert conn.execute('SELECT COUNT(*) FROM samples').fetchone()[0] == manifest['telemetry_samples']
        summaries = json.loads((SOURCE / 'summary.json').read_text())
        for summary in summaries:
            count, watts = conn.execute(
                'SELECT sample_count,mean_power_w FROM steady_trial_stats WHERE trial=? AND scenario_id=?',
                (summary['trial'], summary['scenario_id'])).fetchone()
            assert count == summary['sample_count']
            assert watts == pytest.approx(summary['mean_power_w'], rel=1e-10)
        for workload, count, watts in conn.execute('SELECT workload,sample_count,mean_power_w FROM device_workload_stats'):
            selected = [s for s in summaries if s['kernel'] == workload]
            assert count == sum(s['sample_count'] for s in selected)
            assert watts == pytest.approx(sum(s['estimated_energy_j'] for s in selected) /
                                         sum(s['sample_span_s'] for s in selected))
        assert conn.execute('PRAGMA foreign_key_check').fetchall() == []


def test_inconsistent_hash_rejected_before_database_write(capture, tmp_path):
    (capture / 'telemetry.csv').write_text('corrupted')
    database = tmp_path / 'power.sqlite'
    with pytest.raises(WarehouseError, match='SHA-256 mismatch'):
        ingest_capture(database, capture)
    assert not database.exists()


def test_duplicate_sample_identity_rejected_even_with_updated_hash(capture, tmp_path):
    rewrite(capture, 'telemetry.csv', lambda text: text + text.splitlines()[1] + '\n')
    with pytest.raises(WarehouseError, match='duplicate or invalid sample identity'):
        ingest_capture(tmp_path / 'power.sqlite', capture)


def test_nan_rejected_even_with_updated_hash(capture, tmp_path):
    def corrupt(text):
        rows = list(csv.reader(text.splitlines()))
        rows[1][rows[0].index('power_watts')] = 'nan'
        return '\n'.join(','.join(row) for row in rows) + '\n'
    rewrite(capture, 'telemetry.csv', corrupt)
    with pytest.raises(WarehouseError, match='physical bounds'):
        ingest_capture(tmp_path / 'power.sqlite', capture)


def test_conflicting_run_rejected_without_changing_existing_data(capture, tmp_path):
    database = tmp_path / 'power.sqlite'
    ingest_capture(database, capture)
    manifest = json.loads((capture / 'manifest.json').read_text())
    manifest['source_dirty'] = not manifest['source_dirty']
    (capture / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(WarehouseError, match='conflicting content'):
        ingest_capture(database, capture)
    with sqlite3.connect(database) as conn:
        assert conn.execute('SELECT COUNT(*) FROM runs').fetchone()[0] == 1
        assert conn.execute('SELECT COUNT(*) FROM samples').fetchone()[0] == manifest['telemetry_samples']


def test_mid_insert_failure_rolls_back_entire_capture(capture, tmp_path):
    database = tmp_path / 'power.sqlite'
    ingest_capture(database, SOURCE)
    # Use an initialized empty warehouse to inject a failure after parent inserts.
    with sqlite3.connect(database) as conn:
        for table in ('samples', 'trials', 'artifacts', 'runs'):
            conn.execute(f'DELETE FROM {table}')
        conn.execute("CREATE TRIGGER fail_samples BEFORE INSERT ON samples WHEN NEW.epoch_s > 0 BEGIN SELECT RAISE(ABORT,'injected failure'); END")
    with pytest.raises(sqlite3.IntegrityError, match='injected failure'):
        ingest_capture(database, capture)
    with sqlite3.connect(database) as conn:
        for table in ('runs', 'artifacts', 'trials', 'samples'):
            assert conn.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0] == 0


def test_concurrent_same_run_has_one_writer(tmp_path):
    database = tmp_path / 'power.sqlite'
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: ingest_capture(database, SOURCE), range(2)))
    assert sorted(results) == [False, True]


def replace_sensor(text, sensor, value):
    import io
    reader = csv.DictReader(io.StringIO(text))
    rows = list(reader)
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=reader.fieldnames)
    writer.writeheader()
    for row in rows:
        row[sensor] = value
        writer.writerow(row)
    return output.getvalue()


def test_unavailable_optional_sensor_is_preserved_without_imputation(capture, tmp_path):
    for filename in ('telemetry.csv', 'steady_telemetry.csv'):
        rewrite(capture, filename, lambda text: replace_sensor(text, 'memory_clock_mhz', ''))
    database = tmp_path / 'power.sqlite'
    assert ingest_capture(database, capture)
    with sqlite3.connect(database) as conn:
        rows = conn.execute('SELECT raw_json FROM samples').fetchall()
        assert rows
        assert all(json.loads(row[0])['memory_clock_mhz'] == '' for row in rows)


@pytest.mark.parametrize('value', ['N/A', 'nan', 'inf', '-1', ' '])
def test_present_invalid_optional_sensor_still_rejected(capture, tmp_path, value):
    rewrite(capture, 'telemetry.csv', lambda text: replace_sensor(text, 'memory_clock_mhz', value))
    with pytest.raises(WarehouseError, match='memory_clock_mhz'):
        ingest_capture(tmp_path / 'power.sqlite', capture)


def test_missing_required_power_still_rejected(capture, tmp_path):
    rewrite(capture, 'telemetry.csv', lambda text: replace_sensor(text, 'power_watts', ''))
    with pytest.raises(WarehouseError, match='power_watts'):
        ingest_capture(tmp_path / 'power.sqlite', capture)
