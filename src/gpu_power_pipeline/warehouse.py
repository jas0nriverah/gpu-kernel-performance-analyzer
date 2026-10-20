"""Transactional, provenance-checked SQLite storage for measured board-power captures."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import sqlite3
from pathlib import Path

SCHEMA_VERSION = 1
REQUIRED = ('telemetry.csv', 'steady_telemetry.csv', 'summary.json', 'summary.csv')
DDL = """
CREATE TABLE IF NOT EXISTS runs (
 run_id TEXT PRIMARY KEY, gpu_uuid TEXT NOT NULL, gpu_name TEXT NOT NULL,
 source_commit TEXT NOT NULL, source_path TEXT NOT NULL, content_sha256 TEXT NOT NULL,
 manifest_json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS artifacts (
 run_id TEXT REFERENCES runs(run_id), path TEXT NOT NULL, sha256 TEXT NOT NULL,
 PRIMARY KEY(run_id,path));
CREATE TABLE IF NOT EXISTS trials (
 run_id TEXT REFERENCES runs(run_id), trial INTEGER NOT NULL CHECK(trial>0),
 scenario_id TEXT NOT NULL, workload TEXT NOT NULL, summary_json TEXT NOT NULL,
 PRIMARY KEY(run_id,trial,scenario_id));
CREATE TABLE IF NOT EXISTS samples (
 run_id TEXT NOT NULL, epoch_s REAL NOT NULL CHECK(epoch_s>0),
 trial INTEGER NOT NULL, scenario_id TEXT NOT NULL, power_watts REAL NOT NULL CHECK(power_watts>=0),
 steady INTEGER NOT NULL CHECK(steady IN (0,1)), raw_json TEXT NOT NULL,
 PRIMARY KEY(run_id,epoch_s),
 FOREIGN KEY(run_id,trial,scenario_id) REFERENCES trials(run_id,trial,scenario_id));
CREATE VIEW IF NOT EXISTS steady_trial_stats AS
WITH intervals AS (
 SELECT *, LAG(epoch_s) OVER w AS previous_epoch,
 LAG(power_watts) OVER w AS previous_watts
 FROM samples WHERE steady=1
 WINDOW w AS (PARTITION BY run_id,trial,scenario_id ORDER BY epoch_s)
)
SELECT run_id,trial,scenario_id,COUNT(*) AS sample_count,
 SUM((epoch_s-previous_epoch)*(power_watts+previous_watts)/2) AS energy_j,
 MAX(epoch_s)-MIN(epoch_s) AS span_s,
 SUM((epoch_s-previous_epoch)*(power_watts+previous_watts)/2)/(MAX(epoch_s)-MIN(epoch_s)) AS mean_power_w
FROM intervals GROUP BY run_id,trial,scenario_id;
CREATE VIEW IF NOT EXISTS device_workload_stats AS
SELECT r.gpu_uuid,r.gpu_name,t.workload,COUNT(*) AS trial_count,
 SUM(s.sample_count) AS sample_count,SUM(s.energy_j)/SUM(s.span_s) AS mean_power_w,
 SUM(s.energy_j) AS energy_j,SUM(s.span_s) AS span_s
FROM steady_trial_stats s JOIN trials t USING(run_id,trial,scenario_id)
JOIN runs r USING(run_id) GROUP BY r.gpu_uuid,r.gpu_name,t.workload;
"""


class WarehouseError(ValueError):
    """A capture violates the warehouse input contract."""


def _require(condition, message):
    if not condition:
        raise WarehouseError(message)


def _number(value, label, low=0, high=None):
    try:
        number = float(value)
    except (ValueError, TypeError) as exc:
        raise WarehouseError(f'{label}: expected a number') from exc
    _require(math.isfinite(number) and number >= low and (high is None or number <= high),
             f'{label}: non-finite or outside physical bounds')
    return number


def _csv(blob, label):
    reader = csv.DictReader(io.StringIO(blob.decode('utf-8')))
    _require(reader.fieldnames and len(set(reader.fieldnames)) == len(reader.fieldnames),
             f'{label}: missing or duplicate CSV columns')
    rows = list(reader)
    _require(rows and all(None not in row and None not in row.values() for row in rows),
             f'{label}: empty or malformed CSV')
    return rows


def _validate(directory):
    directory = Path(directory).resolve()
    manifest_blob = (directory / 'manifest.json').read_bytes()
    manifest = json.loads(manifest_blob)
    _require(manifest.get('schema_version') == 1, 'manifest: unsupported schema_version (expected 1)')
    _require(manifest.get('status') == 'complete', 'manifest: capture must be complete')
    run = manifest.get('run_id')
    device = manifest.get('device', {})
    _require(isinstance(run, str) and bool(run), 'manifest: missing run_id')
    _require(all(isinstance(device.get(k), str) and device[k] for k in ('gpu_uuid', 'gpu_name')),
             'manifest: missing device identity')
    _require(bool(re.fullmatch(r'[0-9a-f]{40}', str(manifest.get('source_commit', '')))),
             'manifest: source_commit must be a full Git SHA')
    hashes = manifest.get('artifacts_sha256', {})
    _require(isinstance(hashes, dict) and all(n in hashes for n in REQUIRED),
             'manifest: required artifact hashes are missing')
    blobs = {}
    for name, digest in hashes.items():
        path = (directory / name).resolve()
        _require(path.is_relative_to(directory) and path != directory, f'unsafe artifact path: {name}')
        _require(bool(re.fullmatch(r'[0-9a-f]{64}', str(digest))), f'{name}: invalid SHA-256')
        blob = path.read_bytes()
        _require(hashlib.sha256(blob).hexdigest() == digest, f'{name}: SHA-256 mismatch')
        if name in REQUIRED:
            blobs[name] = blob
    summaries = json.loads(blobs['summary.json'])
    _require(isinstance(summaries, list) and summaries, 'summary.json: expected nonempty list')
    trials = {}
    for summary in summaries:
        key = (summary['trial'], summary['scenario_id'])
        _require(type(key[0]) is int and key[0] > 0 and key[1] and key not in trials,
                 'summary.json: invalid or duplicate trial/scenario identity')
        _require(summary.get('verification_passed') is True, f'{key}: kernel verification failed')
        _require(summary['scenario_id'] == f"{summary['kernel']}-{summary['problem_size']}-{summary['block_size']}",
                 f'{key}: inconsistent scenario identity')
        for name, value in summary.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                _number(value, f'{key}.{name}')
        trials[key] = summary
    csv_summaries = _csv(blobs['summary.csv'], 'summary.csv')
    _require(len(csv_summaries) == len(trials), 'summary.csv: row count disagrees with JSON')
    seen = set()
    for row in csv_summaries:
        key = (int(row['trial']), row['scenario_id'])
        _require(key in trials and key not in seen, 'summary.csv: inconsistent or duplicate identity')
        seen.add(key)
        for name, value in row.items():
            expected = trials[key].get(name)
            _require(expected is not None and str(expected) == value, f'summary.csv: {key}.{name} disagrees with JSON')
    samples = {}
    for row in _csv(blobs['telemetry.csv'], 'telemetry.csv'):
        for name, expected in [('run_id', run), ('gpu_uuid', device['gpu_uuid']), ('gpu_name', device['gpu_name']),
                               ('backend', manifest.get('backend')), ('power_scope', manifest.get('power_scope'))]:
            _require(row.get(name) == expected, f'telemetry.csv: {name} identity mismatch')
        key = (int(row['trial']), row['scenario_id'])
        _require(key in trials and row['workload_type'] == trials[key]['kernel'], 'telemetry.csv: unknown workload/trial')
        _require(row['session_id'] == f'{run}_trial_{key[0]}', 'telemetry.csv: session identity mismatch')
        _require(row['phase'] in ('idle', 'workload'), 'telemetry.csv: unexpected phase')
        for name in ('epoch_s', 'monotonic_s', 'query_duration_s', 'power_watts'):
            _number(row[name], f'telemetry.csv.{name}')
        # Capture writes unavailable optional sensors as empty CSV cells. Preserve
        # that absence; a supplied sensor value must still be finite and physical.
        for name, low, high in (
            ('graphics_clock_mhz', 0, None), ('memory_clock_mhz', 0, None),
            ('gpu_utilization_pct', 0, 100), ('memory_utilization_pct', 0, 100),
            ('temperature_c', -273.15, None),
        ):
            value = row[name]  # Missing columns remain a schema error.
            if value is not None and value != '':
                _number(value, f'telemetry.csv.{name}', low=low, high=high)
        epoch = float(row['epoch_s'])
        _require(epoch > 0 and epoch not in samples, 'telemetry.csv: duplicate or invalid sample identity (epoch_s)')
        samples[epoch] = (row, False)
    for row in _csv(blobs['steady_telemetry.csv'], 'steady_telemetry.csv'):
        epoch = _number(row['epoch_s'], 'steady_telemetry.csv.epoch_s')
        _require(epoch in samples and not samples[epoch][1], 'steady_telemetry.csv: missing or duplicate sample identity')
        raw = samples[epoch][0]
        _require(row.get('phase') == 'steady' and raw['phase'] == 'workload'
                 and dict(row, phase=raw['phase']) == raw, 'steady_telemetry.csv: sample disagrees with raw telemetry')
        samples[epoch] = (raw, True)
    _require(manifest.get('telemetry_samples') == len(samples), 'manifest: telemetry_samples count mismatch')
    _require(manifest.get('steady_samples') == sum(s for _, s in samples.values()), 'manifest: steady_samples count mismatch')
    for key, summary in trials.items():
        selected = sorted((epoch, float(row['power_watts'])) for epoch, (row, steady) in samples.items()
                          if steady and (int(row['trial']), row['scenario_id']) == key)
        _require(len(selected) >= 2 and len(selected) == summary['sample_count'], f'{key}: steady sample_count mismatch')
        span = selected[-1][0] - selected[0][0]
        energy = sum((b[0]-a[0])*(b[1]+a[1])/2 for a, b in zip(selected, selected[1:], strict=False))
        for field, observed in [('mean_power_w', energy/span), ('estimated_energy_j', energy), ('sample_span_s', span),
                                ('min_power_w', min(p for _, p in selected)), ('max_power_w', max(p for _, p in selected))]:
            _require(math.isclose(observed, summary[field], rel_tol=1e-9, abs_tol=1e-7), f'{key}: {field} disagrees with steady samples')
    fingerprint = hashlib.sha256(manifest_blob).hexdigest()
    return directory, manifest, fingerprint, trials, samples


def ingest_capture(database, directory):
    """Validate every artifact, then atomically ingest; return True only for a new run."""
    directory, manifest, fingerprint, trials, samples = _validate(directory)
    with sqlite3.connect(database, timeout=30) as connection:
        connection.execute('PRAGMA foreign_keys=ON')
        version = connection.execute('PRAGMA user_version').fetchone()[0]
        _require(version in (0, SCHEMA_VERSION), f'warehouse: unsupported schema version {version}')
        connection.executescript(DDL)
        connection.execute('BEGIN IMMEDIATE')
        connection.execute(f'PRAGMA user_version={SCHEMA_VERSION}')
        run = manifest['run_id']
        previous = connection.execute('SELECT content_sha256 FROM runs WHERE run_id=?', (run,)).fetchone()
        if previous:
            _require(previous[0] == fingerprint, f'{run}: conflicting content for existing run_id')
            return False
        device = manifest['device']
        connection.execute('INSERT INTO runs VALUES (?,?,?,?,?,?,?)',
                           (run, device['gpu_uuid'], device['gpu_name'], manifest['source_commit'], str(directory),
                            fingerprint, json.dumps(manifest, sort_keys=True)))
        artifacts = dict(manifest['artifacts_sha256'], **{'manifest.json': fingerprint})
        connection.executemany('INSERT INTO artifacts VALUES (?,?,?)', [(run, n, h) for n, h in artifacts.items()])
        connection.executemany('INSERT INTO trials VALUES (?,?,?,?,?)',
                               [(run, trial, scenario, summary['kernel'], json.dumps(summary, sort_keys=True))
                                for (trial, scenario), summary in trials.items()])
        connection.executemany('INSERT INTO samples VALUES (?,?,?,?,?,?,?)',
                               [(run, epoch, int(row['trial']), row['scenario_id'], float(row['power_watts']),
                                 int(steady), json.dumps(row, sort_keys=True)) for epoch, (row, steady) in samples.items()])
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    ingest = sub.add_parser('ingest')
    ingest.add_argument('--database', required=True)
    ingest.add_argument('captures', nargs='+')
    query = sub.add_parser('query')
    query.add_argument('--database', required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'ingest':
            for capture in args.captures:
                print(json.dumps({'capture': capture, 'inserted': ingest_capture(args.database, capture)}))
        else:
            with sqlite3.connect(f'file:{Path(args.database).resolve()}?mode=ro', uri=True) as connection:
                connection.row_factory = sqlite3.Row
                print(json.dumps([dict(row) for row in connection.execute('SELECT * FROM device_workload_stats ORDER BY gpu_uuid,workload')], indent=2))
    except (WarehouseError, OSError, KeyError, TypeError, ValueError, sqlite3.Error) as exc:
        parser.exit(2, f'warehouse: {exc}\n')


if __name__ == '__main__':
    main()
