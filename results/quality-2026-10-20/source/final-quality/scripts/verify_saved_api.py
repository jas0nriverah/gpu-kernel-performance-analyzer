"""Verify a saved evaluation bundle through the in-process FastAPI interface."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from gpu_power_pipeline.api import create_app
from gpu_power_pipeline.data import load_dataset
from gpu_power_pipeline.inference import PowerModel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--transfer-dir', '--evaluation-dir', dest='transfer_dir', type=Path, required=True)
    parser.add_argument('--telemetry', type=Path, required=True)
    parser.add_argument('--out', type=Path, help='New evidence JSON; existing files are never overwritten.')
    parser.add_argument('--expect-unseen-gpu', action='store_true',
                        help='Verify rejection of out-of-domain GPU inputs instead of serving them.')
    args = parser.parse_args()
    output = args.out or args.transfer_dir / 'api-verification.json'
    if output.exists():
        parser.error(f'Choose a new --out; evidence already exists: {output}')
    evidence = json.loads((args.transfer_dir / 'evaluation.json').read_text())
    bundle = Path(evidence['saved_model'])
    assert hashlib.sha256((bundle / "model.joblib").read_bytes()).hexdigest() == evidence["saved_model_sha256"]
    model = PowerModel.from_dir(bundle)
    frame = load_dataset("local_nvidia_smi", data_path=str(args.telemetry))
    required_fields = list(model.metadata.feature_order)
    if model.metadata.reference_profile.get('categorical', {}).get('gpu_name', {}).get('categories'):
        required_fields = list(dict.fromkeys(required_fields + ['gpu_name']))
    records = json.loads(frame.reindex(columns=required_fields).to_json(orient='records'))
    direct = model.predict(records)
    app = create_app(str(bundle.parent.parent), model.metadata.model_name,
                     model.metadata.version)
    served_batches = []
    with TestClient(app) as client:
        health = client.get('/health')
        readiness = client.get('/ready')
        metadata = client.get('/model')
        for start in range(0, len(records), 128):
            batch = records[start:start + 128]
            response = client.post('/predict', json={'records': batch})
            if args.expect_unseen_gpu:
                assert response.status_code == 422, response.text
                assert 'gpu_name' in response.json()['detail']
                assert 'not a GPU represented' in response.json()['detail']
            else:
                assert response.status_code == 200, response.text
                body = response.json()
                assert body['model_name'] == model.metadata.model_name
                assert body['model_version'] == model.metadata.version
                assert [item['index'] for item in body['predictions']] == list(range(len(batch)))
                served_batches.extend(item['predicted_power_watts'] for item in body['predictions'])
        counters = client.get('/metrics').json()
    assert health.status_code == readiness.status_code == metadata.status_code == 200
    assert health.json()['model_loaded'] is True
    served = np.asarray(served_batches)
    if not args.expect_unseen_gpu:
        np.testing.assert_allclose(served, direct, rtol=0, atol=1e-10)
    result = {
        'scope': 'In-process TestClient; no network or latency benchmark.',
        'saved_model': str(bundle), 'records': len(records), 'health': health.json(),
        'model_http_status': metadata.status_code, 'predict_http_status': response.status_code,
        'expected_unseen_gpu_rejection': args.expect_unseen_gpu,
        'max_absolute_difference_watts': None if args.expect_unseen_gpu else float(np.max(np.abs(served - direct))),
        'all_predictions_finite': None if args.expect_unseen_gpu else bool(np.isfinite(served).all()),
        'served_records': len(served), 'metrics': counters,
        'telemetry_sha256': hashlib.sha256(args.telemetry.read_bytes()).hexdigest(),
    }
    assert args.expect_unseen_gpu or result['all_predictions_finite']
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
