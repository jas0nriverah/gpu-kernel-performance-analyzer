import pandas as pd
import pytest

from gpu_power_pipeline.data import load_dataset
from gpu_power_pipeline.preprocessing import infer_feature_columns
from gpu_power_pipeline.train import get_train_test_indices


def test_collector_identity_survives_loader(tmp_path):
    path = tmp_path / 'trace.csv'
    pd.DataFrame(dict(power_watts=[150., 170.], timestamp=['2026-09-22T00:00:00Z', '2026-09-22T00:00:01Z'],
                      gpu_utilization_pct=[50, 70], memory_utilization_pct=[20, 30], temperature_c=[45, 46],
                      session_id=['trial_1', 'trial_2'], workload_type=['gemm_tiled', 'vector_add'],
                      gpu_uuid=['GPU-test'] * 2, gpu_name=['H100'] * 2,
                      scenario_id=['gemm-256', 'vec-1024'], gpu_id=[0, 0])).to_csv(path, index=False)
    df = load_dataset(source='local_nvidia_smi', data_path=str(path))
    assert df.session_id.tolist() == ['trial_1', 'trial_2']
    assert df.workload_type.tolist() == ['gemm_tiled', 'vector_add']
    assert df.gpu_uuid.tolist() == ['GPU-test', 'GPU-test']
    numeric, categorical = infer_feature_columns(df)
    assert not {'session_id', 'timestamp', 'scenario_id', 'gpu_id', 'gpu_uuid', 'source_file'} & set(numeric + categorical)


@pytest.mark.parametrize('strategy', ['grouped', 'time'])
def test_splits_never_silently_fall_back_to_random(strategy):
    with pytest.raises(ValueError):
        get_train_test_indices(pd.DataFrame({'power_watts': [100, 200, 300]}), strategy, .2, 42)
