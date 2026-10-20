#!/usr/bin/env bash
#SBATCH --job-name=gpu-quality-a100-eval
#SBATCH --account=ece
#SBATCH --qos=coe-ice
#SBATCH --partition=ice-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --export=NONE
#SBATCH --output=/home/hice1/jriverah3/gpu-kernel-performance-analyzer-1/outputs/quality-2026-10-20/a100/followup-slurm-%j.log
set -euo pipefail
cd /home/hice1/jriverah3/gpu-kernel-performance-analyzer-1
A100_ROOT="$PWD/outputs/quality-2026-10-20/a100"
A100_PYTHON="$PWD/.venv/bin/python"
export A100_ROOT
trap 'code=$?; A100_EXIT_CODE=$code "$A100_PYTHON" - <<'"'"'PY'"'"'
import json,os,socket
from datetime import datetime,timezone
from pathlib import Path
p=Path(os.environ["A100_ROOT"])/("followup-receipt-"+os.environ["SLURM_JOB_ID"]+".json")
p.write_text(json.dumps({"job_id":os.environ.get("SLURM_JOB_ID"),"gpu_dependency":"afterok:6426654","host":socket.gethostname(),"finished_utc":datetime.now(timezone.utc).isoformat(),"exit_code":int(os.environ["A100_EXIT_CODE"]),"partition":os.environ.get("SLURM_JOB_PARTITION"),"qos":os.environ.get("SLURM_JOB_QOS"),"cpus":os.environ.get("SLURM_CPUS_PER_TASK"),"memory_mb":os.environ.get("SLURM_MEM_PER_NODE")},indent=2)+"\n")
PY
exit "$code"' EXIT
source /etc/profile.d/z00_lmod_pace.sh
module load gcc/12.3.0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
unset GPU_UUID CUDA_VISIBLE_DEVICES
for name in frozen-evaluation known-device-challenge transfer; do
  [[ ! -e "$A100_ROOT/$name" ]] || { echo "Refusing existing output $name" >&2; exit 1; }
done
"$A100_PYTHON" - <<'PY'
import hashlib,json,os
from pathlib import Path
root=Path(os.environ['A100_ROOT'])
manifest=json.loads((root/'cpu-followup-input-hashes.json').read_text())
for name,digest in manifest['files'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
capture=json.loads((root/'power/manifest.json').read_text());assert capture['status']=='complete'
for name,digest in capture['artifacts_sha256'].items():assert hashlib.sha256((root/'power'/name).read_bytes()).hexdigest()==digest,name
PY
for suite in baseline trial-1 trial-2 trial-3 edges; do
  "$A100_PYTHON" -m gpu_kernel_analyzer artifacts validate --run-dir "$A100_ROOT/timing/$suite"
done
"$A100_PYTHON" -m gpu_power_pipeline.final_evaluation --development-dir outputs/quality-2026-10-20/model-development --test "$A100_ROOT/power/steady_telemetry.csv" --outdir "$A100_ROOT/frozen-evaluation" > "$A100_ROOT/frozen-evaluation.log" 2>&1
"$A100_PYTHON" -m gpu_power_pipeline.final_evaluation --development-dir outputs/quality-2026-10-20/known-device-development --test "$A100_ROOT/power/steady_telemetry.csv" --outdir "$A100_ROOT/known-device-challenge" > "$A100_ROOT/known-device-challenge.log" 2>&1
"$A100_PYTHON" -m gpu_power_pipeline.transfer --train results/h100-power-2026-09-22/steady_telemetry.csv --test "$A100_ROOT/power/steady_telemetry.csv" --outdir "$A100_ROOT/transfer" --model linear_regression > "$A100_ROOT/transfer.log" 2>&1
A100_DATABASE="$A100_ROOT/warehouse-${SLURM_JOB_ID}.sqlite"
[[ ! -e "$A100_DATABASE" ]]
export A100_DATABASE
"$A100_PYTHON" - <<'PY'
import json,os,sqlite3
from pathlib import Path
from gpu_power_pipeline.warehouse import ingest_capture
root=Path(os.environ['A100_ROOT']); database=Path(os.environ['A100_DATABASE'])
first=ingest_capture(database,root/'power');second=ingest_capture(database,root/'power')
assert first is True and second is False
with sqlite3.connect(database) as c:
    c.row_factory=sqlite3.Row
    result={'database':str(database),'first_inserted':first,'repeat_inserted':second,'integrity_check':c.execute('PRAGMA integrity_check').fetchone()[0],'foreign_key_check':[list(r) for r in c.execute('PRAGMA foreign_key_check')], 'counts':{table:c.execute('SELECT COUNT(*) FROM '+table).fetchone()[0] for table in ['runs','artifacts','trials','samples']},'steady_samples':c.execute('SELECT COUNT(*) FROM samples WHERE steady=1').fetchone()[0]}
    summary=[dict(row) for row in c.execute('SELECT * FROM device_workload_stats ORDER BY gpu_uuid,workload')]
assert result['integrity_check']=='ok' and not result['foreign_key_check']
(root/'warehouse-verification.json').write_text(json.dumps(result,indent=2)+'\n')
(root/'warehouse-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
PY
