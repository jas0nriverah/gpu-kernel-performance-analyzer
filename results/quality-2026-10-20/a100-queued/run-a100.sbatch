#!/usr/bin/env bash
#SBATCH --job-name=gpu-quality-a100
#SBATCH --account=ece
#SBATCH --qos=coe-ice
#SBATCH --partition=ice-gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --export=NONE
#SBATCH --output=/home/hice1/jriverah3/gpu-kernel-performance-analyzer-1/outputs/quality-2026-10-20/a100/slurm-%j.log
set -euo pipefail
cd /home/hice1/jriverah3/gpu-kernel-performance-analyzer-1
A100_ROOT="$PWD/outputs/quality-2026-10-20/a100"
A100_PYTHON="$PWD/.venv/bin/python"
export A100_ROOT
trap 'code=$?; A100_EXIT_CODE=$code "$A100_PYTHON" - <<'"'"'PY'"'"'
import json,os,socket
from datetime import datetime,timezone
from pathlib import Path
p=Path(os.environ["A100_ROOT"])/"execution-receipt.json"
p.write_text(json.dumps({"job_id":os.environ.get("SLURM_JOB_ID"),"host":socket.gethostname(),"finished_utc":datetime.now(timezone.utc).isoformat(),"exit_code":int(os.environ["A100_EXIT_CODE"]),"gpu_uuid":os.environ.get("GPU_UUID"),"account":os.environ.get("SLURM_JOB_ACCOUNT"),"partition":os.environ.get("SLURM_JOB_PARTITION"),"qos":os.environ.get("SLURM_JOB_QOS"),"cpus":os.environ.get("SLURM_CPUS_PER_TASK"),"memory_mb":os.environ.get("SLURM_MEM_PER_NODE")},indent=2)+"\n")
PY
exit "$code"' EXIT
source /etc/profile.d/z00_lmod_pace.sh
module load gcc/12.3.0
module load cuda/12.6.1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
unset GPU_UUID CUDA_ARCHITECTURES BUILD_DIR OUT_DIR MODE
# SLURM_JOB_GPUS is the scheduler's physical allocation, unlike inherited H200 state.
[[ "${SLURM_JOB_GPUS:-}" != *,* && -n "${SLURM_JOB_GPUS:-}" ]]
export GPU_UUID=$(nvidia-smi -i "$SLURM_JOB_GPUS" --query-gpu=uuid --format=csv,noheader)
export CUDA_VISIBLE_DEVICES="$GPU_UUID"
nvidia-smi -i "$GPU_UUID" --query-gpu=uuid,name,driver_version,memory.total --format=csv > "$A100_ROOT/allocated-gpu.csv"
grep -q A100 "$A100_ROOT/allocated-gpu.csv"
uname -a > "$A100_ROOT/native-host.txt"
module list 2> "$A100_ROOT/modules.txt"
"$A100_PYTHON" -m pip freeze > "$A100_ROOT/environment.txt"
scontrol show job "$SLURM_JOB_ID" > "$A100_ROOT/slurm-job.txt"
"$A100_PYTHON" - <<'PY'
import hashlib,json,tarfile,subprocess,os
from pathlib import Path
from datetime import datetime,timezone
root=Path.cwd(); out=Path(os.environ['A100_ROOT'])
files=[]
for directory in ['src','scripts','benchmarks','configs']:
    files += [p for p in (root/directory).rglob('*') if p.is_file() and not p.is_symlink() and '__pycache__' not in p.parts and p.suffix not in {'.pyc','.so'}]
files += [root/'pyproject.toml']
manifest={'captured_utc':datetime.now(timezone.utc).isoformat(),'git_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'files':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}}
(out/'source-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
with tarfile.open(out/'source-overlay.tar.gz','w:gz') as tar:
    for p in sorted(files): tar.add(p,arcname=str(p.relative_to(root)))
PY
PYTHON="$A100_PYTHON" MODE=smoke OUT_DIR="$A100_ROOT/smoke" bash scripts/cross_validate_gpu.sh > "$A100_ROOT/smoke.log" 2>&1
PYTHON="$A100_PYTHON" MODE=full OUT_DIR="$A100_ROOT/timing" bash scripts/cross_validate_gpu.sh > "$A100_ROOT/timing.log" 2>&1
"$A100_PYTHON" -m gpu_kernel_analyzer power capture --binary "$A100_ROOT/timing/build/gpu_benchmark" --scenarios configs/power_h100.yaml --device "$GPU_UUID" --duration-seconds 15 --trials 3 --outdir "$A100_ROOT/power" > "$A100_ROOT/power.log" 2>&1
"$A100_PYTHON" scripts/summarize_power.py --run-dir "$A100_ROOT/power" > "$A100_ROOT/power-summary.log" 2>&1
"$A100_PYTHON" - <<'PY'
import hashlib,json,os
from pathlib import Path
root=Path(os.environ['A100_ROOT'])/'power'; manifest=json.loads((root/'manifest.json').read_text())
assert manifest['status']=='complete'
for name,digest in manifest['artifacts_sha256'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
(root.parent/'power-hash-verification.json').write_text(json.dumps({'status':'ok','artifacts_verified':len(manifest['artifacts_sha256'])},indent=2)+'\n')
PY
