"""The lightweight analyzer installation does not require ML/API dependencies."""
import importlib.util

collect_ignore = []
if any(importlib.util.find_spec(name) is None for name in ('pandas', 'sklearn', 'joblib')):
    collect_ignore.append('power')
