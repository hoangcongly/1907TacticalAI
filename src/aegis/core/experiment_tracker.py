"""ExperimentTracker Singleton — Ghi nhận DSR trials và SHA-256 param hashes."""

import json
import hashlib
import os
import subprocess
import importlib.metadata
from datetime import datetime, timezone
import threading
from typing import Dict, Any, Optional

from aegis.core.trial_classes import TrialClass


class ExperimentTracker:
    """
    Singleton class để quản lý việc ghi nhận các lần chạy thử nghiệm (trials).
    Đảm bảo tính nhất quán (thread-safe) khi ghi log JSONL từ nhiều luồng.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, log_dir: str = "logs/experiments"):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ExperimentTracker, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, log_dir: str = "logs/experiments"):
        if getattr(self, '_initialized', False):
            return
            
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Tạo file log cho phiên chạy hiện tại (theo ngày)
        date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
        self.log_file = os.path.join(self.log_dir, f"trials_{date_str}.jsonl")
        
        self._write_lock = threading.Lock()
        self._initialized = True

    def hash_params(self, params: Dict[str, Any]) -> str:
        """
        Mã hóa cấu hình thử nghiệm bằng SHA-256. 
        Dùng sort_keys=True để đảm bảo tính nhất quán của chuỗi băm.
        """
        # Loại bỏ các tham số không ảnh hưởng đến logic (nếu có, tuỳ dự án)
        serialized = json.dumps(params, sort_keys=True)
        return hashlib.sha256(serialized.encode('utf-8')).hexdigest()

    def _get_git_commit(self) -> str:
        try:
            commit_hash = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'], 
                stderr=subprocess.STDOUT
            ).decode('utf-8').strip()
            return commit_hash
        except Exception:
            return "unknown"

    def _get_env_versions(self) -> Dict[str, str]:
        packages = ["numpy", "polars", "numba", "scikit-learn"]
        versions = {}
        for pkg in packages:
            try:
                versions[pkg] = importlib.metadata.version(pkg)
            except importlib.metadata.PackageNotFoundError:
                versions[pkg] = "unknown"
        return versions

    def log_trial(self, trial_class: TrialClass, params: Dict[str, Any], metrics: Dict[str, Any]) -> str:
        """
        Ghi lại một lần chạy thử nghiệm xuống file JSONL.
        Trả về param_hash.
        """
        param_hash = self.hash_params(params)
        
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "trial_class": trial_class.value,
            "param_hash": param_hash,
            "git_commit": self._get_git_commit(),
            "env_versions": self._get_env_versions(),
            "params": params,
            "metrics": metrics
        }
        
        with self._write_lock:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, sort_keys=True) + "\n")
                
        return param_hash
