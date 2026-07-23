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


def _json_default(obj: Any) -> Any:
    """Helper chuyển đổi an toàn các đối tượng numpy/pandas sang primitive python cho json.dumps."""
    import numpy as np
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    return str(obj)


class ExperimentTracker:
    """
    Singleton class để quản lý việc ghi nhận các lần chạy thử nghiệm (trials).
    Đảm bảo tính nhất quán (thread-safe) khi ghi log JSONL từ nhiều luồng.
    """
    _instance = None
    _lock = threading.Lock()
    _initialized: bool = False
    log_dir: Optional[str] = None

    def __new__(cls, log_dir: str = "logs/experiments"):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ExperimentTracker, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, log_dir: str = "logs/experiments"):
        if getattr(self, '_initialized', False):
            if self.log_dir != log_dir:
                import warnings
                warnings.warn(f"ExperimentTracker là Singleton. Đã khởi tạo với log_dir='{self.log_dir}'. Bỏ qua tham số log_dir='{log_dir}'.")
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
        serialized = json.dumps(params, sort_keys=True, default=_json_default)
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

    def _get_canonical_constants(self) -> Dict[str, Any]:
        """
        Đọc và ghi nhận các hằng số chiến lược & microstructure guards từ sổ cân bằng hằng số
        (`config/aegis_canonical_parameters.yaml`) vào nhật ký thử nghiệm (ExperimentTracker)
        nhằm tuân thủ tuyệt đối kỷ luật quản lý hằng số theo Request 9.
        """
        try:
            import yaml  # type: ignore
            # Tìm đường dẫn từ gốc dự án
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
            reg_path = os.path.join(project_root, "config", "aegis_canonical_parameters.yaml")
            if os.path.exists(reg_path):
                with open(reg_path, "r", encoding="utf-8") as f:
                    reg = yaml.safe_load(f)
                return {
                    "microstructure_guards": {
                        k: v.get("value") for k, v in reg.get("microstructure_guards", {}).items() if isinstance(v, dict)
                    },
                    "strategic_parameters": {
                        k: v.get("value") for k, v in reg.get("strategic_parameters", {}).items() if isinstance(v, dict)
                    },
                    "architectural_constants": {
                        k: v.get("value") for k, v in reg.get("architectural_constants", {}).items() if isinstance(v, dict)
                    },
                }
        except Exception:
            pass
        return {}

    def log_trial(self, trial_class: TrialClass, params: Dict[str, Any], metrics: Dict[str, Any]) -> str:
        """
        Ghi lại một lần chạy thử nghiệm xuống file JSONL.
        Trả về param_hash.
        """
        param_hash = self.hash_params(params)
        canonical_constants = self._get_canonical_constants()
        
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "trial_class": trial_class.value,
            "param_hash": param_hash,
            "git_commit": self._get_git_commit(),
            "env_versions": self._get_env_versions(),
            "canonical_constants": canonical_constants,
            "params": params,
            "metrics": metrics
        }
        
        with self._write_lock:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, sort_keys=True, default=_json_default) + "\n")
                
        return param_hash

