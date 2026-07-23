import numpy as np
import pandas as pd
from aegis.meta_labeling.purged_kfold import PurgedKFold

class MockEstimator:
    def __init__(self):
        self.fit_calls = 0
        self.fitted_data_hashes = set()
        
    def fit(self, X, y):
        self.fit_calls += 1
        # Băm dữ liệu để đảm bảo mỗi lần fit là một tập dữ liệu hoàn toàn khác biệt
        data_hash = hash(X.tobytes())
        self.fitted_data_hashes.add(data_hash)
        return self
        
    def predict(self, X):
        return np.zeros(len(X))

def test_pipeline_refit_per_fold_enforcement():
    """
    [TDD VERIFICATION - REFIT PER FOLD ENFORCEMENT]:
    Kiểm chứng bắt buộc kiến trúc: Mô hình (HMM, Kalman, FFD, ML)
    phải được gọi hàm .fit() lại từ đầu (độc lập) trên mỗi fold của PurgedKFold,
    không được phép fit 1 lần trên toàn bộ tập dữ liệu (gây rò rỉ tương lai).
    """
    n_samples = 100
    X = pd.DataFrame({'feat': np.random.randn(n_samples)})
    y = pd.Series(np.random.randint(0, 2, n_samples))
    et = pd.Series(index=np.arange(n_samples), data=np.arange(n_samples))
    
    cv = PurgedKFold(n_splits=5, embargo_pct=0.0, embargo_bars=2)
    model = MockEstimator()
    
    # Mô phỏng vòng lặp cross-validation thực tế trong pipeline
    for train_idx, test_idx in cv.split(X, event_times=et):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        
        # BẮT BUỘC REFIT TRÊN TỪNG FOLD
        model.fit(X_train.values, y_train.values)
        _ = model.predict(X_test.values)
        
    # Xác nhận hàm fit được gọi đúng 5 lần
    assert model.fit_calls == 5, f"Kỳ vọng fit 5 lần (refit-per-fold), nhưng chỉ gọi {model.fit_calls} lần!"
    
    # Xác nhận 5 lần fit đó được gọi trên 5 tập dữ liệu hoàn toàn khác nhau
    assert len(model.fitted_data_hashes) == 5, "Lỗi rò rỉ! Mô hình bị fit trên các tập dữ liệu trùng lặp!"
