# Bảng Đối Chiếu Module A–K <-> Thư Mục Thực Tế

| Module (Master Blueprint v11.8) | Thư mục trong cây aegis-trading-system |
|---|---|
| **Giai đoạn 0 (Core, PIT, Parity)** | `src/aegis/core/` |
| **Module A (Ingestion, Outlier, Bars, Gap)** | `src/aegis/data/` |
| **Module A.3 (Fractional Differentiation)** | `src/aegis/features/fractional_diff.py` |
| **Module B (Kalman, HMM Causal, GHE)** | `src/aegis/features/kalman/`, `src/aegis/features/regime/` |
| **Module C (CUSUM Events, Triple-Barrier, Trailing)** | `src/aegis/labeling/` |
| **Module D (Consensus Feature Selection)** | `src/aegis/meta_labeling/feature_selection/` |
| **Module E (PurgedKFold, Calibration, Empirical Kelly)** | `src/aegis/meta_labeling/` |
| **Module F (CPCV 15-Fold, DSR, PBO, Absolute Index)** | `src/aegis/validation/` + `src/aegis/pipelines/cpcv_pipeline.py` |
| **Module G (Execution Simulation, Queue Sim)** | `src/aegis/execution/` |
| **Module H (Parity Test Harness & CI)** | `tests/integration/test_full_chain_golden_path.py` + `.github/workflows/parity_check.yml` |
| **Module I (Shadow Mode Readiness Gate Kép)** | `src/aegis/shadow/` |
| **Module J (Portfolio Risk, Circuit Breakers, Ledoit-Wolf)** | `src/aegis/risk/` |
| **Module K (Data Governance, Funding Accrual, L2 Depth)** | `src/aegis/data/universe.py` (K.3) + `src/aegis/governance/` |
