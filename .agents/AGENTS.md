# Project Rules — Aegis Trading System (Institutional Trend-Following Platform)

## Mandatory Architectural Context & Specifications
Whenever answering user queries, designing features, or modifying code inside this repository, you **MUST** strictly adhere to and consult the two foundational engineering documents:

1. **Master Blueprint v11.8 (`docs/architecture.md`)**:
   - Contains the exhaustive, 100% unabridged mathematical formulations, AFML references, order flow microstructure formulas, Kalman P-matrix closure, and production hardening specs (Module A through Module K).
   - Enforces forward-only Causal HMM alpha pass (`hmm_causal.py`), windowed FFD default (`select_ffd_production_engine`), empirical Kelly sizing log-growth (`kelly_empirical.py`), and absolute bar-index offset resolution (`resolve_absolute_exit_idx`).

2. **Data Contracts v11.9 (`CONSTRACT.md`)**:
   - Governs the strict schemas and behavioral rules between Track A (Data & Signal: `SIGNAL_BAR_SCHEMA`) and Track B (Labeling & Decision: `TRADE_RECORD_SCHEMA`).
   - Enforces that `timestamp_ms` in `SIGNAL_BAR_SCHEMA` is Point-in-Time `knowledge_time`, and that `exit_idx_absolute` (`= entry_idx + 1 + exit_idx_relative`) is the sole index used for `fill_price_exit` and `funding_accrued` lookups.

3. **Mandatory Living Engineering Explanation Report (`docs/engineering_audit_and_explanation_report.md`)**:
   - Whenever working on, explaining, or creating a new module/file, you **MUST** both:
     1. **Update `PHẦN I: TỔNG QUAN HỆ THỐNG & TRIẾT LÝ THIẾT KẾ`** to reflect the newly integrated components, update the project scope list, and keep the Master System Architecture Mermaid diagram up-to-date.
     2. **Append a comprehensive, institutional-grade, line-by-line breakdown (`PHẦN IX...`)** right after the existing sections. Include clear explanations of *what* each function/class does, *why* it is designed that way, and how it aligns with the Master Blueprint.
   - **MANDATORY MERMAID FLOW DIAGRAMS:** For every file or task breakdown, you **MUST** include a visual Mermaid flow diagram (````mermaid ... ````) illustrating the exact data flow, control logic, execution branches, or dual-layer verification pipeline of that specific component.

## Code Scope & Firewall Rules

- **`src/aegis/`** is the production boundary. Only code that has passed full CPCV 15-Fold, DSR >= 0.95, and PBO <= 0.40 validation belongs here.
- **`research/`** is for sandbox exploration (TFT, XGBoost, RL, Markov scanners). Nothing inside `src/aegis/` may ever import from `research/`.

## Mandatory Standard Operating Procedure (SOP) for All Tasks

For **EVERY** task assigned, the following 3-Phase SOP must be strictly adhered to:

### PHASE 1: TASK INITIATION (Preparation & Contracts)
1. **Locate the File:** Determine exactly where the code will live (e.g., `src/aegis/core/` or `src/aegis/features/`).
2. **Contract Alignment BEFORE Coding:** Review `CONSTRACT.md` (or relevant contract specs) to ensure the function signature and data types match exactly what the other Track expects. Do not code blindly and check later.
3. **Dependency Check:** Check the Backlog for dependencies. Ensure prior functions are stable and imported correctly. Review existing tests of shared modules (like `ExperimentTracker` or `PurgedKFold`) to understand who depends on them before modifying.
4. **Establish Mathematical/Logical Goals:** Clarify the core formula or algorithm (e.g., MAD, EM loops, FFD weights) before typing code.
5. **Determinism (Seed):** Explicitly fix random seeds for any stochastic operations (EM initialization, bootstrap) to ensure reproducible experiment hashes.

### PHASE 2: EXECUTION (Coding & Optimization)
1. **Clean Causal Logic:** Code in `src/aegis/` must be strictly causal (no look-ahead bias).
2. **Symmetric Long/Short:** Any logic that branches by `side` MUST explicitly handle and test both Long and Short scenarios symmetrically.
3. **Performance Optimization:** 
   - Arrays: Maximize Numpy/Pandas vectorization. NO nested for-loops on DataFrames.
   - Tick-level loops: Must use Numba `@njit` and compile successfully without type errors. Do not pre-allocate max size naively; estimate empirical ratios.
4. **Sanitization & Bounds:**
   - Matrices: Use Cholesky decomposition (`np.linalg.cholesky`) to guarantee Positive Definite (PD), not just diagonal checks.
   - Scalars: Cap extreme values (e.g., huge time deltas over weekends) explicitly, don't just floor against divide-by-zero.

### PHASE 3: VERIFICATION (Quant & Audit Logging)
1. **Quantitative Testing:**
# Project Rules — Aegis Trading System (Institutional Trend-Following Platform)

## Mandatory Architectural Context & Specifications
Whenever answering user queries, designing features, or modifying code inside this repository, you **MUST** strictly adhere to and consult the two foundational engineering documents:

1. **Master Blueprint v11.8 (`docs/architecture.md`)**:
   - Contains the exhaustive, 100% unabridged mathematical formulations, AFML references, order flow microstructure formulas, Kalman P-matrix closure, and production hardening specs (Module A through Module K).
   - Enforces forward-only Causal HMM alpha pass (`hmm_causal.py`), windowed FFD default (`select_ffd_production_engine`), empirical Kelly sizing log-growth (`kelly_empirical.py`), and absolute bar-index offset resolution (`resolve_absolute_exit_idx`).

2. **Data Contracts v11.9 (`CONSTRACT.md`)**:
   - Governs the strict schemas and behavioral rules between Track A (Data & Signal: `SIGNAL_BAR_SCHEMA`) and Track B (Labeling & Decision: `TRADE_RECORD_SCHEMA`).
   - Enforces that `timestamp_ms` in `SIGNAL_BAR_SCHEMA` is Point-in-Time `knowledge_time`, and that `exit_idx_absolute` (`= entry_idx + 1 + exit_idx_relative`) is the sole index used for `fill_price_exit` and `funding_accrued` lookups.

3. **Mandatory Living Engineering Explanation Report (`docs/engineering_audit_and_explanation_report.md`)**:
   - Whenever working on, explaining, or creating a new module/file, you **MUST** both:
     1. **Update `PHẦN I: TỔNG QUAN HỆ THỐNG & TRIẾT LÝ THIẾT KẾ`** to reflect the newly integrated components, update the project scope list, and keep the Master System Architecture Mermaid diagram up-to-date.
     2. **Append a comprehensive, institutional-grade, line-by-line breakdown (`PHẦN IX...`)** right after the existing sections. Include clear explanations of *what* each function/class does, *why* it is designed that way, and how it aligns with the Master Blueprint.
   - **MANDATORY MERMAID FLOW DIAGRAMS:** For every file or task breakdown, you **MUST** include a visual Mermaid flow diagram (````mermaid ... ````) illustrating the exact data flow, control logic, execution branches, or dual-layer verification pipeline of that specific component.

## Code Scope & Firewall Rules

- **`src/aegis/`** is the production boundary. Only code that has passed full CPCV 15-Fold, DSR >= 0.95, and PBO <= 0.40 validation belongs here.
- **`research/`** is for sandbox exploration (TFT, XGBoost, RL, Markov scanners). Nothing inside `src/aegis/` may ever import from `research/`.

## Mandatory Standard Operating Procedure (SOP) for All Tasks

For **EVERY** task assigned, the following 3-Phase SOP must be strictly adhered to:

### PHASE 1: TASK INITIATION (Preparation & Contracts)
1. **Locate the File:** Determine exactly where the code will live (e.g., `src/aegis/core/` or `src/aegis/features/`).
2. **Contract Alignment BEFORE Coding:** Review `CONSTRACT.md` (or relevant contract specs) to ensure the function signature and data types match exactly what the other Track expects. Do not code blindly and check later.
3. **Dependency Check:** Check the Backlog for dependencies. Ensure prior functions are stable and imported correctly. Review existing tests of shared modules (like `ExperimentTracker` or `PurgedKFold`) to understand who depends on them before modifying.
4. **Establish Mathematical/Logical Goals:** Clarify the core formula or algorithm (e.g., MAD, EM loops, FFD weights) before typing code.
5. **Determinism (Seed):** Explicitly fix random seeds for any stochastic operations (EM initialization, bootstrap) to ensure reproducible experiment hashes.

### PHASE 2: EXECUTION (Coding & Optimization)
1. **Clean Causal Logic:** Code in `src/aegis/` must be strictly causal (no look-ahead bias).
2. **Symmetric Long/Short:** Any logic that branches by `side` MUST explicitly handle and test both Long and Short scenarios symmetrically.
3. **Performance Optimization:** 
   - Arrays: Maximize Numpy/Pandas vectorization. NO nested for-loops on DataFrames.
   - Tick-level loops: Must use Numba `@njit` and compile successfully without type errors. Do not pre-allocate max size naively; estimate empirical ratios.
4. **Sanitization & Bounds:**
   - Matrices: Use Cholesky decomposition (`np.linalg.cholesky`) to guarantee Positive Definite (PD), not just diagonal checks.
   - Scalars: Cap extreme values (e.g., huge time deltas over weekends) explicitly, don't just floor against divide-by-zero.

### PHASE 3: VERIFICATION (Quant & Audit Logging)
1. **Quantitative Testing:**
   - Write unit tests in `tests/`.
   - Test both Long/Short branches explicitly.
   - Use Synthetic Data with known properties to prove mathematical correctness.
   - Test Edge Cases: Empty arrays, NaN, inf, zero variance. The function must not crash.
   - Regression/Parity Tests: For refactoring tasks, prove the new output perfectly matches the old O(N^2) output.
2. **Causal & Leakage Check:** Ensure $t$ never sees $t+1$. Configuration parameters must be Point-in-Time (PIT).
3. **Performance Verification:** Confirm Numba C-level compilation. Benchmark execution time and peak memory before/after on real-scale data.
4. **Bidirectional Contract & Type Alignment:** Ensure outputs to AND inputs from Track B match exactly, including strict dtypes (float32 vs float64) to prevent drift.
5. **Experiment Tracking:** Call `ExperimentTracker`. Include `param_hash`, git commit hash, and core library versions (numpy, polars, etc.) in the log.
6. **Engineering Audit Update:** Append to `docs/engineering_audit_and_explanation_track_a.md` using the exact format. If out-of-scope issues were found and fixed, put them in a dedicated "Phát hiện phát sinh" subsection.
7. **Peer Review:** Critical or shared logic must be peer-reviewed before being considered closed.
8. **Github Sync & Track B Handover:** After every 3 completed tasks, you must commit and push all changes to the origin repository. Alongside the push, output a summary specifically targeted at Track B members, highlighting any interface changes, shared enums, or contract updates they need to pay attention to in order to prevent integration failures.

**FINAL REQUIREMENT FOR EVERY TASK:** Always output exactly which files were modified, and the exact lines added/changed so the user can easily find and review them.
