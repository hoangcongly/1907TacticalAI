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
