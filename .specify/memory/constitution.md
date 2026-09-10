# Aegis Institutional Trading System Constitution

## Core Principles

### I. Zero-Leakage & Temporal Invariance (NON-NEGOTIABLE)
All signal generation, labeling, and feature engineering must adhere strictly to point-in-time causality. Accessing future data ($i+1$, $i+k$, or shift(-1)) at time $i$ is a fatal violation.
- **Pre-Slice Protocol**: Arrays MUST be physically sliced (`future_bars = full_bars[entry+1 : effective_end]`) BEFORE passing to simulation logic.
- **Absolute Indexing**: Any execution or data-contract module MUST resolve relative offsets to absolute indices (`exit_idx_absolute = entry_idx + 1 + exit_idx_relative`).
- **Temporal Invariants in CV**: Cross-validation MUST enforce PurgedKFold/CPCV with absolute embargo bars (`embargo_bars = 24`). Temporal purging invariants ($t_{1,i} \le t_{0,\text{test}}$ for prior train and $t_{0,j} \ge t_{1,\text{test}} + \text{embargo\_step}$ for post train) are mandatory.

### II. Immutability & Strict Data Contracts
Financial data passing across pipeline stages must be immutable to eliminate silent state mutation bugs.
- **Immutable Structures**: All trade records and financial payloads MUST use `@dataclass(frozen=True)` or Pydantic models with `frozen=True`. Mutable dict or TypedDict are strictly forbidden for inter-module communication. State updates MUST return new instances via `dataclasses.replace(old_record, **updates)`.
- **Schema Enforcement**: DataFrames passing boundary gates MUST be validated against strict schemas (`SignalBarSchema`, `TradeRecordSchema`) with `strict=True` to reject unexpected columns or data type mismatch.

### III. Canonical Parameter Single Source of Truth
Parameter drift between Research, Backtest, and Live Execution is strictly prohibited.
- **Registry Enforcement**: All architectural constants and strategy hyperparameters MUST be fetched from `config/aegis_canonical_parameters.yaml` (e.g., `n_states=2`, `t_max_live_follow=120`, `t_max_live_fade=40`, `embargo_bars=24`, `m_sl_follow=2.0`, `m_sl_fade=1.5`, `spoofing_discount=0.70`).
- **Override Hierarchy**: Explicit parameter priorities MUST be hardcoded (`embargo_bars` strictly overrides `autocorrelation_lag_threshold`, which strictly overrides `embargo_pct`).

### IV. Dimensional Neutrality & Log-Return Space
Indicators and statistical filters must be invariant to absolute price levels and scale.
- **Dimensionless Mechanics**: CUSUM accumulators and volatility thresholds MUST operate in Log-Return space $r_t = \ln(P_t / P_{t-1})$.
- **No EWMA Price Denominators**: Never use $EWMA(P_t)$ as price denominators for thresholds to avoid the Flash Crash Lag Trap.
- **Timescale Scaling**: Always apply Square-Root of Time scaling ($\sqrt{T}$) when normalizing intraday volatility against macro ATR metrics ($h_{\text{log}, t} = \text{multiplier} \cdot \frac{\text{ATR}_t}{P_t \cdot \sqrt{N_{\text{bars/period}}}}$).

### V. Asymmetric Microstructure & Execution Hardening
Order sizing, stop-loss calculations, and margin mechanics must mirror real-world exchange order books and fee dynamics.
- **Asymmetric SL Rounding (`round_sl_safe`)**: Stop-Loss prices must be rounded directional-safe according to instrument `tick_size`: `math.floor` for Long positions (moving SL further down) and `math.ceil` for Short/Fade positions (moving SL further up).
- **Notional Precision**: Position sizing MUST be floored (`math.floor` / `round_down`) to the exact `lot_step_size`.
- **Dynamic Funding & Margin Erosion**: Liquidation calculators MUST account for cumulative funding loss (`funding_accrued_pct`) eroding `margin_loss_allowance`.
- **PnL Accounting Rules**: The LIQUIDATION exit branch MUST NEVER deduct `fee_exit` (preventing double-counting with exchange clearance fees) and MUST account for net bidirectional funding without double-counting initial margin.
- **Isolated Margin Isolation**: `AccountStateTracker` must isolate Wallet Balance from Unrealized PnL. Open position floating profits CANNOT be counted as available margin for opening new positions.

### VI. Fast-Fail & Deterministic Risk Governance
The system must fail fast on corrupted data rather than degrading into unpredictable behavior.
- **No Silent Fallbacks**: Invalid quantitative inputs (e.g., $L < 1.0$, NaN/Inf arrays, side=0, or unaligned timestamps) MUST immediately raise `ValueError(...)`. Silent fallbacks to default leverage or neutral mode are strictly forbidden.
- **Refit-per-fold Pipeline**: Models (FFD, HMM, Kalman, Meta-Labeler) MUST be re-fitted independently on each fold during cross-validation. Global single-fit is prohibited.

## Institutional Risk, Margin & Execution Standards

### Trade Mode Routing (`classify_trade_mode`)
- Single Source of Truth for trade classification (`follow`, `fade`, `none`).
- When `n_states == 2`: Fade mode requires `fade_enabled AND p_chop_i > max(0.80, fade_regime_gate_threshold)`.
- When `n_states >= 3`: Fade mode requires independent dual checks `p_i < 0.2 AND p_chop_i > fade_regime_gate_threshold`.
- Deadzone $[0.2, 0.5)$ strictly routes to `none` (Stand Aside - Zero Risk).

### Empirical Kelly & Volatility Targeting Sizing
- **Small Sample Capping**: Empirical Kelly fraction $f^*$ must be dynamically capped by sample size $N$: $f_{\text{max\_dynamic}} = \min(f_{\text{max\_cap}}, \max(1.0, \sqrt{N}))$.
- **Vol-Targeting Upscaling Gate**: Volatility scaling up to $2.5\times$ is allowed during calm regimes (`clamp(vol_ratio, 0.2, 2.5)`), but the final position notional MUST be validated and clamped against the maximum safe leverage gate ($L_{\max}$) derived from the Stop-Loss distance.

### Progressive Fallback & API Rate Limit Defense
- **Tier 0 ($\le 500\text{ms}$)**: Standard Limit orders.
- **Tier 1 ($500\text{ms} < \text{lag} \le 2000\text{ms}$)**: Switch to `POST_ONLY_LIMIT` with extended TTL ($15\text{s}$), safety buffer $\times 1.5$, and Exponential Backoff (`backoff_multiplier = 2.0`, `max_retry_attempts = 3`) to prevent OTR Ban / HTTP 429.
- **Tier 2 ($> 2000\text{ms}$ or corrupt L2)**: Cancel resting limit orders. Route risk-reducing intents (EXIT, SL) to `FORCE_MARKET`; route risk-increasing intents (ENTRY) to `ABORT_ENTRY`.

### Multi-Tier Circuit Breaker
- **Tier 1 (5% Peak Drawdown)**: Reduce max position size by 50%.
- **Tier 2 (10% Peak Drawdown)**: Flatten 100% of open positions, freeze new entries for 24 hours.
- **Tier 3 (15% Peak Drawdown)**: Emergency Kill-Switch. Halt trading, cancel all orders, require manual human re-authorization.
- All drawdown metrics MUST be calculated from the High-Water Mark (`peak_equity`), never initial capital.

## AI-Agentic Workflow, Quality Gates & Refit Protocols

### Test-First Development (TDD)
- Tests MUST be written and fail BEFORE implementing any core logic or bug fix.
- Unit tests MUST NOT reside inside production algorithm files in `src/`. All tests MUST be placed under `tests/` following proper separation of concerns.
- NEVER use placeholder tests (`assert True` or mock pass-throughs) for acceptance or full-chain golden path checks.

### Python-Rust Parity Assurance
- Any numerical algorithm implemented in Python MUST have matching outputs in Rust RTK with maximum absolute difference $< 10^{-9}$ (and $0$ sign flips across full-chain execution).

### Full Pipeline Retrain Requirement
- Any architectural modification affecting event generation (e.g., CUSUM changes) or labeling MUST trigger an automated full pipeline retrain (Refit Stage 0 through Stage 5) to regenerate synchronized binary artifacts (`artifacts/*.json`, `*.onnx`).

## Governance
- **Supremacy**: This Constitution supersedes all ad-hoc coding instructions, prompt variations, and informal specifications.
- **Amendments**: Amendments to this Constitution require formal backtest/CPCV mathematical proof, approval from the Lead Quant, and explicit updating of `config/aegis_canonical_parameters.yaml`.
- **Automated Verification**: All Pull Requests and AI-generated code MUST pass automated regression testing (`pytest tests/ -v`) and parameter registry compliance checks (`test_canonical_params.py`).

**Version**: 12.0.0 | **Ratified**: 2026-07-23 | **Last Amended**: 2026-07-23
