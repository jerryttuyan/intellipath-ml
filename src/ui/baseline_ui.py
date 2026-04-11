"""
Streamlit UI for running the traffic forecasting baseline experiment.
"""

from __future__ import annotations

import html
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
else:
    repo_root = Path(__file__).resolve().parents[2]

if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.config import CONFIG
from src.presets import PRESET_KEYS, list_presets, load_preset, save_preset
from src.suites import list_suites


def _inject_styles() -> None:
    st.markdown(
        """
<style>
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}
.app-card {
    border: 1px solid rgba(49, 51, 63, 0.2);
    border-radius: 14px;
    padding: 0.9rem 1rem;
    margin-bottom: 1rem;
    background: rgba(250, 250, 255, 0.35);
}
</style>
        """,
        unsafe_allow_html=True,
    )


def _default_run_state() -> dict[str, Any]:
    return {
        "data_path": str(CONFIG.get("data_path", "data/raw/LargeST/gla/gla_his_2019.h5")),
        "horizon": int(CONFIG.get("horizon", 15)),
        "target_node": "" if CONFIG.get("target_node") is None else str(CONFIG.get("target_node")),
        "target_node_index": int(CONFIG.get("target_node_index", 0)),
        "num_target_nodes": int(CONFIG.get("num_target_nodes", 10)),
        "train_ratio": float(CONFIG.get("train_ratio", 0.7)),
        "val_ratio": float(CONFIG.get("val_ratio", 0.15)),
        "rf_n_estimators": int(CONFIG.get("rf_n_estimators", 100)),
        "rf_random_state": int(CONFIG.get("rf_random_state", 42)),
        "rf_n_jobs": int(CONFIG.get("rf_n_jobs", 1)),
        "results_path": str(CONFIG.get("results_path", "results/baseline_comparison.csv")),
        "summary_results_path": str(CONFIG.get("summary_results_path", "results/baseline_summary.csv")),
    }


def _build_args(run_state: dict[str, Any], preset_name: str) -> list[str]:
    args = [
        "--data-path",
        run_state["data_path"],
        "--horizon",
        str(run_state["horizon"]),
        "--target-node-index",
        str(run_state["target_node_index"]),
        "--num-target-nodes",
        str(run_state["num_target_nodes"]),
        "--train-ratio",
        str(run_state["train_ratio"]),
        "--val-ratio",
        str(run_state["val_ratio"]),
        "--rf-n-estimators",
        str(run_state["rf_n_estimators"]),
        "--rf-random-state",
        str(run_state["rf_random_state"]),
        "--rf-n-jobs",
        str(run_state["rf_n_jobs"]),
        "--results-path",
        run_state["results_path"],
        "--summary-results-path",
        run_state["summary_results_path"],
        "--show-config",
    ]
    if preset_name:
        args.extend(["--preset", preset_name])
    target_node_raw = run_state.get("target_node")
    target_node = "" if target_node_raw is None else str(target_node_raw).strip()
    if target_node:
        args.extend(["--target-node", target_node])
    return args


def _render_results(summary_results_path: str, results_path: str) -> None:
    summary_file = repo_root / summary_results_path
    results_file = repo_root / results_path

    if summary_file.exists():
        summary_df = pd.read_csv(summary_file)
        st.subheader("Summary Results")
        if {"model", "mean_mae", "mean_rmse"}.issubset(summary_df.columns):
            best = summary_df.sort_values("mean_mae").iloc[0]
            col1, col2, col3 = st.columns(3)
            col1.metric("Best Model", str(best["model"]))
            col2.metric("Best Mean MAE", f"{float(best['mean_mae']):.4f}")
            col3.metric("Best Mean RMSE", f"{float(best['mean_rmse']):.4f}")
            chart_df = summary_df.set_index("model")[["mean_mae", "mean_rmse"]]
            st.bar_chart(chart_df)
        st.dataframe(summary_df, use_container_width=True)
    else:
        st.info(f"Summary file not found yet: `{summary_results_path}`")

    if results_file.exists():
        results_df = pd.read_csv(results_file)
        st.subheader("Per-Node Results")
        st.dataframe(results_df, use_container_width=True)
    else:
        st.info(f"Detailed results file not found yet: `{results_path}`")


def _render_run_history() -> None:
    history_file = repo_root / "results" / "run_history.csv"
    st.subheader("Run History Dashboard")
    if not history_file.exists():
        st.info("No run history yet. Execute a run to populate `results/run_history.csv`.")
        return

    history_df = pd.read_csv(history_file)
    if history_df.empty:
        st.info("Run history exists but has no rows.")
        return

    if "timestamp_utc" in history_df.columns:
        history_df["timestamp_utc"] = pd.to_datetime(history_df["timestamp_utc"], errors="coerce")
        history_df = history_df.sort_values("timestamp_utc")

    if {"rf_mean_mae", "persistence_mean_mae"}.issubset(history_df.columns):
        history_df["rf_mae_gain"] = history_df["persistence_mean_mae"] - history_df["rf_mean_mae"]
    if {"linear_mean_mae", "persistence_mean_mae"}.issubset(history_df.columns):
        history_df["linear_mae_gain"] = history_df["persistence_mean_mae"] - history_df["linear_mean_mae"]
    if {"rf_mean_rmse", "persistence_mean_rmse"}.issubset(history_df.columns):
        history_df["rf_rmse_gain"] = history_df["persistence_mean_rmse"] - history_df["rf_mean_rmse"]
    if {"linear_mean_rmse", "persistence_mean_rmse"}.issubset(history_df.columns):
        history_df["linear_rmse_gain"] = history_df["persistence_mean_rmse"] - history_df["linear_mean_rmse"]

    metric_cols = [
        col
        for col in [
            "rf_mean_mae",
            "linear_mean_mae",
            "persistence_mean_mae",
            "rf_mae_gain",
            "linear_mae_gain",
        ]
        if col in history_df.columns
    ]
    if metric_cols:
        st.markdown("**MAE Trend Across Runs**")
        mae_plot_df = history_df.set_index("run_id")[metric_cols]
        st.line_chart(mae_plot_df)

    rmse_cols = [
        col
        for col in [
            "rf_mean_rmse",
            "linear_mean_rmse",
            "persistence_mean_rmse",
            "rf_rmse_gain",
            "linear_rmse_gain",
        ]
        if col in history_df.columns
    ]
    if rmse_cols:
        st.markdown("**RMSE Trend Across Runs**")
        rmse_plot_df = history_df.set_index("run_id")[rmse_cols]
        st.line_chart(rmse_plot_df)

    runtime_cols = [col for col in ["elapsed_seconds", "num_target_nodes", "rf_n_estimators"] if col in history_df.columns]
    if runtime_cols:
        st.markdown("**Runtime and Scale Trend**")
        runtime_plot_df = history_df.set_index("run_id")[runtime_cols]
        st.line_chart(runtime_plot_df)

    display_cols = [
        col
        for col in [
            "run_id",
            "timestamp_utc",
            "preset",
            "num_target_nodes",
            "rf_n_estimators",
            "rf_n_jobs",
            "rf_mean_mae",
            "linear_mean_mae",
            "persistence_mean_mae",
            "rf_mae_gain",
            "linear_mae_gain",
            "elapsed_seconds",
        ]
        if col in history_df.columns
    ]
    st.markdown("**Run History Table**")
    st.dataframe(history_df[display_cols], use_container_width=True)


def _render_routing_dashboard() -> None:
    st.subheader("Routing Dashboard")
    summary_path = repo_root / "results" / "routing_summary.csv"
    comparison_path = repo_root / "results" / "routing_comparison.csv"
    history_path = repo_root / "results" / "routing_history.csv"

    if summary_path.exists():
        summary_df = pd.read_csv(summary_path)
        if not summary_df.empty:
            st.markdown("**Latest Routing Summary**")
            best_row = summary_df.sort_values("mean_realized_cost").iloc[0]
            c1, c2, c3 = st.columns(3)
            c1.metric("Best Routing Method", str(best_row["method"]))
            c2.metric("Best Mean Travel Cost", f"{float(best_row['mean_realized_cost']):.4f}")
            c3.metric("Best Mean Regret", f"{float(best_row['mean_regret']):.4f}")
            if {"method", "mean_realized_cost", "p90_realized_cost"}.issubset(summary_df.columns):
                chart_df = summary_df.set_index("method")[["mean_realized_cost", "p90_realized_cost"]]
                st.bar_chart(chart_df)
            st.dataframe(summary_df, use_container_width=True)
    else:
        st.info("No routing summary found yet. Run `python src/run_routing_experiment.py` first.")

    if comparison_path.exists():
        comparison_df = pd.read_csv(comparison_path)
        if not comparison_df.empty:
            st.markdown("**Latest Routing Per-Route Results (sample)**")
            st.dataframe(comparison_df.head(200), use_container_width=True)

    if history_path.exists():
        history_df = pd.read_csv(history_path)
        if not history_df.empty:
            st.markdown("**Routing Run History**")
            if "timestamp_utc" in history_df.columns:
                history_df["timestamp_utc"] = pd.to_datetime(history_df["timestamp_utc"], errors="coerce")
                history_df = history_df.sort_values("timestamp_utc")

            trend_cols = [
                col
                for col in ["best_mean_realized_cost", "elapsed_seconds", "num_nodes", "num_od_pairs"]
                if col in history_df.columns
            ]
            if trend_cols:
                trend_df = history_df.set_index("run_id")[trend_cols]
                st.line_chart(trend_df)

            display_cols = [
                col
                for col in [
                    "run_id",
                    "timestamp_utc",
                    "prediction_model",
                    "horizon_minutes",
                    "num_nodes",
                    "num_timestamps",
                    "num_od_pairs",
                    "best_method",
                    "best_mean_realized_cost",
                    "elapsed_seconds",
                ]
                if col in history_df.columns
            ]
            st.dataframe(history_df[display_cols], use_container_width=True)


def _render_explainer_sections() -> None:
    st.markdown("### How To Use This Page")
    st.markdown(
        """
1. Load a shared preset from the sidebar (or keep defaults).
2. Adjust settings for this run (node count, model size, CPU workers).
3. Run the experiment and check the **Run Output** config block.
4. Read **Summary Results** and compare against **Persistence** baseline.
5. Track trends in the **Run History Dashboard**.
        """
    )

    with st.expander("Metric Glossary and Interpretation", expanded=False):
        st.markdown(
            """
- **Persistence baseline**: predicts the future traffic value as the current value.  
  It is a simple benchmark; learned models should beat it.

- **MAE (Mean Absolute Error)**: average absolute prediction error.  
  **Lower is better.**

- **RMSE (Root Mean Squared Error)**: similar to MAE but penalizes larger misses more.  
  **Lower is better.**

- **RF gain (MAE/RMSE gain)**: `Persistence error - Random Forest error`.  
  Positive gain means Random Forest is better than persistence.

- **Linear gain (MAE/RMSE gain)**: `Persistence error - Linear Regression error`.  
  Positive gain means Linear Regression is better than persistence.

- **num_target_nodes**: how many sensors are evaluated.  
  Larger values are more representative but take longer.

- **rf_n_estimators**: number of trees in Random Forest.  
  More trees may improve stability but increase runtime.

- **rf_n_jobs**: CPU parallelism for Random Forest (`-1` all cores, `1` single core).  
  Set per machine comfort/performance.
            """
        )


def _ensure_runtime_state() -> None:
    if "run_state" not in st.session_state:
        st.session_state.run_state = _default_run_state()
    if "selected_preset" not in st.session_state:
        st.session_state.selected_preset = ""
    if "process" not in st.session_state:
        st.session_state.process = None
    if "process_log_path" not in st.session_state:
        st.session_state.process_log_path = ""
    if "process_log_handle" not in st.session_state:
        st.session_state.process_log_handle = None
    if "process_status" not in st.session_state:
        st.session_state.process_status = "idle"
    if "process_command" not in st.session_state:
        st.session_state.process_command = ""
    if "process_started_at" not in st.session_state:
        st.session_state.process_started_at = None
    if "last_run_elapsed_seconds" not in st.session_state:
        st.session_state.last_run_elapsed_seconds = None
    if "selected_suite" not in st.session_state:
        st.session_state.selected_suite = "required_suite"


def _tail_log_text(log_path: str, max_chars: int = 30000, tail_lines: int = 12) -> str:
    path = Path(log_path)
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="ignore")
    if len(text) > max_chars:
        text = text[-max_chars:]
    lines = text.splitlines()[-tail_lines:]
    return "\n".join(lines)


def _render_live_log_panel(log_text: str) -> None:
    escaped = html.escape(log_text)
    st.markdown(
        f"""
<div style="height: 220px; overflow-y: auto; border: 1px solid rgba(49, 51, 63, 0.2); border-radius: 8px; padding: 0.5rem; background: #0e1117;">
<pre style="margin: 0; color: #d5d7e0; font-size: 0.85rem; white-space: pre-wrap;">{escaped}</pre>
</div>
        """,
        unsafe_allow_html=True,
    )


def _start_background_process(command: list[str]) -> None:
    runs_dir = repo_root / "results" / "ui_logs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = runs_dir / f"{run_id}.log"
    log_handle = log_path.open("w", encoding="utf-8", buffering=1)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        command,
        cwd=repo_root,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )

    st.session_state.process = process
    st.session_state.process_log_path = str(log_path)
    st.session_state.process_log_handle = log_handle
    st.session_state.process_status = "running"
    st.session_state.process_command = " ".join(command)
    st.session_state.process_started_at = time.time()


def _start_single_run(args: list[str]) -> None:
    _start_background_process([sys.executable, "-u", "src/run_baseline_experiment.py", *args])


def _start_suite_run(suite_name: str) -> None:
    _start_background_process(
        [sys.executable, "-u", "src/run_experiment_suite.py", "--suite", suite_name]
    )


def _stop_background_run() -> None:
    process = st.session_state.process
    if process is None:
        return
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
    if st.session_state.process_log_handle:
        st.session_state.process_log_handle.close()
        st.session_state.process_log_handle = None
    st.session_state.process = None
    if st.session_state.process_started_at is not None:
        st.session_state.last_run_elapsed_seconds = time.time() - st.session_state.process_started_at
    st.session_state.process_started_at = None
    st.session_state.process_status = "stopped"


def _refresh_process_state() -> None:
    process = st.session_state.process
    if process is None or st.session_state.process_status != "running":
        return
    exit_code = process.poll()
    if exit_code is None:
        return
    if st.session_state.process_log_handle:
        st.session_state.process_log_handle.close()
        st.session_state.process_log_handle = None
    if st.session_state.process_started_at is not None:
        st.session_state.last_run_elapsed_seconds = time.time() - st.session_state.process_started_at
    st.session_state.process_started_at = None
    st.session_state.process_status = "completed" if exit_code == 0 else "failed"
    st.session_state.process = None


def main() -> None:
    st.set_page_config(page_title="IntelliPath Baseline Runner", layout="wide")
    _inject_styles()
    st.title("IntelliPath Baseline Runner")
    st.caption("Run and reproduce forecasting baselines without editing source code.")
    _render_explainer_sections()
    _ensure_runtime_state()
    _refresh_process_state()

    with st.sidebar:
        st.markdown("### Shared Presets")
        preset_names = list_presets(repo_root)
        preset_options = ["(none)"] + preset_names
        selected = st.selectbox(
            "Load preset",
            options=preset_options,
            help=(
                "Choose a shared, non-device preset from config/presets. "
                "Applying a preset fills common experiment settings."
            ),
            index=0 if not st.session_state.selected_preset else preset_options.index(st.session_state.selected_preset)
            if st.session_state.selected_preset in preset_options
            else 0,
        )
        if st.button("Apply preset", use_container_width=True):
            if selected == "(none)":
                st.session_state.selected_preset = ""
                st.info("Using defaults and manual overrides.")
            else:
                loaded = load_preset(repo_root, selected)
                for key in PRESET_KEYS:
                    if key in loaded:
                        if key == "target_node":
                            st.session_state.run_state[key] = "" if loaded[key] is None else str(loaded[key])
                        else:
                            st.session_state.run_state[key] = loaded[key]
                st.session_state.selected_preset = selected
                st.success(f"Loaded preset: {selected}")
                st.rerun()

        st.markdown("---")
        new_preset_name = st.text_input(
            "Save current settings as preset",
            value="",
            help=(
                "Saves portable experiment settings (no device-specific values like CPU core count). "
                "Use lowercase names like 'report_300_nodes'."
            ),
        )
        if st.button("Save preset", use_container_width=True):
            payload = {
                key: st.session_state.run_state.get(key)
                for key in PRESET_KEYS
            }
            try:
                saved_path = save_preset(repo_root, new_preset_name, payload)
                st.success(f"Saved {saved_path.relative_to(repo_root)}")
            except Exception as exc:  # pylint: disable=broad-except
                st.error(str(exc))

        st.markdown("---")
        st.markdown("### Experiment Suites")
        suite_names = list_suites(repo_root)
        if suite_names:
            suite_index = suite_names.index(st.session_state.selected_suite) if st.session_state.selected_suite in suite_names else 0
            st.session_state.selected_suite = st.selectbox(
                "Suite",
                options=suite_names,
                index=suite_index,
                help="A suite runs multiple required experiments in sequence.",
            )
            if st.button(
                "Run Selected Suite",
                use_container_width=True,
                disabled=st.session_state.process_status == "running",
            ):
                _start_suite_run(st.session_state.selected_suite)
                st.rerun()
        else:
            st.info("No suite definitions found in config/suites.")

    with st.form("baseline_config_form"):
        st.markdown('<div class="app-card">', unsafe_allow_html=True)
        st.subheader("Experiment Settings")
        col1, col2, col3 = st.columns(3)

        with col1:
            data_path = st.text_input(
                "Data path (default: `data/raw/LargeST/gla/gla_his_2019.h5`)",
                value=str(st.session_state.run_state["data_path"]),
                help="Path to the input traffic history file (.h5).",
            )
            horizon = st.number_input(
                "Horizon (minutes, default: 15)",
                min_value=1,
                value=int(st.session_state.run_state["horizon"]),
                step=1,
                help="How many minutes ahead to predict (e.g., 15 means forecast 15 minutes into the future).",
            )
            target_node = st.text_input(
                "Target node (optional, default: blank)",
                value="" if st.session_state.run_state["target_node"] is None else str(st.session_state.run_state["target_node"]),
                help="Leave blank to use index + count node sweep.",
            )
            target_node_index = st.number_input(
                "Target node start index (default: 0)",
                min_value=0,
                value=int(st.session_state.run_state["target_node_index"]),
                step=1,
                help="Starting column position when selecting a consecutive node range.",
            )

        with col2:
            num_target_nodes = st.number_input(
                "Number of target nodes (default: 10)",
                min_value=1,
                value=int(st.session_state.run_state["num_target_nodes"]),
                step=1,
                help="How many consecutive sensor nodes to evaluate.",
            )
            train_ratio = st.number_input(
                "Train ratio (default: 0.70)",
                min_value=0.01,
                max_value=0.98,
                value=float(st.session_state.run_state["train_ratio"]),
                step=0.01,
                format="%.2f",
                help="Fraction of timeline used for training.",
            )
            val_ratio = st.number_input(
                "Validation ratio (default: 0.15)",
                min_value=0.01,
                max_value=0.98,
                value=float(st.session_state.run_state["val_ratio"]),
                step=0.01,
                format="%.2f",
                help="Fraction of timeline used for validation (train + val must stay under 1).",
            )

        with col3:
            rf_n_estimators = st.number_input(
                "RF n_estimators (default: 100)",
                min_value=1,
                value=int(st.session_state.run_state["rf_n_estimators"]),
                step=1,
                help="Number of trees in the Random Forest (higher can improve stability but runs slower).",
            )
            rf_random_state = st.number_input(
                "RF random_state (default: 42)",
                value=int(st.session_state.run_state["rf_random_state"]),
                step=1,
                help="Seed for reproducibility. Keep fixed for fair team comparisons.",
            )
            rf_n_jobs = st.number_input(
                "RF n_jobs (default: -1)",
                value=int(st.session_state.run_state["rf_n_jobs"]),
                step=1,
                help="-1 uses all cores, 1 single core, or any positive cap.",
            )

        st.subheader("Output Paths")
        results_path = st.text_input(
            "Detailed results CSV (default: `results/baseline_comparison.csv`)",
            value=str(st.session_state.run_state["results_path"]),
            help="Where to save per-node prediction metrics.",
        )
        summary_results_path = st.text_input(
            "Summary results CSV (default: `results/baseline_summary.csv`)",
            value=str(st.session_state.run_state["summary_results_path"]),
            help="Where to save model-level aggregated metrics.",
        )
        st.markdown("</div>", unsafe_allow_html=True)

        run_clicked = st.form_submit_button(
            "Run Baseline Experiment",
            type="primary",
            disabled=st.session_state.process_status == "running",
        )

    if run_clicked:
        st.session_state.run_state.update(
            {
                "data_path": data_path,
                "horizon": int(horizon),
                "target_node": target_node,
                "target_node_index": int(target_node_index),
                "num_target_nodes": int(num_target_nodes),
                "train_ratio": float(train_ratio),
                "val_ratio": float(val_ratio),
                "rf_n_estimators": int(rf_n_estimators),
                "rf_random_state": int(rf_random_state),
                "rf_n_jobs": int(rf_n_jobs),
                "results_path": results_path,
                "summary_results_path": summary_results_path,
            }
        )

        if train_ratio + val_ratio >= 1:
            st.error("Invalid split: train_ratio + val_ratio must be less than 1.")
            st.stop()

        args = _build_args(st.session_state.run_state, st.session_state.selected_preset)
        _start_single_run(args)
        st.rerun()

    st.subheader("Live Run Output")
    if st.session_state.process_command:
        st.code(st.session_state.process_command, language="bash")

    stop_col, status_col = st.columns([1, 3])
    with stop_col:
        if st.button(
            "Stop Run",
            disabled=st.session_state.process_status != "running",
            type="secondary",
            use_container_width=True,
        ):
            _stop_background_run()
            st.rerun()
    with status_col:
        st.markdown(f"**Status:** `{st.session_state.process_status}`")
        if st.session_state.process_status == "running" and st.session_state.process_started_at is not None:
            current_elapsed = time.time() - st.session_state.process_started_at
            st.markdown(f"Current run elapsed: `{current_elapsed:.1f}s`")
        if st.session_state.last_run_elapsed_seconds is not None:
            st.markdown(f"Last run duration: `{st.session_state.last_run_elapsed_seconds:.1f}s`")

    log_text = _tail_log_text(st.session_state.process_log_path) if st.session_state.process_log_path else ""
    _render_live_log_panel(log_text)

    _render_results(
        summary_results_path=str(st.session_state.run_state["summary_results_path"]),
        results_path=str(st.session_state.run_state["results_path"]),
    )
    _render_run_history()
    _render_routing_dashboard()

    if st.session_state.process_status == "running":
        time.sleep(1)
        st.rerun()


if __name__ == "__main__":
    main()
