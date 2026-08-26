#!/usr/bin/env python3
"""Automated Cloud Synchronization Utility for MLflow (Local SQLite -> DAGsHub).

Scans local `mlflow.db`, identifies any experiments or runs that have not yet
been pushed to DAGsHub, and uploads all their hyperparameters, epoch metric
histories, visual plots, configuration snapshots, and DVC tags.
"""

import argparse
import os
import sys
from os.path import join as pjoin
from typing import Any

import dagshub
import mlflow
from mlflow.entities import Metric, Param, RunTag
from mlflow.tracking import MlflowClient


def get_default_paths() -> tuple[str, str]:
    """Resolves default paths for pump2d_smart and local mlflow.db."""
    smart_dir = os.path.abspath(pjoin(os.path.dirname(__file__), "..", ".."))
    local_db_path = pjoin(smart_dir, "mlflow.db")
    return smart_dir, local_db_path


def sync_experiments(
    repo_owner: str = "victor101pepe",
    repo_name: str = "POC_Tests_Water_Pumps",
    local_db_path: str = "",
    dry_run: bool = False,
) -> int:
    """Synchronizes all missing local runs from SQLite to DAGsHub.

    Args:
        repo_owner: DAGsHub username or organization.
        repo_name: DAGsHub repository name.
        local_db_path: Path to local SQLite database file.
        dry_run: If True, only previews runs to be synced without uploading.

    Returns:
        int: Number of newly synchronized runs.
    """
    smart_dir, default_db = get_default_paths()
    local_db_file = local_db_path or default_db

    if not os.path.exists(local_db_file):
        print(f"[Error] Local database not found at: {local_db_file}")
        return 0

    local_uri = f"sqlite:///{os.path.abspath(local_db_file)}"
    print("==================================================")
    print(" 🔄 Automated MLflow Cloud Sync -> DAGsHub")
    print("==================================================")
    print(f" Local SQLite DB: {local_db_file}")
    print(f" Target Repo:     {repo_owner}/{repo_name}")
    print("==================================================")

    # 1. Connect to DAGsHub
    try:
        dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
        remote_uri = mlflow.get_tracking_uri()
    except Exception as e:
        print(f"[Error] Could not initialize DAGsHub connection: {e}")
        return 0

    local_client = MlflowClient(tracking_uri=local_uri)
    remote_client = MlflowClient(tracking_uri=remote_uri)

    local_experiments = local_client.search_experiments()
    total_synced_runs = 0

    for local_exp in local_experiments:
        if local_exp.name == "Default":
            continue

        print(f"\n📁 Checking Experiment: '{local_exp.name}' (Local ID: {local_exp.experiment_id})")

        # Ensure experiment exists on DAGsHub
        remote_exp = remote_client.get_experiment_by_name(local_exp.name)
        if remote_exp is None:
            if dry_run:
                print(f"   [Dry Run] Would create remote experiment '{local_exp.name}'")
                remote_exp_id = "dry_run_id"
            else:
                remote_exp_id = remote_client.create_experiment(name=local_exp.name)
                print(f"   ✓ Created remote experiment '{local_exp.name}' (Remote ID: {remote_exp_id})")
        else:
            remote_exp_id = remote_exp.experiment_id
            print(f"   ✓ Found remote experiment '{local_exp.name}' (Remote ID: {remote_exp_id})")

        # Get existing remote runs to avoid duplicates
        existing_remote_runs = set()
        if remote_exp is not None:
            remote_runs = remote_client.search_runs(experiment_ids=[remote_exp_id])
            for r in remote_runs:
                r_name = r.data.tags.get("mlflow.runName", r.info.run_name or "")
                if r_name:
                    existing_remote_runs.add(r_name)
                # Also check local_run_id tag if present
                local_id_tag = r.data.tags.get("local_run_id")
                if local_id_tag:
                    existing_remote_runs.add(local_id_tag)

        # Get local runs
        local_runs = local_client.search_runs(
            experiment_ids=[local_exp.experiment_id],
            order_by=["attribute.start_time ASC"],
        )

        untracked_runs = []
        for r in local_runs:
            r_name = r.data.tags.get("mlflow.runName", r.info.run_name or "")
            if r_name not in existing_remote_runs and r.info.run_id not in existing_remote_runs:
                untracked_runs.append(r)

        if not untracked_runs:
            print(f"   ✓ All {len(local_runs)} local runs are already synced on DAGsHub!")
            continue

        print(f"   🚀 Found {len(untracked_runs)} untracked run(s) to upload:\n")

        for idx, local_run in enumerate(untracked_runs, 1):
            run_name = local_run.data.tags.get("mlflow.runName", local_run.info.run_name or f"run_{idx}")
            print(f"   [{idx}/{len(untracked_runs)}] Syncing '{run_name}' (Local ID: {local_run.info.run_id})...")

            if dry_run:
                print("      [Dry Run] Would upload parameters, metrics, and plots.")
                total_synced_runs += 1
                continue

            # Create remote run
            tags = [RunTag(k, str(v)) for k, v in local_run.data.tags.items() if not k.startswith("mlflow.source")]
            tags.append(RunTag("mlflow.runName", run_name))
            tags.append(RunTag("local_run_id", local_run.info.run_id))

            new_run = remote_client.create_run(
                experiment_id=remote_exp_id,
                start_time=local_run.info.start_time,
                tags={t.key: t.value for t in tags},
                run_name=run_name,
            )
            new_run_id = new_run.info.run_id

            # Set DVC pointer tags
            model_filename = f"model_{new_run_id}.pt"
            dvc_pointer = f"{model_filename}.dvc"
            remote_client.set_tag(new_run_id, "model_filename", model_filename)
            remote_client.set_tag(new_run_id, "dvc_pointer", dvc_pointer)

            # Upload Parameters
            params = [Param(k, str(v)[:490]) for k, v in local_run.data.params.items()]
            params.append(Param("model_filename", model_filename))
            params.append(Param("dvc_pointer", dvc_pointer))
            for i in range(0, len(params), 100):
                remote_client.log_batch(new_run_id, params=params[i:i+100])

            # Upload Full Metric Histories
            all_metrics = []
            for metric_key in local_run.data.metrics.keys():
                history = local_client.get_metric_history(local_run.info.run_id, metric_key)
                for m in history:
                    all_metrics.append(Metric(key=m.key, value=m.value, timestamp=m.timestamp, step=m.step))

            for i in range(0, len(all_metrics), 500):
                remote_client.log_batch(new_run_id, metrics=all_metrics[i:i+500])

            print(f"      ✓ Uploaded {len(params)} parameters and {len(all_metrics)} metric steps.")

            # Upload Visual Artifacts (plots and configs)
            # Check mlflow_artifacts or experiments/<run_name>/plots
            possible_art_paths = [
                pjoin(smart_dir, "mlflow_artifacts", str(local_exp.experiment_id), local_run.info.run_id, "artifacts"),
                pjoin(smart_dir, "experiments", run_name, "plots"),
            ]

            for art_dir in possible_art_paths:
                if os.path.exists(art_dir):
                    if art_dir.endswith("plots"):
                        remote_client.log_artifacts(new_run_id, art_dir, artifact_path="plots")
                        print("      ✓ Uploaded evaluation plots")
                    else:
                        for item in os.listdir(art_dir):
                            item_path = pjoin(art_dir, item)
                            if item == "checkpoints":
                                continue  # Keep binary weights in DVC
                            if os.path.isdir(item_path):
                                remote_client.log_artifacts(new_run_id, item_path, artifact_path=item)
                            elif os.path.isfile(item_path):
                                remote_client.log_artifact(new_run_id, item_path)
                        print("      ✓ Uploaded plots and config snapshots")
                    break

            # Set Terminated Status
            remote_client.set_terminated(
                new_run_id,
                status=local_run.info.status,
                end_time=local_run.info.end_time,
            )
            print(f"      ✓ Done -> DAGsHub Run ID: {new_run_id}\n")
            total_synced_runs += 1

    print("==================================================")
    if dry_run:
        print(f" [Dry Run Complete] {total_synced_runs} run(s) would be synchronized.")
    else:
        print(f" 🚀 Sync Complete! {total_synced_runs} new run(s) uploaded to DAGsHub.")
        print(f" 🌐 Dashboard: https://dagshub.com/{repo_owner}/{repo_name}.mlflow")
    print("==================================================")
    return total_synced_runs


def main():
    parser = argparse.ArgumentParser(description="Sync local MLflow SQLite runs to DAGsHub.")
    parser.add_argument("--repo-owner", default="victor101pepe", help="DAGsHub repo owner")
    parser.add_argument("--repo-name", default="POC_Tests_Water_Pumps", help="DAGsHub repo name")
    parser.add_argument("--db-path", default="", help="Path to local mlflow.db")
    parser.add_argument("--dry-run", action="store_true", help="Preview runs to sync without uploading")

    args = parser.parse_args()
    sync_experiments(
        repo_owner=args.repo_owner,
        repo_name=args.repo_name,
        local_db_path=args.db_path,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
