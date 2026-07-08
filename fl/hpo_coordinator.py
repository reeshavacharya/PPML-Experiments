"""FL HPO coordinator for SLURM-based parallel trials.

Runs sequentially: for each trial it submits SLURM job(s) for that trial,
waits for them to finish, then reads the result and reports it back to Optuna.
isic/nih submit one combined single-GPU job (server + all clients); fets/cas
submit a separate CPU-only server job plus a multi-GPU client array job.

Usage (on the login node or in a long-running SLURM job):
    python fl/hpo_coordinator.py --dataset fets [--n_trials 20] [--hpo_rounds 15] [--num_clients 23]
    python fl/hpo_coordinator.py --dataset cas  [--n_trials 20] [--hpo_rounds 15] [--num_clients 5]
    python fl/hpo_coordinator.py --dataset isic [--n_trials 20] [--hpo_rounds 15] [--num_clients 6]
    python fl/hpo_coordinator.py --dataset nih  [--n_trials 20] [--hpo_rounds 15] [--num_clients 5]

The Optuna study is persisted in optuna_fl{_dataset}.db — re-running resumes from where it left off.
"""
import argparse
import json
import os
import subprocess
import sys
import time

import optuna
from optuna.samplers import TPESampler

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from utils.dataset_config import get_dataset_config

HPO_CONFIGS_DIR = os.path.join(project_root, "fl", "hpo_configs")
HPO_RESULTS_DIR = os.path.join(project_root, "fl", "hpo_results")
SLURM_DIR = os.path.join(project_root, "fl", "slurm")

# Warm-start: seed the very first FL trial with hyperparameters already known
# to work well for that dataset's own centralized HPO (ISIC/NIH only — fets/
# cas keep the original cold-start behavior). Centralized and federated
# search the same (lr, weight_decay) space for the same model, so the
# centralized best is a reasonable, better-than-random prior even though FL's
# distributed-averaging dynamics could shift the true optimum somewhat. Only
# enqueued on a genuinely fresh study so resuming this script doesn't keep
# re-adding the seed trial.
_WARM_START_PARAMS = {
    "isic": {"lr": 2.0513382630874486e-05, "weight_decay": 2.9375384576328313e-06},
    "nih": {"lr": 1.0994335574766187e-05, "weight_decay": 0.0008123245085588687},
    "cas": {"lr": 1e-4, "weight_decay": 1e-5},  # MONAI SwinUNETR recommended defaults for a new dataset
}

# Note: unlike train/hpo_centralized.py, this coordinator has no mid-trial
# pruning at all — run_fl_trial waits for the whole SLURM job to finish and
# reports one final value via study.tell(), with no trial.report()/
# should_prune() calls during a run. So there's no pruner to tune here; FL's
# HPO time budget is controlled directly via --n_trials/--hpo_rounds instead.


def sample_hps(trial, strategy_name):
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    hp = {"lr": lr, "weight_decay": weight_decay}

    if strategy_name != "fedpidavg":
        # Plain FedAvg (IID clients, e.g. CAS): nothing else to tune here.
        return hp

    # PID weights: sample raw values and normalize so alpha+beta+gamma=1
    alpha_raw = trial.suggest_float("alpha_raw", 0.05, 0.9)
    beta_raw = trial.suggest_float("beta_raw", 0.05, 0.9)
    gamma_raw = trial.suggest_float("gamma_raw", 0.05, 0.9)
    total = alpha_raw + beta_raw + gamma_raw
    hp["alpha"], hp["beta"], hp["gamma"] = alpha_raw / total, beta_raw / total, gamma_raw / total
    hp["cost_history_window"] = trial.suggest_int("cost_history_window", 3, 10)
    hp["poisson_include_prob"] = trial.suggest_float("poisson_include_prob", 0.05, 0.3)
    return hp


def sbatch(script_path, env_vars=None, dependency=None, array=None):
    """Submit a SLURM job and return the job ID."""
    cmd = ["sbatch"]
    if dependency:
        cmd += [f"--dependency={dependency}"]
    if array:
        cmd += [f"--array={array}"]
    if env_vars:
        exports = ",".join(f"{k}={v}" for k, v in env_vars.items())
        cmd += [f"--export=ALL,{exports}"]
    cmd.append(script_path)

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"sbatch failed:\n{result.stderr}")

    # "Submitted batch job 123456"
    job_id = result.stdout.strip().split()[-1]
    print(f"  Submitted {os.path.basename(script_path)} → job {job_id}", flush=True)
    return job_id


def wait_for_jobs(job_ids, poll_interval=60):
    """Poll squeue until all given job IDs are no longer running/pending."""
    ids_str = ",".join(str(j) for j in job_ids)
    print(f"  Waiting for jobs: {ids_str}", flush=True)
    while True:
        result = subprocess.run(
            ["squeue", "--jobs", ids_str, "--noheader", "--format=%i"],
            capture_output=True, text=True,
        )
        active = [l.strip() for l in result.stdout.splitlines() if l.strip()]
        if not active:
            break
        print(f"  Still running: {active}  (next check in {poll_interval}s)", flush=True)
        time.sleep(poll_interval)
    print("  All jobs finished.", flush=True)


def run_fl_trial(trial_number, hp_config, args):
    """Submit server + client SLURM jobs for one HPO trial and wait for completion."""
    # Filenames are dataset-namespaced — HPO_CONFIGS_DIR/HPO_RESULTS_DIR are
    # shared across every dataset's coordinator, and each dataset's Optuna
    # study numbers its own trials from 0 independently, so concurrent
    # coordinators (e.g. fets + isic running at the same time) would
    # otherwise silently overwrite each other's config/result file whenever
    # their trial numbers happened to coincide.
    config_path = os.path.join(HPO_CONFIGS_DIR, f"trial_{args.dataset}_{trial_number}.json")
    result_path = os.path.join(HPO_RESULTS_DIR, f"trial_{args.dataset}_{trial_number}.json")

    with open(config_path, "w") as f:
        json.dump(hp_config, f, indent=2)

    # Delete any stale result file left over from a previous study reusing the
    # same trial number (e.g. re-running with a different --study_name/
    # --storage starts trial numbering back at 0). Without this, a trial
    # whose job crashes before writing its own result would silently read the
    # old file instead of falling through to the "result file missing" case.
    if os.path.exists(result_path):
        os.remove(result_path)

    # Delete stale server address file so clients wait for the new server to write it.
    # Without this, clients may start simultaneously with the server and read the
    # previous trial's address, connecting to a dead host.
    _addr_suffix = "" if args.dataset == "fets" else f"_{args.dataset}"
    addr_file = os.path.join(project_root, "fl", f"server_address{_addr_suffix}.txt")
    if os.path.exists(addr_file):
        os.remove(addr_file)
        print(f"  Deleted stale {addr_file}", flush=True)

    env = {
        "TRIAL_ID": str(trial_number),
        "HPO_ROUNDS": str(args.hpo_rounds),
        "HP_CONFIG": config_path,
        "RESULT_PATH": result_path,
        "NUM_CLIENTS": str(args.num_clients),
        "HPO_MAX_STEPS": str(args.hpo_max_steps),
        "HPO_SUBSAMPLE_FRAC": "" if args.hpo_subsample_frac is None else str(args.hpo_subsample_frac),
    }

    # isic/nih run server + all clients as one combined single-GPU SLURM job
    # (fl/slurm/hpo_trial_{isic,nih}.sh) — the clients already serialize GPU
    # residency via a per-GPU flock (see fl/client_isic.py's
    # _acquire_gpu/_release_gpu), so consolidating onto 1 GPU instead of the
    # old 2-GPU (server job + 2-array-task client job) layout costs wall-clock
    # per round, not VRAM headroom, and it frees a whole GPU per trial that
    # was routinely stuck PENDING waiting for 2 simultaneous GPU nodes.
    # fets/cas keep the original separate server+client-array layout
    # unchanged (rest of this function).
    if args.dataset in ("isic", "nih"):
        trial_script = f"hpo_trial_{args.dataset}.sh"
        result_key = "mean_balanced_acc" if args.dataset == "isic" else "mean_auc"
        trial_jid = sbatch(os.path.join(SLURM_DIR, trial_script), env_vars=env)
        wait_for_jobs([trial_jid], poll_interval=args.poll_interval)

        if not os.path.exists(result_path):
            print(f"  WARNING: result file missing ({result_path}). Returning 0.0.", flush=True)
            return 0.0

        with open(result_path) as f:
            result = json.load(f)
        metric_value = float(result.get(result_key, 0.0))
        print(f"  Trial {trial_number} result: {result_key}={metric_value:.4f}", flush=True)
        return metric_value

    # Separate SLURM scripts per dataset (rather than flag-injecting a shared
    # script) — matches this repo's existing one-script-per-experiment-variant
    # convention (e.g. fedpidavg_server.sh vs server.sh).
    if args.dataset == "fets":
        server_script, client_script, client_array = "hpo_server.sh", "hpo_client.sh", "1-4"
        result_key = "mean_dice"
    else:
        # hpo_client_cas.sh groups the 5 CAS clients into 3 concurrent-GPU
        # tasks (2+2+1) — muma_2021 rarely has 5 free GPUs at once.
        server_script, client_script, client_array = "hpo_server_cas.sh", "hpo_client_cas.sh", "1-5"
        result_key = "mean_dice"

    server_jid = sbatch(os.path.join(SLURM_DIR, server_script), env_vars=env)
    # Clients start after server is running (not strictly after it exits)
    client_jid = sbatch(
        os.path.join(SLURM_DIR, client_script),
        env_vars=env,
        dependency=f"after:{server_jid}",
        array=client_array,
    )

    wait_for_jobs([server_jid, client_jid], poll_interval=args.poll_interval)

    if not os.path.exists(result_path):
        print(f"  WARNING: result file missing ({result_path}). Returning 0.0.", flush=True)
        return 0.0

    with open(result_path) as f:
        result = json.load(f)
    metric_value = float(result.get(result_key, 0.0))
    print(f"  Trial {trial_number} result: {result_key}={metric_value:.4f}", flush=True)
    return metric_value


def main():
    parser = argparse.ArgumentParser(description="FL HPO coordinator for SwinUNETR / ViT-B/16 FL experiments")
    parser.add_argument("--dataset", type=str, default="fets", choices=["fets", "cas", "isic", "nih"])
    parser.add_argument("--n_trials", type=int, default=20, help="Total Optuna trials")
    parser.add_argument("--hpo_rounds", type=int, default=15, help="FL rounds per trial")
    parser.add_argument("--num_clients", type=int, default=None, help="Default: per-dataset config")
    parser.add_argument("--study_name", type=str, default=None, help="Default: per-dataset")
    parser.add_argument("--storage", type=str, default=None, help="Default: per-dataset sqlite file")
    parser.add_argument("--poll_interval", type=int, default=60,
                        help="Seconds between squeue polls while waiting for jobs")
    parser.add_argument("--hpo_max_steps", type=int, default=0,
                        help="Limit client training to N gradient steps per round (0=unlimited; passed to server via HPO_MAX_STEPS env)")
    parser.add_argument("--hpo_subsample_frac", type=float, default=None,
                        help="isic/nih only: subsample each client's training split to this fraction, e.g. 0.4 (default: no subsampling)")
    args = parser.parse_args()

    if args.dataset == "isic":
        from data_loader.isic_dataset import NUM_CLIENTS
        strategy_name = "fedavg"
        if args.num_clients is None:
            args.num_clients = NUM_CLIENTS
    elif args.dataset == "nih":
        from data_loader.nih_chest_dataset import NUM_CLIENTS
        strategy_name = "fedavg"
        if args.num_clients is None:
            args.num_clients = NUM_CLIENTS
    else:
        cfg = get_dataset_config(args.dataset)
        strategy_name = cfg["default_strategy"]
        if args.num_clients is None:
            args.num_clients = cfg["default_num_clients"]

    suffix = "" if args.dataset == "fets" else f"_{args.dataset}"
    if args.study_name is None:
        args.study_name = f"fl_hpo{suffix}"
    if args.storage is None:
        args.storage = f"sqlite:///{os.path.join(project_root, f'optuna_fl{suffix}.db')}"
    hpo_best_path = os.path.join(project_root, "fl", f"hpo_best_config{suffix}.json")

    os.makedirs(HPO_CONFIGS_DIR, exist_ok=True)
    os.makedirs(HPO_RESULTS_DIR, exist_ok=True)

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="maximize",
        sampler=TPESampler(seed=42),
        load_if_exists=True,
    )

    if args.dataset in _WARM_START_PARAMS and len(study.trials) == 0:
        study.enqueue_trial(_WARM_START_PARAMS[args.dataset])
        print(f"Warm-started study with known-good params: {_WARM_START_PARAMS[args.dataset]}", flush=True)

    already_done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    remaining = args.n_trials - already_done
    if remaining <= 0:
        print(f"Study already has {already_done} completed trials. Nothing to do.")
    else:
        print(f"Starting FL HPO: {remaining} trial(s) remaining (target={args.n_trials})", flush=True)

    for _ in range(remaining):
        trial = study.ask()
        hp_config = sample_hps(trial, strategy_name)

        print(f"\n[Trial {trial.number}] HP config: {json.dumps(hp_config, indent=2)}", flush=True)

        try:
            metric_value = run_fl_trial(trial.number, hp_config, args)
            study.tell(trial, metric_value)
        except Exception as e:
            print(f"[Trial {trial.number}] FAILED: {e}", flush=True)
            study.tell(trial, 0.0, state=optuna.trial.TrialState.FAIL)

    _RESULT_KEYS = {
        "isic": ("best_val_balanced_accuracy", "Val Balanced Accuracy"),
        "nih": ("best_val_mean_auc_roc", "Val Mean AUC-ROC"),
    }
    metric_key, metric_label = _RESULT_KEYS.get(args.dataset, ("best_val_mean_dice", "Val mean Dice"))

    best = study.best_trial
    print("\n" + "=" * 60)
    print(f"Best FL trial #{best.number}  {metric_label}: {best.value:.4f}")
    for k, v in best.params.items():
        print(f"  {k}: {v:.6g}")
    print("=" * 60)

    result = {
        "study_name": args.study_name,
        "best_trial": best.number,
        metric_key: round(best.value, 4),
        "best_params": {k: float(v) for k, v in best.params.items()},
        "n_trials_completed": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
        "hpo_rounds": args.hpo_rounds,
    }
    with open(hpo_best_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Best config saved to {hpo_best_path}")


if __name__ == "__main__":
    main()
