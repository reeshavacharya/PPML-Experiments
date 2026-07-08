import argparse
import json
import os
import platform
import random
import sys

import flwr as fl

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from models.vit_classifier import get_model
from data_loader.nih_chest_dataset import NUM_CLASSES, NUM_CLIENTS
from fl.strategy_fedavg import CASFedAvgStrategy, fit_metrics_aggregation_fn


def _ts():
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def evaluate_metrics_aggregation_fn(results):
    if not results:
        return {}
    total = sum(n for n, _ in results)
    weighted = sum(n * m.get("val_mean_auc", 0) for n, m in results)
    return {"val_mean_auc": weighted / total if total else 0}


def main():
    parser = argparse.ArgumentParser(description="Flower Server for ViT-B/16 NIH ChestX-ray14 FL experiments (plain FedAvg, IID)")
    parser.add_argument("--server_address", type=str, default="0.0.0.0:8082")
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--num_clients", type=int, default=NUM_CLIENTS)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--hp_config", type=str, default=None,
                        help="Path to JSON file with HPO config (overrides lr/weight_decay)")
    parser.add_argument("--result_path", type=str, default=None,
                        help="Path to write trial result JSON {mean_auc, best_round} for HPO coordinator")
    parser.add_argument("--lr", type=float, default=None,
                        help="Client learning rate to broadcast via configure_fit (HPO mode)")
    parser.add_argument("--weight_decay", type=float, default=None,
                        help="Client weight decay to broadcast via configure_fit (HPO mode)")
    parser.add_argument("--seed", type=int, default=42, help="Seeds Python's random")
    args = parser.parse_args()

    random.seed(args.seed)

    model_fn = lambda: get_model(num_classes=NUM_CLASSES, pretrained=True)

    seed_suffix = "" if args.seed == 42 else f"_seed{args.seed}"
    resume_filename = f"fl_resume_nih{seed_suffix}.pth"
    best_model_filename = f"vit_nih_fl{seed_suffix}_best.pth"
    resume_path = os.path.join(project_root, "checkpoint", resume_filename)

    if args.hp_config:
        with open(args.hp_config) as f:
            hp = json.load(f)
        args.lr = hp.get("lr", args.lr)
        args.weight_decay = hp.get("weight_decay", args.weight_decay)
        print(f"[{_ts()}] Loaded HP config from {args.hp_config}: {hp}", flush=True)

    checkpoint_dir = os.path.join(project_root, "checkpoint")

    round_offset = 0
    initial_parameters = None
    best_metric = 0.0
    best_round = 0

    if args.resume and os.path.exists(resume_path):
        import torch
        print(f"[{_ts()}] Resuming from {resume_path}...", flush=True)
        ckpt = torch.load(resume_path, weights_only=False)
        round_offset = ckpt["round"]
        best_metric = ckpt["best_dice"]
        best_round = ckpt["best_round"]
        initial_parameters = fl.common.ndarrays_to_parameters(ckpt["parameters"])
        print(f"[{_ts()}]   Completed rounds: {round_offset}", flush=True)
        print(f"[{_ts()}]   Best val mean AUC-ROC: {best_metric:.4f} (round {best_round})", flush=True)
    elif args.resume:
        print(f"[{_ts()}] --resume but no checkpoint found. Starting fresh.", flush=True)

    remaining_rounds = args.rounds - round_offset
    if remaining_rounds <= 0:
        print(f"[{_ts()}] All {args.rounds} rounds completed.", flush=True)
        return

    print("=" * 60, flush=True)
    print(f"[{_ts()}] FedAvg Server (nih) — {platform.node()}", flush=True)
    print(f"  Rounds: {args.rounds} (starting from {round_offset + 1})", flush=True)
    print(f"  Clients: {args.num_clients}", flush=True)
    print(f"  Bind: {args.server_address}", flush=True)
    print("=" * 60, flush=True)

    # NIH-specific handoff file (not the shared fl/server_address.txt used by
    # fets/cas) — a concurrent ISIC + NIH run once caused a client to read
    # the wrong dataset's server hostname off the shared file, and (far more
    # seriously) both servers were bound to the same default port 8080, so
    # a client could receive a completely incompatible model checkpoint
    # from the wrong experiment with no error until the shape-mismatch
    # crash. Per-dataset port (see --server_address default) + per-dataset
    # handoff file together eliminate both failure modes.
    addr_file = os.path.join(project_root, "fl", "server_address_nih.txt")
    with open(addr_file, "w") as f:
        f.write(platform.node())

    strategy = CASFedAvgStrategy(
        checkpoint_dir=checkpoint_dir,
        model_fn=model_fn,
        round_offset=round_offset,
        best_dice=best_metric,
        best_round=best_round,
        client_lr=args.lr,
        client_wd=args.weight_decay,
        resume_filename=resume_filename,
        best_model_filename=best_model_filename,
        metric_key="val_mean_auc",
        metric_label="mean_auc",
        log_label="NIHFedAvgStrategy: plain FedAvg (5 IID clients)",
        initial_parameters=initial_parameters,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=1,
        min_evaluate_clients=1,
        # NOT the same as num_clients: FedAvg.num_fit_clients()/
        # num_evaluate_clients() return this value (not min_fit_clients) as
        # the min_num_clients passed to client_manager.sample() on *every*
        # round, which blocks in wait_for() for up to 24h. Setting this to
        # num_clients forced every round to wait for all clients
        # simultaneously connected — same bug found and fixed in
        # fl/server.py, caused multi-hour Round 1 stalls here too.
        min_available_clients=1,
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
    )

    fl.server.start_server(
        server_address=args.server_address,
        config=fl.server.ServerConfig(num_rounds=remaining_rounds, round_timeout=3600),
        strategy=strategy,
    )

    summary = strategy.get_summary(args.rounds)
    summary_path = os.path.join(project_root, "fl", f"training_summary_nih{seed_suffix}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    if args.result_path:
        os.makedirs(os.path.dirname(os.path.abspath(args.result_path)), exist_ok=True)
        with open(args.result_path, "w") as f:
            json.dump({
                "mean_auc": summary["best_val_mean_auc"],
                "best_round": summary["best_round"],
            }, f)
        print(f"[{_ts()}] HPO trial result written to {args.result_path}", flush=True)

    print(f"\n[{_ts()}] {summary['algorithm']} Training Summary", flush=True)
    print(f"  Best Val Mean AUC-ROC: {summary['best_val_mean_auc']} (round {summary['best_round']})", flush=True)
    print(f"  Total SGD Steps: {summary['total_sgd_steps']}", flush=True)
    print(f"  Communication Cost: {summary['communication_cost_gb']} GB", flush=True)
    print(f"  Training Time: {summary['training_time_hours']} hours", flush=True)
    print(f"  Convergence Score: {summary['projected_convergence_score']}", flush=True)
    print(f"  Model Parameters: {summary['model_parameter_count_millions']}M", flush=True)
    print(f"  Summary saved to {summary_path}", flush=True)
    print(f"[{_ts()}] Server shutting down.", flush=True)


if __name__ == "__main__":
    main()
