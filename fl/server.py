import argparse
import flwr as fl
import os
import torch
import sys
import platform
import numpy as np
from datetime import datetime
from collections import OrderedDict

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from models.swin_unetr import get_model

RESUME_PATH = os.path.join(project_root, "checkpoint", "fl_resume.pth")


def _ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def fit_metrics_aggregation_fn(results):
    if not results:
        return {}
    total_examples = sum([num_examples for num_examples, _ in results])
    weighted_train_loss = sum(
        [num_examples * metrics.get("train_loss", 0) for num_examples, metrics in results]
    )
    return {"train_loss": weighted_train_loss / total_examples}


def evaluate_metrics_aggregation_fn(results):
    if not results:
        return {}
    total_examples = sum([num_examples for num_examples, _ in results])
    weighted_val_dice = sum(
        [num_examples * metrics.get("val_dice", 0) for num_examples, metrics in results]
    )
    return {"val_dice": weighted_val_dice / total_examples}


class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(self, checkpoint_dir, round_offset=0, best_dice=0.0, best_round=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self._known_cids = set()
        self._best_dice = best_dice
        self._best_round = best_round
        self._latest_ndarrays = None
        self._round_offset = round_offset

    def _actual_round(self, server_round):
        return server_round + self._round_offset

    def _log_client_presence(self, client_manager, phase, server_round):
        available = client_manager.num_available()
        print(f"[{_ts()}] Round {self._actual_round(server_round)} [{phase}] — {available} client(s) available", flush=True)

    def configure_fit(self, server_round, parameters, client_manager):
        self._log_client_presence(client_manager, "FIT-CONFIG", server_round)
        configs = super().configure_fit(server_round, parameters, client_manager)
        current_cids = {proxy.cid for proxy, _ in configs}
        newly_joined = current_cids - self._known_cids
        disconnected = self._known_cids - current_cids
        for cid in sorted(newly_joined):
            print(f"[{_ts()}]   CONNECTED: client {cid}", flush=True)
        for cid in sorted(disconnected):
            print(f"[{_ts()}]   DISCONNECTED: client {cid}", flush=True)
        self._known_cids = current_cids
        return configs

    def aggregate_fit(self, server_round, results, failures):
        actual = self._actual_round(server_round)
        print(f"[{_ts()}] Round {actual} [FIT-AGG] — {len(results)} result(s), {len(failures)} failure(s)", flush=True)
        for proxy, fit_res in results:
            print(f"  client {proxy.cid}: {fit_res.num_examples} examples, metrics={fit_res.metrics}", flush=True)
        for failure in failures:
            print(f"  FAILURE: {failure}", flush=True)

        aggregated_weights, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_weights is not None:
            self._latest_ndarrays = fl.common.parameters_to_ndarrays(aggregated_weights)
            self._save_resume(actual)

        return aggregated_weights, aggregated_metrics

    def _save_resume(self, actual_round):
        if self._latest_ndarrays is None:
            return
        torch.save({
            "round": actual_round,
            "parameters": [arr for arr in self._latest_ndarrays],
            "best_dice": self._best_dice,
            "best_round": self._best_round,
        }, RESUME_PATH)
        print(f"[{_ts()}] Resume checkpoint saved (round {actual_round}) -> {RESUME_PATH}", flush=True)

    def _save_best(self, actual_round):
        if self._latest_ndarrays is None:
            return
        model = get_model()
        params_dict = zip(model.state_dict().keys(), self._latest_ndarrays)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

        save_path = os.path.join(self.checkpoint_dir, "swin_unetr_fl_best.pth")
        torch.save(model.state_dict(), save_path)
        print(f"[{_ts()}] Best model updated (round {actual_round}, dice {self._best_dice:.4f}) -> {save_path}", flush=True)

    def configure_evaluate(self, server_round, parameters, client_manager):
        self._log_client_presence(client_manager, "EVAL-CONFIG", server_round)
        return super().configure_evaluate(server_round, parameters, client_manager)

    def aggregate_evaluate(self, server_round, results, failures):
        actual = self._actual_round(server_round)
        print(f"[{_ts()}] Round {actual} [EVAL-AGG] — {len(results)} result(s), {len(failures)} failure(s)", flush=True)
        for proxy, eval_res in results:
            print(f"  client {proxy.cid}: loss={eval_res.loss:.4f}, metrics={eval_res.metrics}", flush=True)
        for failure in failures:
            print(f"  FAILURE: {failure}", flush=True)

        aggregated = super().aggregate_evaluate(server_round, results, failures)

        if results:
            total = sum(eval_res.num_examples for _, eval_res in results)
            weighted = sum(eval_res.num_examples * eval_res.metrics.get("val_dice", 0) for _, eval_res in results)
            avg_dice = weighted / total if total > 0 else 0

            if avg_dice > self._best_dice:
                self._best_dice = avg_dice
                self._best_round = actual
                self._save_best(actual)
                self._save_resume(actual)
            else:
                print(f"[{_ts()}] Round {actual} dice {avg_dice:.4f} did not beat best {self._best_dice:.4f} (round {self._best_round})", flush=True)

        return aggregated


def main():
    parser = argparse.ArgumentParser(description="Flower Server for FeTS 2022")
    parser.add_argument("--server_address", type=str, default="0.0.0.0:8080", help="Address to bind the server to")
    parser.add_argument("--rounds", type=int, default=100, help="Total number of FL rounds")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    args = parser.parse_args()

    checkpoint_dir = os.path.join(project_root, "checkpoint")
    round_offset = 0
    initial_parameters = None
    best_dice = 0.0
    best_round = 0

    if args.resume and os.path.exists(RESUME_PATH):
        print(f"[{_ts()}] Resuming from {RESUME_PATH}...", flush=True)
        ckpt = torch.load(RESUME_PATH, weights_only=False)
        round_offset = ckpt["round"]
        best_dice = ckpt["best_dice"]
        best_round = ckpt["best_round"]
        initial_parameters = fl.common.ndarrays_to_parameters(ckpt["parameters"])
        print(f"[{_ts()}]   Completed rounds: {round_offset}", flush=True)
        print(f"[{_ts()}]   Best dice so far: {best_dice:.4f} (round {best_round})", flush=True)
        print(f"[{_ts()}]   Remaining rounds: {args.rounds - round_offset}", flush=True)
    elif args.resume:
        print(f"[{_ts()}] --resume specified but no checkpoint found at {RESUME_PATH}. Starting fresh.", flush=True)

    remaining_rounds = args.rounds - round_offset
    if remaining_rounds <= 0:
        print(f"[{_ts()}] All {args.rounds} rounds already completed. Nothing to do.", flush=True)
        return

    print("=" * 60, flush=True)
    print(f"[{_ts()}] FL Server — {platform.node()}", flush=True)
    print(f"  Total rounds: {args.rounds} (starting from {round_offset + 1})", flush=True)
    print(f"  Bind address: {args.server_address}", flush=True)
    print(f"  Waiting for 3 clients to connect...", flush=True)
    print("=" * 60, flush=True)

    addr_file = os.path.join(project_root, "fl", "server_address.txt")
    with open(addr_file, "w") as f:
        f.write(platform.node())
    print(f"[{_ts()}] Server hostname written to {addr_file}", flush=True)

    strategy = SaveModelStrategy(
        checkpoint_dir=checkpoint_dir,
        round_offset=round_offset,
        best_dice=best_dice,
        best_round=best_round,
        initial_parameters=initial_parameters,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=3,
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
    )

    fl.server.start_server(
        server_address=args.server_address,
        config=fl.server.ServerConfig(num_rounds=remaining_rounds),
        strategy=strategy,
    )

    print(f"[{_ts()}] FL training complete. Server shutting down.", flush=True)


if __name__ == "__main__":
    main()
