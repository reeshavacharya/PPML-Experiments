"""Plain FedAvg strategy for IID FL settings (ImageCAS's 5 synthetic shards).

Shares the checkpointing/resume/logging scaffolding of fl/server.py's
FedPIDAvgStrategy but drops the PID aggregation weighting and Poisson
institution exclusion — those exist specifically to handle FeTS's natural,
highly non-IID institution sizes, which don't apply to IID synthetic shards.
Aggregation itself is the parent class's standard example-count-weighted
average (flwr.server.strategy.FedAvg.aggregate_fit).
"""
import os
import time
from collections import OrderedDict
from datetime import datetime

import flwr as fl
import torch


def _ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def fit_metrics_aggregation_fn(results):
    if not results:
        return {}
    total = sum(n for n, _ in results)
    wloss = sum(n * m.get("train_loss", 0) for n, m in results)
    return {"train_loss": wloss / total if total else 0}


def evaluate_metrics_aggregation_fn(results):
    if not results:
        return {}
    total = sum(n for n, _ in results)
    wdice = sum(n * m.get("val_dice", 0) for n, m in results)
    return {"val_dice": wdice / total if total else 0}


class CASFedAvgStrategy(fl.server.strategy.FedAvg):
    def __init__(
        self,
        checkpoint_dir,
        model_fn,
        round_offset=0,
        best_dice=0.0,
        best_round=0,
        client_lr=None,
        client_wd=None,
        client_max_steps=0,
        resume_filename="fl_resume.pth",
        best_model_filename="swin_unetr_fl_best.pth",
        metric_key="val_dice",
        metric_label="dice",
        log_label="CASFedAvgStrategy: plain FedAvg (IID clients)",
        *args,
        **kwargs,
    ):
        """metric_key/metric_label let non-Dice tasks (e.g. Fed-ISIC2019's
        classification client, which reports "val_balanced_acc" instead of
        "val_dice") reuse this strategy unchanged. metric_key is the key each
        client's evaluate() metrics dict is read from; metric_label only
        controls the summary JSON's key names. Defaults reproduce the
        original CAS/Dice behavior exactly."""
        super().__init__(*args, **kwargs)
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self._model_fn = model_fn  # zero-arg callable returning a fresh model instance
        self._resume_path = os.path.join(checkpoint_dir, resume_filename)
        self._best_model_path = os.path.join(checkpoint_dir, best_model_filename)
        self._metric_key = metric_key
        self._metric_label = metric_label

        self._client_lr = client_lr
        self._client_wd = client_wd
        self._client_max_steps = client_max_steps

        self._round_offset = round_offset
        self._best_dice = best_dice
        self._best_round = best_round
        self._latest_ndarrays = None
        self._start_time = time.time()

        self._per_client_best_dice = {}
        self._rounds_to_best = {}

        self._total_sgd_steps = 0
        self._parallel_sgd_steps = 0
        self._total_comm_bytes = 0
        self._round_times = []
        self._round_start = None

        print(f"[{_ts()}] {log_label}", flush=True)

    def _actual_round(self, server_round):
        return server_round + self._round_offset

    def _inject_client_hps(self, configs):
        """Add lr/weight_decay/max_steps to FitIns config dicts when running in HPO mode."""
        if self._client_lr is None and self._client_wd is None and not self._client_max_steps:
            return configs
        updated = []
        for proxy, fit_ins in configs:
            new_cfg = dict(fit_ins.config)
            if self._client_lr is not None:
                new_cfg["lr"] = str(self._client_lr)
            if self._client_wd is not None:
                new_cfg["weight_decay"] = str(self._client_wd)
            if self._client_max_steps:
                new_cfg["max_steps"] = str(self._client_max_steps)
            updated.append((proxy, fl.common.FitIns(fit_ins.parameters, new_cfg)))
        return updated

    def configure_fit(self, server_round, parameters, client_manager):
        actual = self._actual_round(server_round)
        self._round_start = time.time()

        if actual == 1:
            time.sleep(30)

        available = client_manager.num_available()
        print(f"[{_ts()}] Round {actual} [FIT-CONFIG] — {available} client(s) available", flush=True)

        configs = super().configure_fit(server_round, parameters, client_manager)
        return self._inject_client_hps(configs)

    def aggregate_fit(self, server_round, results, failures):
        actual = self._actual_round(server_round)
        print(f"[{_ts()}] Round {actual} [FIT-AGG] — {len(results)} result(s), {len(failures)} failure(s)", flush=True)

        max_steps = 0
        for proxy, fit_res in results:
            client_id = fit_res.metrics.get("institution_id", "?")
            steps = int(fit_res.metrics.get("sgd_steps", 0))
            self._total_sgd_steps += steps
            max_steps = max(max_steps, steps)
            print(f"  client {proxy.cid} (id {client_id}): {fit_res.num_examples} examples, "
                  f"steps={steps}, loss={fit_res.metrics.get('train_loss', 0):.4f}", flush=True)
        self._parallel_sgd_steps += max_steps

        for failure in failures:
            print(f"  FAILURE: {failure}", flush=True)

        aggregated_params, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_params is not None:
            self._latest_ndarrays = fl.common.parameters_to_ndarrays(aggregated_params)
            model_bytes = sum(arr.nbytes for arr in self._latest_ndarrays)
            self._total_comm_bytes += model_bytes * len(results) * 2
            self._save_resume(actual)

        return aggregated_params, aggregated_metrics

    def configure_evaluate(self, server_round, parameters, client_manager):
        actual = self._actual_round(server_round)
        available = client_manager.num_available()
        print(f"[{_ts()}] Round {actual} [EVAL-CONFIG] — {available} client(s) available", flush=True)
        return super().configure_evaluate(server_round, parameters, client_manager)

    def aggregate_evaluate(self, server_round, results, failures):
        actual = self._actual_round(server_round)
        print(f"[{_ts()}] Round {actual} [EVAL-AGG] — {len(results)} result(s), {len(failures)} failure(s)", flush=True)

        for proxy, eval_res in results:
            client_id = eval_res.metrics.get("institution_id", "?")
            metric_val = eval_res.metrics.get(self._metric_key, 0)
            print(f"  client {proxy.cid} (id {client_id}): loss={eval_res.loss:.4f}, {self._metric_label}={metric_val:.4f}", flush=True)

            client_key = str(client_id)
            if metric_val > self._per_client_best_dice.get(client_key, 0):
                self._per_client_best_dice[client_key] = metric_val
                self._rounds_to_best[client_key] = actual

        for failure in failures:
            print(f"  FAILURE: {failure}", flush=True)

        aggregated = super().aggregate_evaluate(server_round, results, failures)

        if results:
            total = sum(eval_res.num_examples for _, eval_res in results)
            weighted = sum(eval_res.num_examples * eval_res.metrics.get(self._metric_key, 0) for _, eval_res in results)
            avg_metric = weighted / total if total > 0 else 0

            if avg_metric > self._best_dice:
                self._best_dice = avg_metric
                self._best_round = actual
                self._save_best(actual)
                self._save_resume(actual)
            else:
                print(f"[{_ts()}] Round {actual} {self._metric_label} {avg_metric:.4f} did not beat best {self._best_dice:.4f} (round {self._best_round})", flush=True)

        if self._round_start:
            self._round_times.append(time.time() - self._round_start)

        return aggregated

    def _save_resume(self, actual_round):
        if self._latest_ndarrays is None:
            return
        torch.save({
            "round": actual_round,
            "parameters": list(self._latest_ndarrays),
            "best_dice": self._best_dice,
            "best_round": self._best_round,
        }, self._resume_path)

    def _save_best(self, actual_round):
        if self._latest_ndarrays is None:
            return
        model = self._model_fn()
        params_dict = zip(model.state_dict().keys(), self._latest_ndarrays)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

        torch.save(model.state_dict(), self._best_model_path)
        print(f"[{_ts()}] Best model updated (round {actual_round}, {self._metric_label} {self._best_dice:.4f}) -> {self._best_model_path}", flush=True)

    def get_summary(self, total_rounds):
        elapsed = time.time() - self._start_time
        model = self._model_fn()
        param_count = sum(p.numel() for p in model.parameters())

        if self._round_times:
            round_dices = []
            running_best = 0.0
            for rt in self._round_times:
                if self._best_dice > running_best:
                    running_best = self._best_dice
                round_dices.append(running_best)
            numerator = sum(d * t for d, t in zip(round_dices, self._round_times))
            denominator = sum(self._round_times)
            convergence_score = numerator / denominator if denominator > 0 else 0
        else:
            convergence_score = 0

        return {
            "algorithm": "FedAvg",
            "total_rounds": total_rounds,
            "best_round": self._best_round,
            f"best_val_{self._metric_label}": round(self._best_dice, 4),
            "rounds_to_best_per_client": {k: v for k, v in sorted(self._rounds_to_best.items())},
            f"per_client_best_{self._metric_label}": {k: round(v, 4) for k, v in sorted(self._per_client_best_dice.items())},
            "total_sgd_steps": self._total_sgd_steps,
            "parallel_sgd_steps": self._parallel_sgd_steps,
            "communication_cost_bytes": self._total_comm_bytes,
            "communication_cost_gb": round(self._total_comm_bytes / (1024**3), 2),
            "training_time_seconds": round(elapsed, 1),
            "training_time_hours": round(elapsed / 3600, 2),
            "projected_convergence_score": round(convergence_score, 4),
            "model_parameter_count": param_count,
            "model_parameter_count_millions": round(param_count / 1e6, 2),
        }
