import argparse
import json
import random
import flwr as fl
import os
import torch
import sys
import platform
import time
import numpy as np
from datetime import datetime
from collections import OrderedDict

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from models.swin_unetr import get_model
from utils.dataset_config import get_dataset_config, resolve_manifest_path
from fl.strategy_fedavg import CASFedAvgStrategy


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


class FedPIDAvgStrategy(fl.server.strategy.FedAvg):
    def __init__(
        self,
        checkpoint_dir,
        institution_sizes,
        model_fn,
        alpha=0.45,
        beta=0.45,
        gamma=0.1,
        cost_history_window=5,
        poisson_include_prob=0.1,
        round_offset=0,
        best_dice=0.0,
        best_round=0,
        pid_state=None,
        client_lr=None,
        client_wd=None,
        client_max_steps=0,
        resume_filename="fl_resume.pth",
        best_model_filename="swin_unetr_fl_best.pth",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self._model_fn = model_fn
        self._resume_path = os.path.join(checkpoint_dir, resume_filename)
        self._best_model_path = os.path.join(checkpoint_dir, best_model_filename)

        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma
        self._cost_history_window = cost_history_window
        self._institution_sizes = institution_sizes
        self._poisson_lambda = np.mean(list(institution_sizes.values()))
        self._poisson_threshold = 2 * self._poisson_lambda
        self._poisson_include_prob = poisson_include_prob
        self._outlier_institutions = {
            pid for pid, cnt in institution_sizes.items() if cnt > self._poisson_threshold
        }

        self._client_lr = client_lr
        self._client_wd = client_wd
        self._client_max_steps = client_max_steps

        self._round_offset = round_offset
        self._best_dice = best_dice
        self._best_round = best_round
        self._latest_ndarrays = None
        self._start_time = time.time()

        self._known_cids = set()
        self._cid_to_inst = {}
        self._last_fit_cids = set()
        self._last_fit_institutions = set()

        if pid_state:
            self._cost_history = pid_state.get("cost_history", {})
            self._prev_costs = pid_state.get("prev_costs", {})
            self._per_client_best_dice = pid_state.get("per_client_best_dice", {})
            self._rounds_to_best = pid_state.get("rounds_to_best", {})
        else:
            self._cost_history = {}
            self._prev_costs = {}
            self._per_client_best_dice = {}
            self._rounds_to_best = {}

        self._total_sgd_steps = 0
        self._parallel_sgd_steps = 0
        self._total_comm_bytes = 0
        self._round_times = []
        self._round_start = None

        print(f"[{_ts()}] FedPIDAvg: α={alpha}, β={beta}, γ={gamma}", flush=True)
        print(f"[{_ts()}] Poisson λ={self._poisson_lambda:.1f}, threshold={self._poisson_threshold:.1f}", flush=True)
        print(f"[{_ts()}] Outlier institutions (excluded ~{int((1-poisson_include_prob)*100)}% of rounds): {sorted(self._outlier_institutions)}", flush=True)

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

    # ── Client Selection (Poisson) ──

    def configure_fit(self, server_round, parameters, client_manager):
        actual = self._actual_round(server_round)
        self._round_start = time.time()

        # On the very first round, allow a grace period for all gRPC connections
        # to stabilize after the min_available_clients threshold is crossed.
        if actual == 1:
            time.sleep(30)

        available = client_manager.num_available()
        print(f"[{_ts()}] Round {actual} [FIT-CONFIG] — {available} client(s) available", flush=True)

        configs = super().configure_fit(server_round, parameters, client_manager)

        if not self._cid_to_inst:
            # First round: include all to learn CID→institution mapping
            self._last_round_included_outliers = True
            current_cids = {proxy.cid for proxy, _ in configs}
            newly_joined = current_cids - self._known_cids
            for cid in sorted(newly_joined):
                print(f"[{_ts()}]   CONNECTED: client {cid}", flush=True)
            self._known_cids = current_cids
            return self._inject_client_hps(configs)

        include_outliers = random.random() < self._poisson_include_prob
        self._last_round_included_outliers = include_outliers
        if include_outliers:
            print(f"[{_ts()}]   Including ALL institutions this round (Poisson draw)", flush=True)
            return self._inject_client_hps(configs)

        filtered = [
            (proxy, ins) for proxy, ins in configs
            if self._cid_to_inst.get(proxy.cid) not in self._outlier_institutions
        ]
        excluded = len(configs) - len(filtered)
        if excluded:
            print(f"[{_ts()}]   Excluded {excluded} outlier institution(s) (Poisson selection)", flush=True)

        current_cids = {proxy.cid for proxy, _ in filtered}
        newly_joined = current_cids - self._known_cids
        disconnected = self._known_cids - current_cids
        for cid in sorted(newly_joined):
            print(f"[{_ts()}]   CONNECTED: client {cid}", flush=True)
        for cid in sorted(disconnected):
            inst = self._cid_to_inst.get(cid, "?")
            if inst in self._outlier_institutions:
                print(f"[{_ts()}]   EXCLUDED (outlier): client {cid} (institution {inst})", flush=True)
            else:
                print(f"[{_ts()}]   DISCONNECTED: client {cid}", flush=True)
        self._known_cids = current_cids

        return self._inject_client_hps(filtered)

    # ── PID Aggregation ──

    def aggregate_fit(self, server_round, results, failures):
        actual = self._actual_round(server_round)
        print(f"[{_ts()}] Round {actual} [FIT-AGG] — {len(results)} result(s), {len(failures)} failure(s)", flush=True)

        self._last_fit_cids = set()
        self._last_fit_institutions = set()
        max_steps = 0
        for proxy, fit_res in results:
            inst_id = int(fit_res.metrics.get("institution_id", -1))
            steps = int(fit_res.metrics.get("sgd_steps", 0))
            self._cid_to_inst[proxy.cid] = inst_id
            self._last_fit_cids.add(proxy.cid)
            self._last_fit_institutions.add(inst_id)
            self._total_sgd_steps += steps
            max_steps = max(max_steps, steps)
            print(f"  client {proxy.cid} (inst {inst_id}): {fit_res.num_examples} examples, steps={steps}, loss={fit_res.metrics.get('train_loss', 0):.4f}", flush=True)
        self._parallel_sgd_steps += max_steps

        for failure in failures:
            print(f"  FAILURE: {failure}", flush=True)

        if not results:
            return None, {}

        # First pass: extract costs and institution IDs (lightweight, no params)
        client_costs = []
        client_inst_ids = []
        for proxy, fit_res in results:
            inst_id = self._cid_to_inst[proxy.cid]
            cost = float(fit_res.metrics.get("train_loss", 0))
            client_costs.append(cost)
            client_inst_ids.append(inst_id)

        n_clients = len(results)

        # P term: dataset size proportion
        s_vals = np.array([self._institution_sizes.get(iid, 1) for iid in client_inst_ids], dtype=float)
        S = s_vals.sum()
        p_weights = s_vals / S if S > 0 else np.ones(n_clients) / n_clients

        # D term: cost decrease from previous round
        k_vals = np.zeros(n_clients)
        has_d = False
        for j, iid in enumerate(client_inst_ids):
            iid_key = str(iid)
            if iid_key in self._prev_costs:
                k_vals[j] = self._prev_costs[iid_key] - client_costs[j]
                has_d = True

        K = k_vals.sum()
        if has_d and abs(K) > 1e-8:
            d_weights = k_vals / K
        else:
            d_weights = np.ones(n_clients) / n_clients

        # I term: integral of costs over last N rounds
        m_vals = np.zeros(n_clients)
        has_i = False
        for j, iid in enumerate(client_inst_ids):
            iid_key = str(iid)
            history = self._cost_history.get(iid_key, [])
            if history:
                m_vals[j] = sum(history)
                has_i = True

        I_sum = m_vals.sum()
        if has_i and abs(I_sum) > 1e-8:
            i_weights = m_vals / I_sum
        else:
            i_weights = np.ones(n_clients) / n_clients

        # Combined PID weights
        if not has_d and not has_i:
            weights = p_weights
            print(f"[{_ts()}]   PID: Round 1 — using P-only weights", flush=True)
        else:
            weights = self._alpha * p_weights + self._beta * d_weights + self._gamma * i_weights

        # Normalize
        w_sum = weights.sum()
        if w_sum > 0:
            weights = weights / w_sum
        else:
            weights = np.ones(n_clients) / n_clients

        print(f"[{_ts()}]   PID weights: " + ", ".join(
            f"inst {client_inst_ids[j]}={weights[j]:.4f}" for j in range(n_clients)
        ), flush=True)

        # Streaming weighted aggregation: process one client at a time to limit memory
        # Only one client's parameters are in memory at any given time
        aggregated = None
        ref_dtypes = None
        for j, (proxy, fit_res) in enumerate(results):
            params_j = fl.common.parameters_to_ndarrays(fit_res.parameters)
            if aggregated is None:
                ref_dtypes = [layer.dtype for layer in params_j]
                aggregated = [np.zeros(layer.shape, dtype=np.float64) for layer in params_j]
            for layer_idx in range(len(aggregated)):
                aggregated[layer_idx] += weights[j] * params_j[layer_idx].astype(np.float64)
            del params_j

        # Cast back to original dtypes
        for layer_idx in range(len(aggregated)):
            aggregated[layer_idx] = aggregated[layer_idx].astype(ref_dtypes[layer_idx])

        # Update PID state
        for j, iid in enumerate(client_inst_ids):
            iid_key = str(iid)
            self._prev_costs[iid_key] = client_costs[j]
            history = self._cost_history.get(iid_key, [])
            history.append(client_costs[j])
            if len(history) > self._cost_history_window:
                history = history[-self._cost_history_window:]
            self._cost_history[iid_key] = history

        self._latest_ndarrays = aggregated
        aggregated_params = fl.common.ndarrays_to_parameters(aggregated)

        # Communication cost
        model_bytes = sum(arr.nbytes for arr in aggregated)
        self._total_comm_bytes += model_bytes * n_clients * 2

        self._save_resume(actual)

        aggregated_metrics = fit_metrics_aggregation_fn(
            [(fit_res.num_examples, fit_res.metrics) for _, fit_res in results]
        )

        return aggregated_params, aggregated_metrics

    # ── Evaluation ──

    def configure_evaluate(self, server_round, parameters, client_manager):
        actual = self._actual_round(server_round)
        available = client_manager.num_available()
        print(f"[{_ts()}] Round {actual} [EVAL-CONFIG] — {available} client(s) available", flush=True)

        configs = super().configure_evaluate(server_round, parameters, client_manager)

        if not self._cid_to_inst:
            return configs

        # Only evaluate institutions that successfully completed fit this round.
        # Filter by institution_id (not CID) because gRPC clients get a new UUID
        # on every reconnect — a client that rejoins between fit and eval would
        # have a new CID not in _last_fit_cids even though it trained this round.
        if self._last_fit_institutions:
            configs = [
                (proxy, ins) for proxy, ins in configs
                if self._cid_to_inst.get(proxy.cid) in self._last_fit_institutions
            ]

        # Exclude outlier institutions from evaluation in non-outlier rounds
        include_outliers = self._last_round_included_outliers if hasattr(self, '_last_round_included_outliers') else True
        if not include_outliers:
            configs = [
                (proxy, ins) for proxy, ins in configs
                if self._cid_to_inst.get(proxy.cid) not in self._outlier_institutions
            ]

        print(f"[{_ts()}]   Eval: sending to {len(configs)} client(s)", flush=True)
        return configs

    def aggregate_evaluate(self, server_round, results, failures):
        actual = self._actual_round(server_round)
        print(f"[{_ts()}] Round {actual} [EVAL-AGG] — {len(results)} result(s), {len(failures)} failure(s)", flush=True)

        for proxy, eval_res in results:
            inst = self._cid_to_inst.get(proxy.cid, "?")
            dice = eval_res.metrics.get("val_dice", 0)
            print(f"  client {proxy.cid} (inst {inst}): loss={eval_res.loss:.4f}, dice={dice:.4f}", flush=True)

            inst_key = str(inst)
            if dice > self._per_client_best_dice.get(inst_key, 0):
                self._per_client_best_dice[inst_key] = dice
                self._rounds_to_best[inst_key] = actual

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

        if self._round_start:
            self._round_times.append(time.time() - self._round_start)

        return aggregated

    # ── Checkpointing ──

    def _save_resume(self, actual_round):
        if self._latest_ndarrays is None:
            return
        torch.save({
            "round": actual_round,
            "parameters": list(self._latest_ndarrays),
            "best_dice": self._best_dice,
            "best_round": self._best_round,
            "pid_state": {
                "cost_history": self._cost_history,
                "prev_costs": self._prev_costs,
                "per_client_best_dice": self._per_client_best_dice,
                "rounds_to_best": self._rounds_to_best,
            },
        }, self._resume_path)

    def _save_best(self, actual_round):
        if self._latest_ndarrays is None:
            return
        model = self._model_fn()
        params_dict = zip(model.state_dict().keys(), self._latest_ndarrays)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

        torch.save(model.state_dict(), self._best_model_path)
        print(f"[{_ts()}] Best model updated (round {actual_round}, dice {self._best_dice:.4f}) -> {self._best_model_path}", flush=True)

    def get_summary(self, total_rounds):
        elapsed = time.time() - self._start_time
        model = self._model_fn()
        param_count = sum(p.numel() for p in model.parameters())

        # Projected Convergence Score: Σ(bestDSC_i × round_time_i) / Σ(round_time_i)
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
            "algorithm": "FedPIDAvg",
            "alpha": self._alpha,
            "beta": self._beta,
            "gamma": self._gamma,
            "total_rounds": total_rounds,
            "best_round": self._best_round,
            "best_val_dice": round(self._best_dice, 4),
            "rounds_to_best_per_client": {k: v for k, v in sorted(self._rounds_to_best.items())},
            "per_client_best_dice": {k: round(v, 4) for k, v in sorted(self._per_client_best_dice.items())},
            "total_sgd_steps": self._total_sgd_steps,
            "parallel_sgd_steps": self._parallel_sgd_steps,
            "communication_cost_bytes": self._total_comm_bytes,
            "communication_cost_gb": round(self._total_comm_bytes / (1024**3), 2),
            "training_time_seconds": round(elapsed, 1),
            "training_time_hours": round(elapsed / 3600, 2),
            "projected_convergence_score": round(convergence_score, 4),
            "model_parameter_count": param_count,
            "model_parameter_count_millions": round(param_count / 1e6, 2),
            "poisson_lambda": round(self._poisson_lambda, 1),
            "outlier_institutions": sorted(self._outlier_institutions),
        }


def main():
    parser = argparse.ArgumentParser(description="Flower Server for SwinUNETR FL experiments")
    parser.add_argument("--dataset", type=str, default="fets", choices=["fets", "cas"])
    parser.add_argument("--strategy", type=str, default=None, choices=["fedpidavg", "fedavg"],
                        help="Default: fedpidavg for fets, fedavg for cas (see utils/dataset_config.py)")
    parser.add_argument("--server_address", type=str, default="0.0.0.0:8080")
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--num_clients", type=int, default=None, help="Default: per-dataset config")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to dataset (default: per-dataset config)")
    parser.add_argument("--alpha", type=float, default=0.45, help="FedPIDAvg only")
    parser.add_argument("--beta", type=float, default=0.45, help="FedPIDAvg only")
    parser.add_argument("--gamma", type=float, default=0.1, help="FedPIDAvg only")
    parser.add_argument("--resume", action="store_true")
    # HPO arguments: override PID params + client optimizer HPs from a JSON file
    parser.add_argument("--hp_config", type=str, default=None,
                        help="Path to JSON file with HPO config (overrides alpha/beta/gamma/etc.)")
    parser.add_argument("--result_path", type=str, default=None,
                        help="Path to write trial result JSON {mean_dice, best_round} for HPO coordinator")
    parser.add_argument("--lr", type=float, default=None,
                        help="Client learning rate to broadcast via configure_fit (HPO mode)")
    parser.add_argument("--weight_decay", type=float, default=None,
                        help="Client weight decay to broadcast via configure_fit (HPO mode)")
    parser.add_argument("--hpo_max_steps", type=int, default=0,
                        help="Limit client training to N gradient steps per round (0=unlimited; HPO mode)")
    parser.add_argument("--seed", type=int, default=42, help="Seeds Python's random (used by FedPIDAvg's Poisson draw)")
    args = parser.parse_args()

    random.seed(args.seed)

    cfg = get_dataset_config(args.dataset)
    strategy_name = args.strategy or cfg["default_strategy"]
    num_clients = args.num_clients or cfg["default_num_clients"]
    data_dir = args.data_dir if args.data_dir is not None else cfg["default_data_dir"]
    model_fn = lambda: get_model(in_channels=cfg["in_channels"], out_channels=cfg["out_channels"], feature_size=cfg["feature_size"])

    suffix = "" if args.dataset == "fets" else f"_{args.dataset}"
    seed_suffix = "" if args.seed == 42 else f"_seed{args.seed}"
    resume_filename = f"fl_resume{suffix}{seed_suffix}.pth"
    best_model_filename = f"swin_unetr_fl{suffix}{seed_suffix}_best.pth"
    resume_path = os.path.join(project_root, "checkpoint", resume_filename)

    # --hp_config overrides individual CLI args when present
    if args.hp_config:
        with open(args.hp_config) as f:
            hp = json.load(f)
        args.alpha = hp.get("alpha", args.alpha)
        args.beta = hp.get("beta", args.beta)
        args.gamma = hp.get("gamma", args.gamma)
        args.lr = hp.get("lr", args.lr)
        args.weight_decay = hp.get("weight_decay", args.weight_decay)
        cost_history_window = int(hp.get("cost_history_window", 5))
        poisson_include_prob = float(hp.get("poisson_include_prob", 0.1))
        print(f"[{_ts()}] Loaded HP config from {args.hp_config}: {hp}", flush=True)
    else:
        cost_history_window = 5
        poisson_include_prob = 0.1

    checkpoint_dir = os.path.join(project_root, "checkpoint")

    round_offset = 0
    initial_parameters = None
    best_dice = 0.0
    best_round = 0
    pid_state = None

    if args.resume and os.path.exists(resume_path):
        print(f"[{_ts()}] Resuming from {resume_path}...", flush=True)
        ckpt = torch.load(resume_path, weights_only=False)
        round_offset = ckpt["round"]
        best_dice = ckpt["best_dice"]
        best_round = ckpt["best_round"]
        pid_state = ckpt.get("pid_state")
        initial_parameters = fl.common.ndarrays_to_parameters(ckpt["parameters"])
        print(f"[{_ts()}]   Completed rounds: {round_offset}", flush=True)
        print(f"[{_ts()}]   Best dice: {best_dice:.4f} (round {best_round})", flush=True)
    elif args.resume:
        print(f"[{_ts()}] --resume but no checkpoint found. Starting fresh.", flush=True)

    remaining_rounds = args.rounds - round_offset
    if remaining_rounds <= 0:
        print(f"[{_ts()}] All {args.rounds} rounds completed.", flush=True)
        return

    print("=" * 60, flush=True)
    print(f"[{_ts()}] {strategy_name.upper()} Server ({args.dataset}) — {platform.node()}", flush=True)
    print(f"  Rounds: {args.rounds} (starting from {round_offset + 1})", flush=True)
    print(f"  Clients: {num_clients}", flush=True)
    print(f"  Bind: {args.server_address}", flush=True)
    print("=" * 60, flush=True)

    _addr_suffix = "" if args.dataset == "fets" else f"_{args.dataset}"
    addr_file = os.path.join(project_root, "fl", f"server_address{_addr_suffix}.txt")
    with open(addr_file, "w") as f:
        f.write(platform.node())

    common_kwargs = dict(
        checkpoint_dir=checkpoint_dir,
        model_fn=model_fn,
        round_offset=round_offset,
        best_dice=best_dice,
        best_round=best_round,
        client_lr=args.lr,
        client_wd=args.weight_decay,
        client_max_steps=args.hpo_max_steps,
        resume_filename=resume_filename,
        best_model_filename=best_model_filename,
        initial_parameters=initial_parameters,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        # 21 = 23 institutions minus the 2 fixed FedPIDAvg Poisson outliers
        # (see outlier_institutions below). CAS has no outlier-exclusion
        # concept, so its floor is simply all 5 of its clients — a partial
        # round (e.g. 1 of 5 institutions) produces an HPO/training signal
        # that isn't representative of the federated setting and is treated
        # as unacceptable rather than a usable approximation.
        min_fit_clients=21 if args.dataset == "fets" else (5 if args.dataset == "cas" else 1),
        min_evaluate_clients=1,
        # NOT the same as num_clients: FedAvg.num_fit_clients() returns this
        # value (not min_fit_clients) as the min_num_clients passed to
        # client_manager.sample() on *every* round, which calls wait_for()
        # with a 24h default timeout. Setting this to num_clients forces
        # every round to wait for all institutions to be simultaneously
        # connected — impossible under staggered SLURM scheduling, and the
        # cause of multi-hour deadlocks at "Round N [FIT-CONFIG]" with no
        # further progress. min_fit_clients above already enforces the real
        # per-dataset quorum (21 for fets, 5 for cas); _cid_to_inst is
        # populated incrementally in aggregate_fit, so late joiners before
        # that quorum is hit are handled correctly without this.
        min_available_clients=1,
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
    )

    if strategy_name == "fedpidavg":
        # FedPIDAvg needs natural institution sizes to weight aggregation and
        # decide Poisson-excluded outliers — meaningful for FeTS's non-IID
        # institutions, not applicable to CAS's synthetic IID shards.
        from data_loader.fets_dataset import get_institution_sizes
        abs_data_dir, csv_path = resolve_manifest_path(project_root, data_dir, cfg)
        institution_sizes = get_institution_sizes(csv_path)
        print(f"[{_ts()}] Loaded {len(institution_sizes)} institution sizes", flush=True)

        strategy = FedPIDAvgStrategy(
            institution_sizes=institution_sizes,
            alpha=args.alpha,
            beta=args.beta,
            gamma=args.gamma,
            cost_history_window=cost_history_window,
            poisson_include_prob=poisson_include_prob,
            pid_state=pid_state,
            **common_kwargs,
        )
    else:
        strategy = CASFedAvgStrategy(**common_kwargs)

    fl.server.start_server(
        server_address=args.server_address,
        config=fl.server.ServerConfig(num_rounds=remaining_rounds, round_timeout=3600),
        strategy=strategy,
    )

    summary = strategy.get_summary(args.rounds)
    summary_path = os.path.join(project_root, "fl", f"training_summary{suffix}{seed_suffix}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    if args.result_path:
        os.makedirs(os.path.dirname(os.path.abspath(args.result_path)), exist_ok=True)
        with open(args.result_path, "w") as f:
            json.dump({"mean_dice": summary["best_val_dice"], "best_round": summary["best_round"]}, f)
        print(f"[{_ts()}] HPO trial result written to {args.result_path}", flush=True)

    print(f"\n[{_ts()}] {summary['algorithm']} Training Summary", flush=True)
    print(f"  Best Val Dice: {summary['best_val_dice']} (round {summary['best_round']})", flush=True)
    print(f"  Total SGD Steps: {summary['total_sgd_steps']}", flush=True)
    print(f"  Communication Cost: {summary['communication_cost_gb']} GB", flush=True)
    print(f"  Training Time: {summary['training_time_hours']} hours", flush=True)
    print(f"  Convergence Score: {summary['projected_convergence_score']}", flush=True)
    print(f"  Model Parameters: {summary['model_parameter_count_millions']}M", flush=True)
    print(f"  Summary saved to {summary_path}", flush=True)
    print(f"[{_ts()}] Server shutting down.", flush=True)


if __name__ == "__main__":
    main()
