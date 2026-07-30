import logging
import os
import time
from typing import Dict, Any

import torch
import torch.distributed as dist

from utils.utils import sim_mm_top_k, sim_hard_top_k, sim_soft_top_k, calc_auc_gpu, calc_gauc_gpu, write_info_to_file, _confusion_matrix_at_thresholds, calc_auroc_gpu
from utils.horizon import build_eligible_mask, overlap_summary
from utils.hash_retrieval import (
    RandomProjectionHash,
    hamming_topk,
    should_time_retrieval,
    target_history_cosine,
    topk_recall,
)

def masked_relevance_topk(score, valid, keep_top):
    valid = valid.to(device=score.device, dtype=torch.bool)
    score_order = torch.argsort(score, dim=1, descending=True, stable=True)
    ordered_valid = valid.gather(1, score_order)
    valid_order = torch.argsort(
        ordered_valid.to(torch.uint8), dim=1, descending=True, stable=True
    )
    indices = score_order.gather(1, valid_order)[:, :keep_top]
    invalid = ~valid.gather(1, indices)
    return indices, invalid

def cp_relevance_topk(
    attention,
    query,
    fact,
    valid,
    mm_cosine=None,
    keep_top=50,
):
    raw_score = attention.raw_relevance_logits(
        query, fact, mask=valid, mm_cosine=mm_cosine
    )
    score = raw_score.squeeze(-1).mean(dim=1)
    return masked_relevance_topk(score, valid, keep_top)

def prepare_checkpoint_dir(path):
    os.makedirs(path, exist_ok=True)

def synchronize_after_checkpoint():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

FEATURE_BLOCKS = {
    "ad": ["205", "206", "213", "214", "205_c"],
    "user": ["129_1", "130_1", "130_2", "130_3", "130_4", "130_5"],
    "uni_seq_fn": ["150_2_180", "151_2_180", "150_2_180_c"],
    "short_seq_fn": ["150_1_180", "151_1_180", "150_1_180_c"]
}

def transfer_embedding_to_blocks(emb_dict, feature_block):
    result_embs = {}
    for block_name, block_features in feature_block.items():
        result_embs[block_name] = []
        for fea in block_features:
            if fea in ["205_c", "150_2_180_c", "150_1_180_c"]:
                result_embs[block_name].append(emb_dict[fea].detach())
            else:
                result_embs[block_name].append(emb_dict[fea])
    return result_embs

class Trainer:
    def __init__(
        self,
        dense_model,
        sparse_model,
        dense_opt,
        sparse_opt,
        train_dataloader,
        test_dataloader,
        args,
        device,
        rank,
        world_size,
        keep_top=50,
        logger=None      
    ):
        self.dense_model = dense_model
        self.sparse_model = sparse_model
        self.dense_opt = dense_opt
        self.sparse_opt = sparse_opt
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.args = args

        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.keep_top = keep_top
        self.logger = logger

        self.is_main_process = (self.rank == 0)

        # threadholds for auroc
        self.num_thresholds = 200
        kepsilon = 1e-7
        thresholds = [
            (i + 1) * 1.0 / (self.num_thresholds - 1) for i in range(self.num_thresholds - 2)
        ]
        self.thresholds = [0.0 - kepsilon] + thresholds + [1.0 + kepsilon]
        self.thresholds = torch.tensor(self.thresholds).to(device)

        # steps for log
        self._epoch_index = 0
        self._batch_index = 0
        self._total_steps = 0
        self.eval_diagnostics = None
        self.hash_retriever = None
        if self.args["method"] == "eta-cp-muse":
            self.hash_retriever = RandomProjectionHash(
                input_dim=2 * self.args["embedding_dim"],
                bits=self.args["hash_bits"],
                seed=self.args["hash_seed"],
            ).to(self.device)

        # prepare id_count, assosiated with update_count() function
        # L = len(self.sparse_model.module.feature_maps["150_2_180"])
        # self.id_counts = torch.zeros(L+1, dtype=torch.long, device=device)
        # self.init_id_count()

    def fit(self):
        if self.is_main_process:
            logging.info("Start training...")
        self._total_steps = 0
        for epoch in range(self.args["epochs"]):
            self._epoch_index = epoch
            self.train_epoch()
        
        if "save_ckpt" in self.args and self.args["save_ckpt"]:
            if self.rank == 0:
                self.save_model()
            synchronize_after_checkpoint()

        self.eval()

    def train_epoch(self):
        self._batch_index = 0
        train_loss = 0
        
        self.init_metric()

        if hasattr(self.train_dataloader.dataset, 'set_epoch'):
            self.train_dataloader.dataset.set_epoch(self._epoch_index)

        self.last_log_time = time.time()

        for batch_index, batch_data in enumerate(self.train_dataloader):
            self._batch_index = batch_index
            loss, metrics = self.forward_step(batch_data, mode="train")
            loss_reduced = self._reduce_tensor(loss.detach())
            train_loss = (train_loss * batch_index + loss_reduced.item()) / (batch_index + 1)
            auc, gauc = self.update_metric(metrics)
            
            if self._batch_index % 100 == 0:
                self.log_metric(auc, gauc, loss_reduced.item(), train_loss, "Train")

            self._total_steps += 1
            max_train_steps = self.args.get("max_train_steps")
            if max_train_steps and batch_index + 1 >= max_train_steps:
                break
        
        # log at last train step
        self.log_metric(auc, gauc, loss_reduced.item(), train_loss, "Train")
    
    def forward_step(self, batch, mode="train"):
        if mode == "train":
            self.dense_model.train()
            self.sparse_model.train()
        else:
            self.dense_model.eval()
            self.sparse_model.eval()

        for key, item in batch.items():
            batch[key] = item.to(self.device)
        label = batch["label"]
        uids = batch["129_1"]
        del batch["label"]

        # step 1: transform batch
        batch = self.transform_batch(batch)

        full_seq_embs = None
        if self.args.get("longer_variant"):
            full_seq_embs = self.prepare_full_seq_embs(batch)

        # step 2: apply GSU
        batch, batch_embs = self.apply_general_search(batch, keep_top=self.keep_top)

        # count how many times each id is trained
        # if mode == "train":
        #     # remapped_ids = self.sparse_model.module.lookup("150_2_180", batch["150_2_180"])
        #     # remapped_ids = self.sparse_model.module.lookup("205", batch["205"])
        #     remapped_ids = self.sparse_model.module.lookup("150_1_180", batch["150_1_180"])
        #     self.update_count(remapped_ids)

        # step 3: feature to embedding
        batch_embs.update(self.sparse_model(batch))
        emb_block = transfer_embedding_to_blocks(batch_embs, FEATURE_BLOCKS)

        # step 4: forward pass
        loss, prop = self.dense_model(
            user_embs=emb_block["user"],
            ad_embs=emb_block["ad"],
            uni_seq_embs=emb_block["uni_seq_fn"],
            short_seq_fn=emb_block["short_seq_fn"],
            label=label,
            full_seq_embs=full_seq_embs,
        )

        if mode == "eval" and self.eval_diagnostics is not None:
            dense_model = self.dense_model.module if hasattr(self.dense_model, "module") else self.dense_model
            gate = getattr(dense_model, "last_horizon_gate", None)
            if gate is not None:
                gate = gate.detach().double()
                self.eval_diagnostics[3] += gate.sum()
                self.eval_diagnostics[4] += (gate * gate).sum()
                self.eval_diagnostics[5] += gate.numel()

        accum_steps = 1
        loss = loss / accum_steps

        # step 5: backward pass
        if mode == "train":
            loss.backward()
            if (self._batch_index + 1) % accum_steps == 0:
                self.dense_opt.step()
                self.sparse_opt.step()
                self.dense_opt.zero_grad()
                self.sparse_opt.zero_grad()

        # step 6: compute metrics
        metrics = dict()
        metrics["auc"], metrics["P"], metrics["N"], metrics["wins"] = calc_auc_gpu(label, prop)
        metrics["gauc"], metrics["weighted_auc_sum"], metrics["impression_counts"] = calc_gauc_gpu(label, prop, uids)
        metrics["tp"], metrics["fp"], metrics["tn"], metrics["fn"] = _confusion_matrix_at_thresholds(label, prop, self.thresholds)
        return loss * accum_steps, metrics

    @torch.no_grad()
    def prepare_full_seq_embs(self, batch):
        feature_names = [
            "205",
            "206",
            "205_c",
            "150_2_180",
            "151_2_180",
            "150_2_180_c",
        ]
        embeddings = self.sparse_model(
            {name: batch[name] for name in feature_names}
        )
        return {
            "target_item": embeddings["205"],
            "target_category": embeddings["206"],
            "target_scl": embeddings["205_c"],
            "history_item": embeddings["150_2_180"],
            "history_category": embeddings["151_2_180"],
            "history_scl": embeddings["150_2_180_c"],
            "valid_mask": batch["150_2_180"] != 0,
        }
    
    def transform_batch(self, batch):
        # form
        short_window = self.args.get("short_window", 50)
        short_seq_fn_id = batch["150_2_180"][:, -short_window:]
        short_seq_fn_cate = batch["151_2_180"][:, -short_window:]
        batch["150_1_180"] = short_seq_fn_id
        batch["151_1_180"] = short_seq_fn_cate

        # form keys for scl_embedding
        batch["205_c"] = batch["205"]
        batch["150_2_180_c"] = batch["150_2_180"]
        batch["150_1_180_c"] = batch["150_1_180"]
        return batch

    @torch.no_grad()
    def eval(self):
        if self.is_main_process:
            logging.info("Start evaluation...")

        self._batch_index = 0

        eval_loss = 0
        self.init_metric()
        self.eval_diagnostics = torch.zeros(13, dtype=torch.float64, device=self.device)

        if hasattr(self.test_dataloader.dataset, 'set_epoch'):
            self.test_dataloader.dataset.set_epoch(0)

        self.last_log_time = time.time()

        for batch_index, batch_data in enumerate(self.test_dataloader):
            self._batch_index = batch_index
            loss, metrics = self.forward_step(batch_data, mode="eval")

            loss_reduced = self._reduce_tensor(loss.detach())
            eval_loss = (eval_loss * batch_index + loss_reduced.item()) / (batch_index + 1)

            auc, gauc = self.update_metric(metrics)

            if self._batch_index % 100 == 0:
                self.log_metric(auc, gauc, loss_reduced.item(), eval_loss, "Eval")
            
            self._total_steps += 1
            max_eval_steps = self.args.get("max_eval_steps")
            if max_eval_steps and batch_index + 1 >= max_eval_steps:
                break
        
        # log at last eval step
        self.log_metric(auc, gauc, loss_reduced.item(), eval_loss, "Eval")
        self.log_eval_diagnostics()

    @torch.no_grad()
    def apply_general_search(self, batch, keep_top=50):
        top_k_embs = dict()
        if self.args["method"] in ["cp-muse", "eta-cp-muse"]:
            retrieval_fn = [
                "205",
                "206",
                "205_c",
                "150_2_180",
                "151_2_180",
                "150_2_180_c",
            ]
            retrieval_embs = self.sparse_model(
                {fn: batch[fn] for fn in retrieval_fn}
            )

            query = torch.cat(
                [retrieval_embs["205"], retrieval_embs["206"]], dim=-1
            ).unsqueeze(1)
            fact = torch.cat(
                [
                    retrieval_embs["150_2_180"],
                    retrieval_embs["151_2_180"],
                ],
                dim=-1,
            )
            target_content_emb = retrieval_embs["205_c"].unsqueeze(1)
            uni_seq_content_emb = retrieval_embs["150_2_180_c"]

            valid_mask = batch["150_2_180"] != 0
            eligible_mask = build_eligible_mask(
                valid_mask,
                short_window=self.args.get("short_window", 50),
                policy=self.args.get("long_history_policy", "overlap"),
            )
            dense_model = (
                self.dense_model.module
                if hasattr(self.dense_model, "module")
                else self.dense_model
            )
            if self.args["method"] == "eta-cp-muse":
                collect_eta_diagnostics = (
                    self.eval_diagnostics is not None
                    and self.args.get("collect_retrieval_diagnostics", False)
                )
                collect_eta_timing = (
                    collect_eta_diagnostics
                    and should_time_retrieval(
                        self._batch_index,
                        self.args.get("retrieval_timing_warmup_steps", 0),
                    )
                )

                def hash_search():
                    query_code = self.hash_retriever(query)
                    history_code = self.hash_retriever(fact)
                    return hamming_topk(
                        query_code,
                        history_code,
                        eligible_mask,
                        keep_top=self.args["hash_shortlist"],
                    )

                if collect_eta_timing:
                    hash_start, hash_end = self._start_retrieval_timer()
                shortlist_indices, shortlist_invalid = hash_search()
                if collect_eta_timing:
                    hash_ms = self._stop_retrieval_timer(hash_start, hash_end)

                shortlist_batch = torch.arange(
                    query.shape[0], device=query.device
                ).unsqueeze(1).expand_as(shortlist_indices)
                if collect_eta_timing:
                    rerank_start, rerank_end = self._start_retrieval_timer()
                shortlist_fact = fact[shortlist_batch, shortlist_indices]
                shortlist_mm_cosine = target_history_cosine(
                    target_content_emb,
                    uni_seq_content_emb,
                    indices=shortlist_indices,
                )
                shortlist_valid = ~shortlist_invalid
                rerank_indices, invalid_mask = cp_relevance_topk(
                    dense_model.uni_att_v2,
                    query,
                    shortlist_fact,
                    shortlist_valid,
                    mm_cosine=[shortlist_mm_cosine],
                    keep_top=keep_top,
                )
                top_k_indices = shortlist_indices.gather(1, rerank_indices)
                invalid_mask |= shortlist_invalid.gather(1, rerank_indices)
                if collect_eta_diagnostics:
                    if collect_eta_timing:
                        rerank_ms = self._stop_retrieval_timer(
                            rerank_start, rerank_end
                        )
                        exact_start, exact_end = self._start_retrieval_timer()
                    full_mm_cosine = target_history_cosine(
                        target_content_emb, uni_seq_content_emb
                    )
                    exact_indices, exact_invalid = cp_relevance_topk(
                        dense_model.uni_att_v2,
                        query,
                        fact,
                        eligible_mask,
                        mm_cosine=[full_mm_cosine],
                        keep_top=keep_top,
                    )
                    if collect_eta_timing:
                        exact_ms = self._stop_retrieval_timer(exact_start, exact_end)
                    recall = topk_recall(
                        exact_indices, shortlist_indices, exact_invalid
                    )
                    batch_size = query.shape[0]
                    self.eval_diagnostics[6] += recall.double() * batch_size
                    self.eval_diagnostics[7] += batch_size
                    if collect_eta_timing:
                        self.eval_diagnostics[8] += hash_ms
                        self.eval_diagnostics[9] += rerank_ms
                        self.eval_diagnostics[10] += hash_ms + rerank_ms
                        self.eval_diagnostics[11] += exact_ms
                        self.eval_diagnostics[12] += 1
            else:
                full_mm_cosine = target_history_cosine(
                    target_content_emb, uni_seq_content_emb
                )
                top_k_indices, invalid_mask = cp_relevance_topk(
                    dense_model.uni_att_v2,
                    query,
                    fact,
                    eligible_mask,
                    mm_cosine=[full_mm_cosine],
                    keep_top=keep_top,
                )
            batch_indices = torch.arange(
                query.shape[0], device=query.device
            ).unsqueeze(1).expand(-1, keep_top)

            top_k_embs["205_c"] = retrieval_embs["205_c"]
            top_k_embs["150_2_180_c"] = uni_seq_content_emb[
                batch_indices, top_k_indices
            ]
            top_k_embs["150_2_180_c"][invalid_mask] = 0
            del batch["205_c"]
            del batch["150_2_180_c"]

            for fn in ["150_2_180", "151_2_180"]:
                batch[fn] = batch[fn][batch_indices, top_k_indices]
                batch[fn][invalid_mask] = 0

            if self.args.get("collect_retrieval_diagnostics", False):
                self.last_overlap_summary = overlap_summary(
                    top_k_indices,
                    valid_mask,
                    recent_window=self.args.get("short_window", 50),
                    invalid_mask=invalid_mask,
                )
                if self.eval_diagnostics is not None:
                    self.eval_diagnostics[0] += self.last_overlap_summary["recent_count"]
                    self.eval_diagnostics[1] += self.last_overlap_summary["selected_count"]
                    self.eval_diagnostics[2] += self.last_overlap_summary["expected_recent_count"]

        elif self.args["method"] in ["muse", "din"]:
            # GSU of MUSE
            top_k_fn = ["205_c", "150_2_180_c"]
            for fn in top_k_fn:
                top_k_embs[fn] = batch[fn]
                del batch[fn]
            top_k_embs = self.sparse_model(top_k_embs)
            
            if self.world_size > 1:
                cover_rate_target = self.sparse_model.module.current_scl_cover_rate_target
                cover_rate_seq = self.sparse_model.module.current_scl_cover_rate_seq
            else:
                try:
                    cover_rate_target = self.sparse_model.current_scl_cover_rate_target
                    cover_rate_seq = self.sparse_model.current_scl_cover_rate_seq
                except:
                    cover_rate_target = self.sparse_model.module.current_scl_cover_rate_target
                    cover_rate_seq = self.sparse_model.module.current_scl_cover_rate_seq

            # if you want to see feature cover rate
            # write_info_to_file(f"target cover rate {cover_rate_target }, seq cover rate {cover_rate_seq}", file_path=f"{self.args['exp_name']}_info.txt")

            target_content_emb = top_k_embs["205_c"].unsqueeze(1)
            uni_seq_content_emb = top_k_embs["150_2_180_c"]
            eligible_mask = None
            return_invalid_mask = False
            if self.args["method"] == "muse":
                valid_mask = batch["150_2_180"] != 0
                eligible_mask = build_eligible_mask(
                    valid_mask,
                    short_window=self.args.get("short_window", 50),
                    policy=self.args.get("long_history_policy", "overlap"),
                )
                return_invalid_mask = True
            top_k_result = sim_mm_top_k(
                target_content_emb,
                uni_seq_content_emb,
                keep_top=keep_top,
                eligible_mask=eligible_mask,
                return_invalid_mask=return_invalid_mask,
            )
            if return_invalid_mask:
                top_k_indices, invalid_mask = top_k_result
            else:
                top_k_indices = top_k_result
            batch_indices = torch.arange(target_content_emb.shape[0], device=target_content_emb.device).unsqueeze(1).expand(-1, keep_top)

            top_k_embs["150_2_180_c"] = top_k_embs["150_2_180_c"][batch_indices, top_k_indices]
            if return_invalid_mask:
                top_k_embs["150_2_180_c"][invalid_mask] = 0

            for fn in ["150_2_180", "151_2_180"]:
                batch[fn] = batch[fn][batch_indices, top_k_indices]
                if return_invalid_mask:
                    batch[fn][invalid_mask] = 0

            if return_invalid_mask and self.args.get("collect_retrieval_diagnostics", False):
                self.last_overlap_summary = overlap_summary(
                    top_k_indices,
                    valid_mask,
                    recent_window=self.args.get("short_window", 50),
                    invalid_mask=invalid_mask,
                )
                if self.eval_diagnostics is not None:
                    self.eval_diagnostics[0] += self.last_overlap_summary["recent_count"]
                    self.eval_diagnostics[1] += self.last_overlap_summary["selected_count"]
                    self.eval_diagnostics[2] += self.last_overlap_summary["expected_recent_count"]

        elif self.args["method"] in ["sim-soft"]:
            top_k_fn = ["205", "206", "150_2_180", "151_2_180"]
            for fn in top_k_fn:
                top_k_embs[fn] = batch[fn]
                # del batch[fn]
            top_k_embs = self.sparse_model(top_k_embs)

            target_emb = torch.cat([top_k_embs["205"], top_k_embs["206"]],dim=-1).unsqueeze(1)
            uni_seq_emb = torch.cat([top_k_embs["150_2_180"], top_k_embs["151_2_180"]],dim=-1)

            top_k_indices = sim_soft_top_k(target_emb, uni_seq_emb, keep_top=keep_top)
            batch_indices = torch.arange(target_emb.shape[0], device=target_emb.device).unsqueeze(1).expand(-1, keep_top)

            # for fn in ["150_2_180", "151_2_180"]:
            #     top_k_embs[fn] = top_k_embs[fn][batch_indices, top_k_indices]
            for fn in ["150_2_180", "151_2_180", "150_2_180_c"]:
                batch[fn] = batch[fn][batch_indices, top_k_indices]

            # Note that this process does not accumulate gradients
            # So directly using the embedding from top_k_embs will result in detached embeddings
            top_k_embs = dict()

        elif self.args["method"] in ["sim-hard"]:
            seq_cate = batch["151_2_180"] # (B,L)
            target_cate = batch["206"].unsqueeze(1) # (B,1)
            neg_pad_mask, top_k_indices = sim_hard_top_k(target_cate, seq_cate, keep_top=keep_top)
            batch_indices = torch.arange(target_cate.shape[0], device=target_cate.device).unsqueeze(1).expand(-1, keep_top)
            for fn in ["150_2_180", "151_2_180", "150_2_180_c"]:
                batch[fn] = batch[fn][batch_indices, top_k_indices]
                batch[fn][neg_pad_mask] = 0
        
        elif self.args["method"] in []:
            # do nothing here and leave gsu to dense model forward
            pass

        else:
            raise NotImplementedError

        return batch, top_k_embs

    def log_eval_diagnostics(self):
        if self.eval_diagnostics is None:
            return
        diagnostics = self.eval_diagnostics.clone()
        if self.world_size > 1:
            dist.all_reduce(diagnostics, op=dist.ReduceOp.SUM)
        if not self.is_main_process:
            return
        (
            recent,
            selected,
            expected,
            gate_sum,
            gate_square_sum,
            gate_count,
            eta_recall_sum,
            eta_recall_count,
            eta_hash_ms,
            eta_rerank_ms,
            eta_total_ms,
            eta_exact_ms,
            eta_timing_count,
        ) = diagnostics.tolist()
        if selected > 0:
            logging.info(
                "[Diagnostics] RecentOverlap=%.6f RecentEnrichment=%.6f",
                recent / selected,
                recent / max(expected, 1e-12),
            )
        if gate_count > 0:
            gate_mean = gate_sum / gate_count
            gate_variance = max(gate_square_sum / gate_count - gate_mean * gate_mean, 0.0)
            logging.info(
                "[Diagnostics] HorizonGateMean=%.6f HorizonGateStd=%.6f",
                gate_mean,
                gate_variance ** 0.5,
            )
        if eta_timing_count > 0:
            logging.info(
                "[Diagnostics] ETARecall@%d=%.6f "
                "ETAHashHammingMs=%.3f ETAGatherRerankMs=%.3f "
                "ETATotalMs=%.3f FullExactCPMs=%.3f",
                self.args["hash_shortlist"],
                eta_recall_sum / max(eta_recall_count, 1.0),
                eta_hash_ms / eta_timing_count,
                eta_rerank_ms / eta_timing_count,
                eta_total_ms / eta_timing_count,
                eta_exact_ms / eta_timing_count,
            )

    def _start_retrieval_timer(self):
        if self.thresholds.is_cuda:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            return start, end
        return time.perf_counter(), None

    def _stop_retrieval_timer(self, start, end):
        if end is not None:
            end.record()
            end.synchronize()
            return start.elapsed_time(end)
        return (time.perf_counter() - start) * 1000.0

    def init_metric(self):
        self.accumulated_metric = {
            "PN": 0,
            "wins": 0,
            "weighted_auc_sum": 0,
            "impression_counts": 0,
            "auc": 0,
            "gauc": 0,
            "tp": torch.zeros(self.num_thresholds, dtype=torch.long).to(self.device),
            "fp": torch.zeros(self.num_thresholds, dtype=torch.long).to(self.device),
            "fn": torch.zeros(self.num_thresholds, dtype=torch.long).to(self.device),
            "tn": torch.zeros(self.num_thresholds, dtype=torch.long).to(self.device)
        }

    def update_metric(self, metrics):
        wins = metrics["wins"]
        pn = metrics["P"] * metrics["N"]
        weighted_auc_sum = metrics["weighted_auc_sum"]
        impression_counts = metrics["impression_counts"]

        tp, fp, fn, tn = metrics["tp"], metrics["fp"], metrics["fn"], metrics["tn"]
        # reduce tp, fp, fn, tn
        state = torch.cat([tp, fp, tn, fn], dim=0)
        # Perform all-reduce sum across all processes
        dist.all_reduce(state, op=dist.ReduceOp.SUM)
        # Split back into individual components
        tp, fp, tn, fn = state.split(
            [self.accumulated_metric["tp"].numel(), self.accumulated_metric["fp"].numel(), self.accumulated_metric["tn"].numel(), self.accumulated_metric["fn"].numel()], dim=0
        )
        self.accumulated_metric["tp"] += tp
        self.accumulated_metric["fp"] += fp
        self.accumulated_metric["tn"] += tn
        self.accumulated_metric["fn"] += fn

        wins_reduced = self._reduce_scalar(wins)
        pn_reduced = self._reduce_scalar(pn)
        weighted_auc_sum_reduced = self._reduce_scalar(weighted_auc_sum)
        impression_counts_reduced = self._reduce_scalar(impression_counts)

        self.accumulated_metric["PN"] += pn_reduced
        self.accumulated_metric["wins"] += wins_reduced
        self.accumulated_metric["weighted_auc_sum"] += weighted_auc_sum_reduced
        self.accumulated_metric["impression_counts"] += impression_counts_reduced
        # self.accumulated_metric["auc"] = self.accumulated_metric["wins"] / self.accumulated_metric["PN"]
        self.accumulated_metric["auc"] = calc_auroc_gpu(self.accumulated_metric["tp"], self.accumulated_metric["fp"], self.accumulated_metric["tn"], self.accumulated_metric["fn"])
        self.accumulated_metric["gauc"] = self.accumulated_metric["weighted_auc_sum"] / self.accumulated_metric["impression_counts"]

        auc = wins_reduced / pn_reduced
        gauc = weighted_auc_sum_reduced / impression_counts_reduced
        return auc, gauc

    def log_metric(self, auc, gauc, cur_loss, avg_loss, mode="Train"):
        time_since_last_log = time.time() - self.last_log_time
        if self._batch_index == 0:
            qps = 1 / time_since_last_log if time_since_last_log > 0 else 0.0
        elif self._batch_index % 100 == 0:
            qps = 100 / time_since_last_log if time_since_last_log > 0 else 0.0
        else:
            qps = (self._batch_index % 100) / time_since_last_log if time_since_last_log > 0 else 0.0

        log_info = (f"[{mode}] <Epoch={self._epoch_index}, "
                    f"Gstep={self._total_steps}, Lstep={self._batch_index}, "
                    f"QPS={qps:.2f}, "
                    f"Cur_AUC={auc:.6f}, Avg_AUC={self.accumulated_metric['auc']:.6f}, "
                    f"Cur_GAUC={gauc:.6f}, Avg_GAUC={self.accumulated_metric['gauc']:.6f}, "
                    f"Cur_Loss={cur_loss:.6f}, Avg_Loss={avg_loss:.6f}>")
        if self.is_main_process:
            logging.info(log_info)
        self.last_log_time = time.time()
    
    def _reduce_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        To reduce loss
        """
        if self.world_size > 1:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            tensor /= self.world_size
        return tensor

    def _gather_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Not used now
        """
        if self.world_size > 1:
            # tensor shape (B,2)
            full_tensor = torch.empty(world_size * tensor.shape[0], 2, dtype=tensor.dtype, device=tensor.device)
            dist.all_gather_into_tensor(full_tensor, tensor)
            return full_tensor
        else:
            return tensor

    def _reduce_scalar(self, scalar: float) -> float:
        """
        To reduce metrics
        """
        if self.world_size <= 1:
            return scalar
        tensor = torch.tensor(scalar).to(self.device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        # return (tensor / self.world_size).item()
        return tensor.item()
    
    @torch.no_grad()
    def update_count(self, item_ids: torch.Tensor):
        flat_ids = item_ids.view(-1).long()
        counts = torch.bincount(flat_ids, minlength=self.id_counts.shape[0])
        if self.world_size > 1:
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        self.id_counts += counts.to(self.device)

    def init_id_count(self):
        id_counts_dict = dict()
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        id_counts_dict["150_2_180"] = torch.load("./ckpt/id_counts_uni_seq.pt").to(self.device)
        id_counts_dict["150_1_180"] = torch.load("./ckpt/id_counts_short_seq.pt").to(self.device)
        id_counts_dict["205"] = torch.load("./ckpt/id_counts_target.pt").to(self.device)
        self.id_counts_dict = id_counts_dict
    
    def save_model(self):
        if "ckpt_path" in self.args:
            ckpt_path = self.args["ckpt_path"]
        else:
            ckpt_path = "./ckpt"

        prepare_checkpoint_dir(ckpt_path)
        self.dense_model.module.save_ckpt(ckpt_path=os.path.join(ckpt_path, f"{self.args['exp_name']}_dense.ckpt"), rank=self.rank)
        self.sparse_model.module.save_ckpt(ckpt_path=os.path.join(ckpt_path, f"{self.args['exp_name']}_sparse.ckpt"), rank=self.rank)
        # torch.save(self.id_counts.cpu(), "id_counts.pt")

    # @torch.no_grad()
    # def filter_seq_item_id(self, batch):
    #     remapped_ids = self.sparse_model.module.lookup("150_2_180", batch["150_2_180"])
    #     id_counts_dict = self.id_counts_dict
    #     # mask = torch.zeros(batch["150_2_180"].shape, dtype=torch.bool, device=self.device)
    #     thresholds = {
    #         "150_2_180": 10000,
    #         "150_1_180": 10000,
    #         "205": 1
    #     }
    #     # for key in id_counts_dict:
    #     #     count = id_counts_dict[key]
    #     #     th = thresholds[key]
    #     #     if key in ["150_2_180", "150_1_180"]:
    #     #         mask |= (count[remapped_ids] >= th)
    #     #     else:
    #     #         continue
    #     mask = ((id_counts_dict["150_2_180"][remapped_ids] + id_counts_dict["150_1_180"][remapped_ids]) >= 1000)
    #     batch["150_2_180"][mask] = 0
    #     return batch
    
    # @torch.no_grad()
    # def random_mask_ids(self, batch, mask_prob=0.1, mask_value=0):
    #     if mask_prob == 0.0:
    #         return batch
    #     # Generate a random mask: True for positions to keep, False for positions to mask
    #     rand = torch.rand(batch["150_2_180"].shape, device=self.device)
    #     mask = (rand > mask_prob)  # Keep with prob (1 - mask_prob)
    #     # Apply mask: set masked positions to mask_value
    #     batch["150_2_180"][~mask] = mask_value
    #     return batch
