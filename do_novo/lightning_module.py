import os
import numpy as np
import pandas as pd
import torch, wandb, re
from pyteomics import mass
from lightning.pytorch.core import LightningModule

from do_novo.model.sinusoidal import AugmentedPeakEncoder, FloatEncoder
from do_novo.model.encoder import SpectrumTransformerEncoder, SpectrumTheoCrossAttention
from do_novo.model.decoder import PeptideTransformerDecoder
from do_novo.model.sampler import CrossAttnTopKTheoPeakSampler
from do_novo.predictor import Predictor
from do_novo.utils import aa_match_batch, aa_match_metrics, ptm_aa_match_metrics, ptm_aa_match_batch, sample_theo_mz_candidates, topk_missing_metrics


class DenovoLightningModule(LightningModule):
    def __init__(
            self, 
            d_model,
            n_layers,
            rt_width,
            n_head,
            dropout,
            dim_feedforward,
            tokenizer,
            max_charge,
            lr = 1e-4,
            weight_decay = 1e-9,
            betas = (0.9, 0.98),
            eps = 1e-6,
            train_phase = 'sampler',
            topk = 8
    ):
        super().__init__()
        self.save_hyperparameters()
        self.tokenizer = tokenizer
        self.vocab_size = len(self.tokenizer) + 1
        self.train_phase = train_phase

        self.n_bins_200_300 = 5000
        self.topk = topk
        mz_centers = torch.linspace(200.0, 300.0, steps=self.n_bins_200_300, dtype=torch.float32)
        self.register_buffer("mz_bin_centers_200_300", mz_centers)

        self.peak_encoder = AugmentedPeakEncoder(d_model=d_model, max_rt_wavelength=2*rt_width)
        self.spectrum_encoder = SpectrumTransformerEncoder(
            d_model=d_model,
            n_head=n_head,
            n_layers=n_layers,
            dropout=dropout,
            dim_feedforward=dim_feedforward,
            peak_encoder=self.peak_encoder,
        )
        self.peptide_decoder = PeptideTransformerDecoder(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            n_layers=n_layers,
            dropout=dropout,
            n_tokens=tokenizer,
            max_charge=max_charge
        )
        self.CELoss = torch.nn.CrossEntropyLoss(ignore_index=0)

        self.mz_encoder = FloatEncoder(d_model=d_model, max_wavelength=2*rt_width)
        self.spectrum_theo_cross_attention = SpectrumTheoCrossAttention(
            d_model=d_model,
            n_head=n_head,
            dropout=dropout,
        )
        
        self.topk_theo_sampler = CrossAttnTopKTheoPeakSampler(
            d_model=d_model,
            n_bins=self.n_bins_200_300,
            hidden_dim=d_model,
            k=self.topk,
        )
        
        
        if train_phase == 'teacher':
            for p in self.topk_theo_sampler.parameters():
                p.requires_grad = False
        else:
            for p in self.peak_encoder.parameters():
                p.requires_grad = False
            for p in self.spectrum_encoder.parameters():
                p.requires_grad = False
            for p in self.spectrum_theo_cross_attention.parameters():
                p.requires_grad = True
            for p in self.peptide_decoder.parameters():
                p.requires_grad = False

            for p in self.topk_theo_sampler.parameters():
                p.requires_grad = True


    def _step(self, batch, stage):
        spectra, precursors, peptide_token, frac, fragment_label, theo_mz_bin, missing_mz_bin, peptide_ids, window_upper, window_lower, mz1, peptide_seq = batch
        
        # ----- 0) Predefine part ----- #
        b_size = batch[0].size(0)
        mz_full = self.mz_bin_centers_200_300.unsqueeze(0).expand(spectra.size(0), -1).to(device=self.device, dtype=self.dtype)
        
        
        # ----- 1) Model part ----- #
        emb0, emb0_mask = self.spectrum_encoder(spectra)
        
        if self.train_phase == 'teacher':
            candidate_bin = sample_theo_mz_candidates(
                theo_mz_bin=theo_mz_bin,
                missing_mz_bin=missing_mz_bin,
                peptide_ids=peptide_ids,
                topK=self.topk,
                pos_missing_prob=0.5,
                pos_exist_prob= 0.5,
            )
        else:
            candidate_bin, probs, logits, extras, prior_gate_value = self.topk_theo_sampler(
                emb=emb0,
                emb_mask=emb0_mask,
            )
            
            
        cand_mz = mz_full
        cand_mz_enc = self.mz_encoder(cand_mz)
        # candidate_bin = candidate_bin * theo_mz_bin
        emb, emb_mask, attn_weights, cand_mz_mask = self._run_cross_attn_with_candidate(
            emb0, emb0_mask, cand_mz_enc, candidate_bin
        )
        

        if stage == 'train':
            pred        = self.peptide_decoder(peptide_token[:, :-1], precursors, memory=emb, memory_key_padding_mask=emb_mask)
            pred_token  = torch.argmax(pred[:, 1:, :], dim=2)
        else:
            candidate_bin_refined, pred_token, pred, stats = self._refine_candidate_by_mass(
                probs=probs,
                topk=self.topk,
                emb0=emb0,
                emb0_mask=emb0_mask,
                cand_mz_enc=cand_mz_enc,
                precursors=precursors,
                peptide_token=peptide_token,
                window_lower=window_lower,
                window_upper=window_upper,
                mz1=mz1,
            )
            
            # pred_token, pred = self._greedy_decode(
            #     emb, emb_mask, precursors, peptide_token
            # )
            
                
        # ----- Model part ----- #
        
        
        # ----- 2) Loss part ----- #
        if self.train_phase == 'teacher':
            pred_f  = pred[:,1:,:].reshape(-1, len(self.tokenizer)+1)
            loss_ce = self.CELoss(pred_f, peptide_token[:,1:].flatten())
            
            score = torch.log(attn_weights.mean(dim=1).clamp_min(1e-12))
            loss_attn = self.three_tier_key_rank_loss(score, candidate_bin, theo_mz_bin, missing_mz_bin)
            
            total_loss = loss_ce + loss_attn
            
        else:
            pred_f  = pred[:,1:,:].reshape(-1, len(self.tokenizer)+1)
            loss_ce = self.CELoss(pred_f, peptide_token[:,1:].flatten())
            
            score = torch.log(attn_weights.mean(dim=1).clamp_min(1e-12))
            loss_attn = self.three_tier_key_rank_loss(score, candidate_bin, theo_mz_bin, missing_mz_bin)
            
            focus_loss = self._topk_pairwise_hinge_loss(score=logits, pos_mask=(theo_mz_bin == 1), 
                                                        k_pos=self.topk, k_neg=self.topk * 16, margin=0.5,
                                                        # hard_in_candidate=True, candidate_mask=candidate_bin.bool(),
                                                        # hard_neg=True, pos_max=True,
                                                        )
            
            total_loss = loss_attn + focus_loss
        # ----- Loss part ----- #
        

        # ----- 3) Metric part ----- #
        peptides_pred       = self.tokenizer.detokenize(pred_token,     trim_stop_token=True, join=False, trim_start_token=False)
        peptides_true       = self.tokenizer.detokenize(peptide_token,  trim_stop_token=True, join=False, trim_start_token=True)
        aa_precision, aa_recall, pep_recall = aa_match_metrics(*aa_match_batch(peptides_pred, peptides_true, self.tokenizer.residues))
        ptm_aa_precision, ptm_aa_recall = ptm_aa_match_metrics(*ptm_aa_match_batch(peptides_pred, peptides_true, self.tokenizer.residues))
        
        score = torch.log(attn_weights.mean(dim=1).clamp_min(1e-12))
        attn_theo_gt_bin = (theo_mz_bin == 1) & (candidate_bin == 1)
        attn_missing_gt_bin = (missing_mz_bin == 1) & (candidate_bin == 1)
        attn_theo_hit, attn_theo_recall, attn_theo_hit_top1 = topk_missing_metrics(score=score, gt_mask=attn_theo_gt_bin, k=self.topk)
        attn_missing_hit, attn_missing_recall, attn_missing_hit_top1 = topk_missing_metrics(score=score, gt_mask=attn_missing_gt_bin, k=self.topk)
        
        if self.train_phase != 'teacher':
            score = logits
            sampler_theo_gt_bin = (theo_mz_bin == 1)
            sampler_missing_gt_bin = (missing_mz_bin == 1)
            sampler_theo_hit, sampler_theo_recall, sampler_theo_hit_top1 = topk_missing_metrics(score=score, gt_mask=sampler_theo_gt_bin, k=self.topk)
            sampler_missing_hit, sampler_missing_recall, sampler_missing_hit_top1 = topk_missing_metrics(score=score, gt_mask=sampler_missing_gt_bin, k=self.topk)
        # ----- Metric part ----- #
        
        
        # ----- 4) Logging part ----- #
        self.log(f"{stage}/loss/total",                     total_loss.detach(),on_epoch=True, sync_dist=True, batch_size=b_size)
        self.log(f"{stage}/loss/ce",                        loss_ce.detach(),   on_epoch=True, sync_dist=True, batch_size=b_size)
        self.log(f"{stage}/loss/attention",                 loss_attn.detach(), on_epoch=True, sync_dist=True, batch_size=b_size)
        self.log(f"{stage}/AA Prec",                        aa_precision,       on_epoch=True, sync_dist=True, batch_size=b_size)
        self.log(f"{stage}/AA Recall",                      aa_recall,          on_epoch=True, sync_dist=True, batch_size=b_size)
        self.log(f"{stage}/Pep Recall",                     pep_recall,         on_epoch=True, sync_dist=True, batch_size=b_size)
        self.log(f"{stage}/PTM AA Prec",                    ptm_aa_precision,   on_epoch=True, sync_dist=True, batch_size=b_size)
        self.log(f"{stage}/PTM AA Recall",                  ptm_aa_recall,      on_epoch=True, sync_dist=True, batch_size=b_size)
        self.log(f"{stage}/ATTN_theo_Hit@{self.topk}",      attn_theo_hit,      on_epoch=True, sync_dist=True, batch_size=b_size)
        self.log(f"{stage}/ATTN_theo_Hit@1",                attn_theo_hit_top1, on_epoch=True, sync_dist=True, batch_size=b_size)
        self.log(f"{stage}/ATTN_theo_Recall@{self.topk}",   attn_theo_recall,   on_epoch=True, sync_dist=True, batch_size=b_size)
        self.log(f"{stage}/ATTN_missing_Hit@{self.topk}",   attn_missing_hit,   on_epoch=True, sync_dist=True, batch_size=b_size)
        self.log(f"{stage}/ATTN_missing_Hit@1",             attn_missing_hit_top1,on_epoch=True, sync_dist=True, batch_size=b_size)
        self.log(f"{stage}/ATTN_missing_Recall@{self.topk}",attn_missing_recall,on_epoch=True, sync_dist=True, batch_size=b_size)
        if self.train_phase != 'teacher':
            self.log(f"{stage}/loss/focus",                         focus_loss.detach(),    on_epoch=True, sync_dist=True, batch_size=b_size)
            self.log(f"{stage}/SAMPLER_theo_Hit@{self.topk}",       sampler_theo_hit,       on_epoch=True, sync_dist=True, batch_size=b_size)
            self.log(f"{stage}/SAMPLER_theo_Hit@1",                 sampler_theo_hit_top1,  on_epoch=True, sync_dist=True, batch_size=b_size)
            self.log(f"{stage}/SAMPLER_theo_Recall@{self.topk}",    sampler_theo_recall,    on_epoch=True, sync_dist=True, batch_size=b_size)
            self.log(f"{stage}/SAMPLER_missing_Hit@{self.topk}",    sampler_missing_hit,    on_epoch=True, sync_dist=True, batch_size=b_size)
            self.log(f"{stage}/SAMPLER_missing_Hit@1",              sampler_missing_hit_top1,on_epoch=True, sync_dist=True, batch_size=b_size)
            self.log(f"{stage}/SAMPLER_missing_Recall@{self.topk}", sampler_missing_recall, on_epoch=True, sync_dist=True, batch_size=b_size)
            # self.log(f"{stage}/wrong_ratio_inwin", stats["ratio_inwin_among_wrong"], on_epoch=True, sync_dist=True, batch_size=b_size)
            # self.log(f"{stage}/wrong_ratio_oob",   stats["ratio_oob_among_wrong"],   on_epoch=True, sync_dist=True, batch_size=b_size)

        # ----- Logging part ----- #
        
        return total_loss, peptides_pred, peptides_true, frac
    


    def training_step(self, batch, batch_idx):
        loss, _, _, _ = self._step(batch, 'train')

        return loss


    def validation_step(self, batch, batch_idx):
        loss, peptides_pred, peptides_true, frac = self._step(batch, 'val')

        if batch_idx==0 and self.trainer.is_global_zero and self.logger is not None:
            self.generate_table(peptides_pred, peptides_true, frac, 'gen')
        
        return loss
    

    def test_step(self, batch, batch_idx):
        loss, peptides_pred, peptides_true, frac = self._step(batch, 'test')

        if batch_idx==0 and self.trainer.is_global_zero and self.logger is not None:
            self.generate_table(peptides_pred, peptides_true, frac, 'gen')
        
        return loss

    ######## -------- Predictor -------- ########
    def on_predict_start(self):
        self._predictor = Predictor(self, max_decode_len=50)
        self._pred_rows = []

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        out = self._predictor.predict_batch(batch)

        pep_pred = out["peptide_pred"]
        meta = out.get("meta", {})
        frac = out.get("frac", None)

        spectra, precursors, _, _, _, _, _, _, window_upper, window_lower, mz1, _ = batch

        B = len(pep_pred)

        def meta_get(key, default=None):
            if isinstance(meta, dict) and key in meta:
                return meta[key]
            return [default] * B
        
        name = meta_get("name", None)
        feature_id = meta_get("feature_id", None)

        for i in range(B):
            self._pred_rows.append({
                "name": name[i],
                "scan_id": f"scan_{i}",
                "feature_id": feature_id[i],
                "window_lower": float(window_lower[i].item()),
                "window_upper": float(window_upper[i].item()),
                "precursor_mz": float(precursors[i, 0].item()),
                "charge": int(precursors[i, 1].item()),
                "mz1": float(mz1[i].item()) if mz1 is not None else None,
                "frac": float(frac[i].item()) if (torch.is_tensor(frac) and frac is not None) else frac,
                "peptide_pred": pep_pred[i],
            })

        return None

    def on_predict_epoch_end(self):
        if not self.trainer.is_global_zero:
            return

        df = pd.DataFrame(self._pred_rows)

        outdir = "preds"
        os.makedirs(outdir, exist_ok=True)
        df.to_csv(f"{outdir}/predictions_ptm.csv", index=False)
    ######## -------- Predictor -------- ########



    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay, betas=self.hparams.betas, eps=self.hparams.eps )
        return optimizer


    def generate_table(self, peptides_pred, peptides_true, frac, name):
        if peptides_pred is None:
            return
        table = wandb.Table(columns=["epoch", "pred", "true", "frac"])
        for i in range(len(peptides_pred)):
            pred_str = "".join(peptides_pred[i])
            true_str = "".join(peptides_true[i])
            frac_str = f"{frac[i]}"
            table.add_data(int(self.current_epoch), pred_str, true_str, frac_str)

        self.logger.experiment.log({
            f"{name}/preview": table,
            "epoch": int(self.current_epoch),
        })
        
    def three_tier_key_rank_loss(
        self,
        score,
        candidate_bin,
        theo_mz_bin,
        missing_mz_bin,
        k_pm = 2,       k_pe = 4,      k_neg = 32,
        m_pm_pe = 0.1,  m_pe_neg = 0.2, m_pm_neg = 0.5,
        w_pm_pe = 1.0,  w_pe_neg = 1.0, w_pm_neg = 1.0,
    ) -> torch.Tensor:
        device, dtype = score.device, score.dtype
        B, n_bins = score.shape
        
        candidate_sel = candidate_bin.bool()
        pos_missing = missing_mz_bin.bool() & candidate_sel
        pos_exist   = theo_mz_bin.bool() & ~missing_mz_bin.bool() & candidate_sel
        neg_mask    = candidate_sel & ~(pos_missing | pos_exist)
        

        def topk_vals(x, mask, k):
            if k <= 0:
                return None
            x = x.masked_fill(~mask, -torch.finfo(x.dtype).max)
            valid = mask.sum(dim=1)
            if (valid == 0).all():
                return None
            k_eff = torch.clamp(valid, max=k).to(torch.long)
            v, _ = torch.topk(x, k=min(k, n_bins), dim=1)
            return v, k_eff

        def pair_hinge(a_pack, b_pack, margin):
            if a_pack is None or b_pack is None:
                return None
            a, ka = a_pack  # (B, kAmax), (B,)
            b, kb = b_pack  # (B, kBmax), (B,)

            a_mask = torch.arange(a.size(1), device=device)[None, :] < ka[:, None]
            b_mask = torch.arange(b.size(1), device=device)[None, :] < kb[:, None]
            if (a_mask.sum(dim=1) == 0).all() or (b_mask.sum(dim=1) == 0).all():
                return None

            diff = margin - (a.unsqueeze(2) - b.unsqueeze(1))
            hinge = torch.relu(diff)

            pair_mask = a_mask.unsqueeze(2) & b_mask.unsqueeze(1)
            num = pair_mask.sum().clamp_min(1)
            return (hinge * pair_mask).sum() / num

        pm_pack  = topk_vals(score, pos_missing.bool(), k_pm)
        pe_pack  = topk_vals(score, pos_exist.bool(),   k_pe)
        neg_pack = topk_vals(score, neg_mask.bool(),    k_neg)

        losses = []
        l = pair_hinge(pm_pack, pe_pack,  m_pm_pe)
        if l is not None: losses.append(w_pm_pe * l)
        l = pair_hinge(pe_pack, neg_pack, m_pe_neg)
        if l is not None: losses.append(w_pe_neg * l)
        l = pair_hinge(pm_pack, neg_pack, m_pm_neg)
        if l is not None: losses.append(w_pm_neg * l)

        if not losses:
            return score.sum() * 0.0
        return torch.stack(losses).sum()
    
    
    def _topk_pairwise_hinge_loss(
        self,
        score: torch.Tensor,
        pos_mask: torch.Tensor,
        k_pos: int = 2,
        k_neg: int = 16,
        margin: float = 0.1,
        hard_in_candidate: bool = False,
        candidate_mask: torch.Tensor | None = None,
        hard_neg: bool = False,
        pos_max: bool = False,
    ) -> torch.Tensor:
        
        B, n_bins = score.shape
        device = score.device
        dtype = score.dtype

        pos_mask = pos_mask.bool()

        if hard_in_candidate:
            if candidate_mask is None:
                raise ValueError("hard_in_candidate=True requires candidate_mask (B,N).")
            cand = candidate_mask.bool()
            pos_mask_eff = pos_mask & cand
            neg_mask_eff = (~pos_mask) & cand
        else:
            pos_mask_eff = pos_mask
            neg_mask_eff = ~pos_mask

        losses = []

        for b in range(B):
            pos_idx = pos_mask_eff[b].nonzero(as_tuple=False).squeeze(-1)
            neg_idx = neg_mask_eff[b].nonzero(as_tuple=False).squeeze(-1)
            
            if pos_idx.numel() == 0 or neg_idx.numel() == 0:
                continue

            pos_scores = score[b, pos_idx]  # (num_pos,)
            neg_scores = score[b, neg_idx]  # (num_neg,)
            
            k_pos_b = pos_scores.numel() # min(k_pos, pos_scores.numel())
            k_neg_b = min(k_neg, neg_scores.numel())

            pos_topk, _ = torch.topk(pos_scores, k=k_pos_b)
            neg_topk, _ = torch.topk(neg_scores, k=k_neg_b)

            if hard_neg:
                if pos_max:
                    min_pos = pos_topk.max() # pos_topk.min()
                else:
                    min_pos = pos_topk.min()
                max_neg = neg_topk.max()
                loss = torch.relu(margin - (min_pos - max_neg))
                losses.append(loss)
            else:
                pos_expand = pos_topk.unsqueeze(1)
                neg_expand = neg_topk.unsqueeze(0)
                diff = margin - (pos_expand - neg_expand)
                loss_mat = torch.relu(diff)
                losses.append(loss_mat.mean())

        if len(losses) == 0:
            return score.sum() * 0.0

        return torch.stack(losses).mean()
        
        
        
        
        
        
        
        
    def _run_cross_attn_with_candidate(
        self,
        emb0: torch.Tensor,
        emb0_mask: torch.Tensor,
        cand_mz_enc: torch.Tensor,
        candidate_bin: torch.Tensor,   # (B, N) 0/1
    ):
        cand_mz_mask = (candidate_bin == 0)

        all_masked = cand_mz_mask.all(dim=1)
        if all_masked.any():
            cand_mz_mask[all_masked, 0] = False

        emb, emb_mask, attn_weights = self.spectrum_theo_cross_attention(
            emb0, emb0_mask, theo_mz_enc=cand_mz_enc, theo_mask=cand_mz_mask
        )
        return emb, emb_mask, attn_weights, cand_mz_mask

    @torch.no_grad()
    def _greedy_decode(
        self,
        emb: torch.Tensor,
        emb_mask: torch.Tensor,
        precursors: torch.Tensor,
        peptide_token: torch.Tensor,
    ) -> torch.Tensor:
        B, T = peptide_token.shape
        BOS, EOS = self.tokenizer.index['BOS'], self.tokenizer.index['EOS']

        cur_token = torch.full((B, 1), BOS, dtype=torch.long, device=emb.device)
        finished = torch.zeros(B, dtype=torch.bool, device=emb.device)

        for _ in range(T - 1):
            pred = self.peptide_decoder(cur_token, precursors, memory=emb, memory_key_padding_mask=emb_mask)
            next_scores = torch.softmax(pred[:, -1, :], dim=1)  # (B, V)
            next_aas = torch.argmax(next_scores, dim=1)
            next_aas = torch.where(finished, torch.full_like(next_aas, EOS), next_aas)

            cur_token = torch.cat([cur_token, next_aas.unsqueeze(1)], dim=1)
            finished = finished | (next_aas == EOS)

        return cur_token[:, 1:], pred  # (B, T-1)

    def _mass_oob_indices(
        self,
        pred_token: torch.Tensor,
        precursors: torch.Tensor,
        window_lower: torch.Tensor,
        window_upper: torch.Tensor,
        peptide_token,
        mz1: torch.Tensor = None,
    ) -> list[int]:
        peptides_pred = self.tokenizer.detokenize(
            pred_token, trim_stop_token=True, join=True, trim_start_token=False
        )
        peptides_true = self.tokenizer.detokenize(
            peptide_token, trim_stop_token=True, join=True, trim_start_token=True
        )
        
        # --- stats ---
        wrong_total, wrong_inwin, wrong_oob, wrong_nan = 0, 0, 0, 0
        # --- stats ---
        
        out = []
        for i, pep in enumerate(peptides_pred):
            
            # --- stats ---
            is_wrong = (peptides_pred[i] != peptides_true[i])
            if is_wrong:
                wrong_total += 1
            # --- stats ---
            
            aa_comp = dict(mass.std_aa_comp)
            cam_delta = mass.Composition("H3C2NO")
            aa_comp["C"] = aa_comp["C"] + cam_delta

            mz = self.calc_mz_simple(pep, int(precursors[i,1].item()), aa_comp)
            
            # --- stats ---
            in_window = False
            if np.isfinite(mz):
                in_window = (window_lower[i]-1 <= mz <= window_upper[i]+1)
                
            if is_wrong:
                if not np.isfinite(mz):
                    wrong_nan += 1
                elif in_window:
                    wrong_inwin += 1
                else:
                    wrong_oob += 1
            stats = {
                "wrong_total": wrong_total,
                "wrong_inwin": wrong_inwin,
                "wrong_oob": wrong_oob,
                "wrong_nan": wrong_nan,
                "ratio_inwin_among_wrong": (wrong_inwin / max(1, wrong_total)),
                "ratio_oob_among_wrong":   (wrong_oob   / max(1, wrong_total)),
                "ratio_nan_among_wrong":   (wrong_nan   / max(1, wrong_total)),
                "wrong_finite": (wrong_inwin + wrong_oob),
                "ratio_inwin_among_wrong_finite": (wrong_inwin / max(1, wrong_inwin + wrong_oob)),
                "ratio_oob_among_wrong_finite":   (wrong_oob   / max(1, wrong_inwin + wrong_oob)),
            }
            # --- stats ---
            
            if not np.isfinite(mz):
                out.append(i)
                continue

            if mz1 is not None:
                if abs(mz-mz1[i].item()) >= 1.0:
                    out.append(i)
            else:
                if not (window_lower[i]-1 <= mz <= window_upper[i]+1):
                    out.append(i)
            mz_gt = self.calc_mz_simple(peptides_true[i], int(precursors[i,1].item()), aa_comp)
            if not (window_lower[i]-1 <= mz_gt <= window_upper[i]+1):
                print(peptides_true[i])
                print(f"{mz_gt}\t{window_lower[i]-1}\t{window_upper[i]+1}")
        return out, stats

    @torch.no_grad()
    def _refine_candidate_by_mass(
        self,
        probs: torch.Tensor,          # (B, N)
        topk: int,
        emb0: torch.Tensor,
        emb0_mask: torch.Tensor,
        cand_mz_enc: torch.Tensor,
        precursors: torch.Tensor,
        peptide_token: torch.Tensor,
        window_lower: torch.Tensor,
        window_upper: torch.Tensor,
        mz1: torch.Tensor = None,
    ):
        B, N = probs.shape
        device = probs.device

        topk_idx = torch.topk(probs, topk, dim=1).indices  # (B, K)

        sel = torch.zeros((B, N), dtype=torch.bool, device=device)
        sel.scatter_(1, topk_idx, True)

        pred_token, pred = None, None

        for r in range(topk):
            candidate_bin = sel.to(dtype=probs.dtype)

            emb, emb_mask, _, _ = self._run_cross_attn_with_candidate(
                emb0, emb0_mask, cand_mz_enc, candidate_bin
            )

            pred_token, pred = self._greedy_decode(
                emb, emb_mask, precursors, peptide_token
            )

            oob, stats = self._mass_oob_indices(
                pred_token, precursors, window_lower, window_upper, peptide_token, mz1
            )
            
            if len(oob) == 0:
                break

            if r >= topk:
                break
            for i in oob:
                cur_idx = topk_idx[i]
                sel[i, cur_idx] = False
                sel[i, cur_idx[r]] = True

        return sel.to(dtype=probs.dtype), pred_token, pred, stats
    
    
    
    def calc_mz_simple(self, seq_with_mod: str, z: int, aa_comp):
        MOD_SHIFT = {
            "Oxidation": 15.99491461957,      # UniMod:35 (M)
            "Carbamidomethyl": 57.021463735,  # UniMod:4 (C)
            "Acetyl": 42.01056468403,         # UniMod:1 (N-term)
            "Deamidated": 0.984016,
            "Ammonia-loss": -17.026549,
            "Carbamyl": 43.006,
            "pyro-Glu": -18.011,
            "Glu->pyro-Glu": -18.011,
        }
        PROTON = 1.007276466812
        H2O    = 18.01056468403
        CAM    = 57.021463735
        
        s = str(seq_with_mod)

        shifts = 0.0

        m = re.match(r"^\[([^\]]+)\]-", s)
        if m:
            modname = m.group(1)
            shifts += MOD_SHIFT.get(modname, 0.0)
            s = s[m.end():]

        for modname in re.findall(r"\[([^\]]+)\]", s):
            shifts += MOD_SHIFT.get(modname, 0.0)

        plain = re.sub(r"\[[^\]]+\]", "", s)
        plain = _sanitize_plain(plain)
        if plain is None or len(plain) == 0:
            return float("nan")

        neutral = mass.calculate_mass(sequence=plain, aa_comp=aa_comp) + shifts
        if 'Deamidated' in seq_with_mod:
            neutral = neutral - H2O
        if 'Carbamyl' in seq_with_mod:
            neutral = neutral - PROTON
        mz = (neutral + PROTON * int(z)) / int(z)
        
        return mz
    
def _sanitize_plain(seq: str) -> str:
    AA20 = set("ACDEFGHIKLMNPQRSTVWY")
    seq = seq.upper()
    seq = re.sub(r"[^A-Z]", "", seq)
    seq = "".join([c for c in seq if c in AA20])
    return seq