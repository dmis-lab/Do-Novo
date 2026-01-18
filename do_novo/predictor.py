import numpy as np
import torch
from pyteomics import mass

class Predictor:
    def __init__(self, model, max_decode_len=50):
        self.m = model
        self.max_decode_len = max_decode_len

    @torch.no_grad()
    def greedy_decode_len(self, emb, emb_mask, precursors, max_len=None):
        max_len = max_len or self.max_decode_len
        B = emb.size(0)
        BOS, EOS = self.m.tokenizer.index['BOS'], self.m.tokenizer.index['EOS']

        cur = torch.full((B, 1), BOS, dtype=torch.long, device=emb.device)
        finished = torch.zeros(B, dtype=torch.bool, device=emb.device)
        pred = None

        for _ in range(max_len):
            pred = self.m.peptide_decoder(cur, precursors, memory=emb, memory_key_padding_mask=emb_mask)
            next_tok = torch.argmax(torch.softmax(pred[:, -1, :], dim=1), dim=1)
            next_tok = torch.where(finished, torch.full_like(next_tok, EOS), next_tok)
            cur = torch.cat([cur, next_tok[:, None]], dim=1)
            finished |= (next_tok == EOS)
            if finished.all():
                break
        return cur[:, 1:], pred

    def mass_oob_predonly(self, pred_token, precursors, window_lower, window_upper, mz1=None):
        peps = self.m.tokenizer.detokenize(pred_token, trim_stop_token=True, join=True, trim_start_token=False)
        out = []
        for i, pep in enumerate(peps):
            aa_comp = dict(mass.std_aa_comp)
            aa_comp["C"] = aa_comp["C"] + mass.Composition("H3C2NO")  # CAM
            mz = self.m.calc_mz_simple(pep, int(precursors[i,1].item()), aa_comp)
            if not np.isfinite(mz):
                out.append(i); continue
            if mz1 is not None:
                if abs(mz - mz1[i].item()) >= 1.0:
                    out.append(i)
            else:
                if not (window_lower[i]-1 <= mz <= window_upper[i]+1):
                    out.append(i)
        return out

    @torch.no_grad()
    def refine_candidate_by_mass_predonly(self, probs, topk, emb0, emb0_mask, cand_mz_enc,
                                         precursors, window_lower, window_upper, mz1=None):
        B, N = probs.shape
        topk_idx = torch.topk(probs, topk, dim=1).indices  # (B,K)
        sel = torch.zeros((B, N), dtype=torch.bool, device=probs.device)
        sel.scatter_(1, topk_idx, True)

        pred_token, pred = None, None
        for r in range(topk):
            candidate_bin = sel.to(dtype=probs.dtype)
            emb, emb_mask, _, _ = self.m._run_cross_attn_with_candidate(emb0, emb0_mask, cand_mz_enc, candidate_bin)
            pred_token, pred = self.greedy_decode_len(emb, emb_mask, precursors, max_len=self.max_decode_len)
            oob = self.mass_oob_predonly(pred_token, precursors, window_lower, window_upper, mz1)
            if len(oob) == 0:
                break
            if r >= topk - 1:
                break
            for i in oob:
                cur_idx = topk_idx[i]
                sel[i, cur_idx] = False
                sel[i, cur_idx[r]] = True
        return sel.to(dtype=probs.dtype), pred_token, pred

    @torch.no_grad()
    def predict_batch(self, batch):
        # batch: label-free format from predict-collate
        spectra, precursors, _, frac, _, _, _, _, window_upper, window_lower, mz1, meta = batch

        emb0, emb0_mask = self.m.spectrum_encoder(spectra)

        if self.m.train_phase == "teacher":
            raise RuntimeError("Predict requires sampler-phase checkpoint (train_phase != 'teacher').")

        candidate_bin, probs, logits, extras, prior_gate_value = self.m.topk_theo_sampler(emb=emb0, emb_mask=emb0_mask)

        mz_full = self.m.mz_bin_centers_200_300.unsqueeze(0).expand(spectra.size(0), -1).to(self.m.device, self.m.dtype)
        cand_mz_enc = self.m.mz_encoder(mz_full)

        cand_ref, pred_token, pred = self.refine_candidate_by_mass_predonly(
            probs=probs, topk=self.m.topk, emb0=emb0, emb0_mask=emb0_mask, cand_mz_enc=cand_mz_enc,
            precursors=precursors, window_lower=window_lower, window_upper=window_upper, mz1=mz1
        )

        pep_pred = self.m.tokenizer.detokenize(pred_token, trim_stop_token=True, join=True, trim_start_token=False)

        return {
            "peptide_pred": pep_pred,
            "pred_token": pred_token.detach().cpu(),
            "candidate_bin": cand_ref.detach().cpu(),
            "meta": meta,
            "frac": frac.detach().cpu() if torch.is_tensor(frac) else frac,
        }
