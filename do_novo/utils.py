"""Methods to evaluate peptide-spectrum predictions."""
from typing import Dict, Iterable, List, Tuple
import re
import numpy as np
import pyopenms as oms
import torch
import torch.nn.functional as F
from spectrum_utils.utils import mass_diff
from itertools import combinations_with_replacement

import torch

import torch

def sample_theo_mz_candidates(
    theo_mz_bin: torch.Tensor,        # (B, n_bins) 0/1
    missing_mz_bin: torch.Tensor,     # (B, n_bins) 0/1
    peptide_ids: torch.Tensor,        # (B,)
    topK: int,
    pos_missing_prob: float = 0.7,
    pos_exist_prob: float = 0.3,
) -> torch.Tensor:
    device = theo_mz_bin.device
    B, n_bins = theo_mz_bin.shape
    theo = theo_mz_bin.bool()
    miss = missing_mz_bin.bool()

    pos_missing = miss                          # theo 중 missing
    pos_exist   = theo & ~miss                  # theo 중 observed

    same = peptide_ids[:, None] == peptide_ids[None, :]
    other_theo = ((~same).float() @ theo.float()) > 0
    neg_inb = other_theo & ~theo
    neg_pure = ~(pos_missing | pos_exist | neg_inb)

    out = torch.zeros((B, n_bins), device=device, dtype=torch.bool)
    all_idx = torch.arange(n_bins, device=device)

    def pick(idx: torch.Tensor, k: int) -> torch.Tensor:
        if k <= 0 or idx.numel() == 0:
            return idx.new_empty((0,), dtype=torch.long)
        k = min(k, idx.numel())
        return idx[torch.randperm(idx.numel(), device=device)[:k]]

    for b in range(B):
        used = torch.zeros(n_bins, device=device, dtype=torch.bool)
        chosen = []

        pm = pos_missing[b] & (torch.rand(n_bins, device=device) < pos_missing_prob)
        pe = pos_exist[b]   & (torch.rand(n_bins, device=device) < pos_exist_prob)

        pos_idx = torch.cat([pm.nonzero().squeeze(1), pe.nonzero().squeeze(1)])
        if pos_idx.numel() > 0:
            pos_idx = pos_idx[torch.randperm(pos_idx.numel(), device=device)]
            pos_idx = pos_idx[:min(topK, pos_idx.numel())]
            used[pos_idx] = True
            chosen.append(pos_idx)

        cur = int(sum(x.numel() for x in chosen))
        R = topK - cur
        if R > 0:
            need_inb = (R + 1) // 2
            need_pure = R // 2

            inb_idx = neg_inb[b].nonzero().squeeze(1)
            pure_idx = all_idx[neg_pure[b]]

            c_inb = pick(inb_idx[~used[inb_idx]], need_inb)
            if c_inb.numel():
                used[c_inb] = True
                chosen.append(c_inb)

            cur = int(sum(x.numel() for x in chosen))
            R = topK - cur
            need_pure = R
            c_pure = pick(pure_idx[~used[pure_idx]], need_pure)
            if c_pure.numel():
                used[c_pure] = True
                chosen.append(c_pure)

            cur = int(sum(x.numel() for x in chosen))
            R = topK - cur
            if R > 0:
                c_inb2 = pick(inb_idx[~used[inb_idx]], R)
                if c_inb2.numel():
                    used[c_inb2] = True
                    chosen.append(c_inb2)
                    
        cur = int(sum(x.numel() for x in chosen))
        if cur < topK:
            rest = all_idx[~used]
            c = pick(rest, topK - cur)
            if c.numel():
                chosen.append(c)

        cand = torch.cat(chosen)[:topK] if chosen else all_idx[:0]
        out[b, cand] = True

    return out.float()



def topk_missing_metrics(score, gt_mask, k=8):
    B, n_bins = score.shape
    device = score.device

    k = min(k, n_bins)

    gt_mask_bool = gt_mask.bool()              # (B, n_bins)
    valid = gt_mask_bool.sum(dim=1) > 0        # (B,)

    if not valid.any():
        zero = torch.tensor(0.0, device=device)
        return zero, zero, zero

    score_v = score[valid]                     # (B_valid, n_bins)
    gt_v    = gt_mask_bool[valid]              # (B_valid, n_bins)

    # Top-K index
    _, topk_idx = torch.topk(score_v, k, dim=1)  # (B_valid, k)

    hits = gt_v.gather(1, topk_idx).float()      # (B_valid, k)

    hit_any = (hits.sum(dim=1) > 0).float()      # (B_valid,)
    hit_rate = hit_any.mean()
    
    true_counts = gt_v.sum(dim=1).float()        # (B_valid,)
    recall = (hits.sum(dim=1) / true_counts).mean()
    
    _, top1_idx = torch.topk(score_v, 1, dim=1)  # (B_valid, k)
    hits_top1 = gt_v.gather(1, top1_idx).float()
    hit_top1_any = (hits_top1.sum(dim=1) > 0).float()
    hit_top1_rate = hit_top1_any.mean()

    return hit_rate, recall, hit_top1_rate



def sample_missing_candidates(
    missing_mz_bin: torch.Tensor,
    peptide_ids: torch.Tensor,
    pos_keep_prob: float = 0.7,
    inbatch_neg_prob: float = 0.2,
    min_k: int | None = None,
) -> torch.Tensor:
    
    device = missing_mz_bin.device
    B, n_bins = missing_mz_bin.shape

    pos_mask = (missing_mz_bin == 1)   # (B, n_bins)

    rand_pos    = torch.rand_like(missing_mz_bin, dtype=torch.float, device=device)
    sampled_pos = pos_mask & (rand_pos < pos_keep_prob)   # (B, n_bins)

    same_label = (peptide_ids.unsqueeze(0) == peptide_ids.unsqueeze(1))  # (B, B)
    diff_label = ~same_label                                            # (B, B)

    diff_label_f = diff_label.to(torch.float32)          # (B, B)
    pos_mask_f   = pos_mask.to(torch.float32)            # (B, n_bins)

    other_pos_count = diff_label_f @ pos_mask_f          # (B, n_bins)
    other_pos_mask  = (other_pos_count > 0.0)            # (B, n_bins)

    inbatch_neg_mask = other_pos_mask & (~pos_mask)      # (B, n_bins)

    rand_neg    = torch.rand_like(missing_mz_bin, dtype=torch.float, device=device)
    sampled_inb_neg = inbatch_neg_mask & (rand_neg < inbatch_neg_prob)

    candidate_mask = torch.zeros_like(missing_mz_bin, dtype=torch.bool, device=device)

    for b in range(B):
        pos_idx_all = sampled_pos[b].nonzero(as_tuple=False).squeeze(-1)   # (N_pos,)
        if min_k is None:
            inb_idx_all = sampled_inb_neg[b].nonzero(as_tuple=False).squeeze(-1)
            cand_idx = pos_idx_all
            if inb_idx_all.numel() > 0:
                cand_idx = torch.unique(torch.cat([cand_idx, inb_idx_all]))
            if cand_idx.numel() > 0:
                candidate_mask[b, cand_idx] = True
            continue

        K = min_k

        if pos_idx_all.numel() > K:
            perm = torch.randperm(pos_idx_all.numel(), device=device)
            pos_idx = pos_idx_all[perm[:K]]
            cand_idx = pos_idx
            candidate_mask[b, cand_idx] = True
            continue
        else:
            pos_idx = pos_idx_all
            cand_idx = pos_idx.clone()  # (<=K,)

        remaining = K - cand_idx.numel()
        if remaining <= 0:
            candidate_mask[b, cand_idx] = True
            continue

        inb_idx_all = sampled_inb_neg[b].nonzero(as_tuple=False).squeeze(-1)

        pure_neg_mask = (~pos_mask[b]) & (~other_pos_mask[b])   # (n_bins,)
        pure_idx_all = pure_neg_mask.nonzero(as_tuple=False).squeeze(-1)

        target_inb = remaining // 2
        target_pure = remaining - target_inb

        avail_inb = inb_idx_all.numel()
        avail_pure = pure_idx_all.numel()

        use_inb = min(target_inb, avail_inb)
        use_pure = min(target_pure, avail_pure)

        used = use_inb + use_pure
        still_need = remaining - used

        if still_need > 0 and avail_inb > use_inb:
            extra_inb = min(still_need, avail_inb - use_inb)
            use_inb += extra_inb
            used += extra_inb
            still_need = remaining - used

        if still_need > 0 and avail_pure > use_pure:
            extra_pure = min(still_need, avail_pure - use_pure)
            use_pure += extra_pure
            used += extra_pure
            still_need = remaining - used

        if use_inb > 0:
            perm_inb = torch.randperm(avail_inb, device=device)
            chosen_inb = inb_idx_all[perm_inb[:use_inb]]
            cand_idx = torch.cat([cand_idx, chosen_inb])

        if use_pure > 0:
            perm_pure = torch.randperm(avail_pure, device=device)
            chosen_pure = pure_idx_all[perm_pure[:use_pure]]
            cand_idx = torch.cat([cand_idx, chosen_pure])

        if cand_idx.numel() > 0:
            cand_idx = torch.unique(cand_idx)
            candidate_mask[b, cand_idx] = True

    candidate_bin = candidate_mask.float()
    return candidate_bin.to(device)




def compute_missing_prec_rec(missing_mz_bin, samples):
    with torch.no_grad():
        # target & pred mask
        target = (missing_mz_bin > 0.5)          # (B, n_bins) bool
        pred   = (samples > 0.5)                 # (B, n_bins) bool

        tp = (pred & target).sum(dim=1).float()  # (B,)
        fp = (pred & ~target).sum(dim=1).float()
        fn = (~pred & target).sum(dim=1).float()

        precision = tp / (tp + fp + 1e-8)
        recall    = tp / (tp + fn + 1e-8)

        miss_prec = precision.mean()
        miss_rec  = recall.mean()
        n_true    = target.sum(dim=1).float().mean()
        n_pred    = pred.sum(dim=1).float().mean()
        n_tp      = tp.mean()

        return miss_prec, miss_rec, n_true, n_pred, n_tp




def seq_sets_from_top1(best_idx_flat, lengths, combos):
    seq_sets = []
    offset = 0

    for n in lengths:
        idx_slice = best_idx_flat[offset:offset+n]
        offset += n

        seq_sets.append(
            {seq for idx in idx_slice.tolist() if idx >= 0 for seq in combos[idx]}
        )

    return seq_sets

def batch_top1_idx_ppm(selected_list, masses, tolerance_ppm=20):
    device = masses.device
    lengths = [mz.numel() for mz in selected_list]
    N = sum(lengths)

    if N == 0:
        return torch.empty(0, dtype=torch.long, device=device), lengths

    mz_flat = torch.cat([mz.to(device) for mz in selected_list], dim=0)  # (N,)
    tol = mz_flat * tolerance_ppm * 1e-6                                # (N,)

    diff = (mz_flat.unsqueeze(-1) - masses.unsqueeze(0)).abs()          # (N, M)

    big = torch.finfo(diff.dtype).max
    mask = diff <= tol.unsqueeze(-1)                                    # (N, M)
    diff_masked = diff.masked_fill(~mask, big)                          # (N, M)

    best_idx = diff_masked.argmin(dim=1)                                # (N,)

    has_any = mask.any(dim=1)                                           # (N,)
    best_idx = best_idx.masked_fill(~has_any, -1)

    return best_idx, lengths


def prepare_mass_index(mass_to_combo, device=None, dtype=torch.float32):
    masses = list(mass_to_combo.keys())
    combos = [mass_to_combo[m] for m in masses]
    masses_tensor = torch.tensor(masses, device=device, dtype=dtype)
    return masses_tensor, combos


def make_mass_to_combo_dict(residues, max_len=3, ion_type="neutral", round_decimals=6, min_mass=200.0):
    PROTON = 1.007825      # H+
    H2O    = 18.010565     # H2O

    if ion_type == "neutral":
        offset = 0.0
    elif ion_type == "b":
        offset = PROTON
    elif ion_type == "y":
        offset = PROTON + H2O
    else:
        raise ValueError(f"Unknown ion_type: {ion_type}")

    mass_to_combo = {}
    aas = list(residues.keys())

    for L in range(1, max_len + 1):
        for combo in combinations_with_replacement(aas, L):
            total_mass = sum(residues[a] for a in combo) + offset
            if total_mass < min_mass:
                continue

            key = "".join(sorted(combo))
            total_mass = round(total_mass, round_decimals)

            mass_to_combo.setdefault(total_mass, []).append(key)

    return mass_to_combo


def compute_topk_precision_recall(
    missing_mz_logits,
    missing_mz_bin,
    k=50,
    threshold=0.5,
):
    
    probs = torch.sigmoid(missing_mz_logits)          # (B, N_bins)
    topk_vals, topk_idx = probs.topk(k, dim=1)        # (B, k)

    mask = topk_vals >= threshold                     # (B, k), bool

    gt_at_topk_no_mask = missing_mz_bin.gather(1, topk_idx)   # (B, k), 0/1

    gt_at_topk = gt_at_topk_no_mask * mask.int()

    TP = gt_at_topk.sum(dim=1)                        # (B,)

    pred_cnt = mask.sum(dim=1)                        # (B,)

    true_missing = missing_mz_bin.sum(dim=1)          # (B,)

    precision = torch.zeros_like(TP, dtype=torch.float32)
    nonzero_pred = pred_cnt > 0
    precision[nonzero_pred] = TP[nonzero_pred] / pred_cnt[nonzero_pred]

    recall = torch.zeros_like(TP, dtype=torch.float32)
    nonzero_true = true_missing > 0
    recall[nonzero_true] = TP[nonzero_true] / true_missing[nonzero_true]

    topk_true_cnt = gt_at_topk_no_mask.sum(dim=1)     # (B,)

    recall_in_topk_truth = torch.zeros_like(TP, dtype=torch.float32)
    nonzero_topk_true = topk_true_cnt > 0
    recall_in_topk_truth[nonzero_topk_true] = (
        TP[nonzero_topk_true] / topk_true_cnt[nonzero_topk_true]
    )

    return precision.mean(), recall.mean(), recall_in_topk_truth.mean(), TP.mean()



def compute_missing_precision_recall(missing_mz_logits, missing_mz_bin, threshold=0.5):

    probs = torch.sigmoid(missing_mz_logits)

    preds = (probs >= threshold).float()        # (B, L)

    true  = missing_mz_bin.float()              # (B, L)
    
    preds = preds.flatten()
    true  = true.flatten()

    TP = torch.sum((preds == 1) & (true == 1)).float()
    FP = torch.sum((preds == 1) & (true == 0)).float()
    FN = torch.sum((preds == 0) & (true == 1)).float()

    precision = TP / (TP + FP + 1e-8)
    recall    = TP / (TP + FN + 1e-8)

    return precision, recall, torch.sum(preds == 1).float()


def mz_to_grid_idx(mz_array, mz_min, bins_per_da, l_grid):

    mz_array = np.asarray(mz_array, dtype=np.float32)
    idx = np.floor((mz_array - mz_min) * bins_per_da).astype(int)
    valid = (idx >= 0) & (idx < l_grid)
    return idx[valid]

def build_grid(mz_array, mz_min, mz_max, bins_per_da):
    
    l_grid = int((mz_max - mz_min) * bins_per_da)

    target = np.zeros(l_grid, dtype=np.float32)
    if len(mz_array) == 0:
        return target
    idx = mz_to_grid_idx(mz_array, mz_min, bins_per_da, l_grid)
    target[idx] = 1.0
    return target



def seq_to_token(seq: str):
    tokens = []
    m = re.match(r"^([+-]\d+\.\d+(?:[+-]\d+\.\d+)*)", seq)
    if m:
        tokens.append(m.group(1))
        seq = seq[m.end():]

    tokens += re.findall(r"[A-Z][+-]\d+\.\d+|[A-Z]", seq)
    return tokens

def convert_peptide(peptide):
    reverse_mod_dict = {
        "Q+0.984": "Q[+0.984]",
        "M+15.995": "M[+15.995]",
        "E-18.011": "E[-18.011]",
        "C-17.027": "C[-17.027]",
        "+43.006": "[+43.006]",
        }

    tokens = seq_to_token(peptide)
    new_tokens = [reverse_mod_dict[tok] if tok in reverse_mod_dict else tok for tok in tokens]

    return "".join(new_tokens)

def make_theoretical_by_ions(peptide):
    peptide = convert_peptide(peptide)
    seq = oms.AASequence.fromString(peptide)

    theo_b, theo_y = [], []
    theo_frag = {}
    for z in (1,2):
        for i in range(1, seq.size()):
            b_mz = seq.getPrefix(i).getMonoWeight(oms.Residue.ResidueType.BIon, z)
            y_mz = seq.getSuffix(i).getMonoWeight(oms.Residue.ResidueType.YIon, z)
            theo_b.append(b_mz)
            theo_y.append(y_mz)
            theo_frag[b_mz] = [f"b{i}^{z}", f"{str(seq.getPrefix(i))}"]
            theo_frag[y_mz] = [f"y{i}^{z}", f"{str(seq.getSuffix(i))}"]
    return np.concatenate([np.asarray(theo_b, float), np.asarray(theo_y, float)]), theo_frag


def get_init_aug_feat(init_mass_pred, spectra):
    device = spectra.device
    dtype  = spectra.dtype
    B      = init_mass_pred.shape[0]

    init_mass_aug = init_mass_pred.to(device=device, dtype=dtype).unsqueeze(-1)
    init_intensity_aug = spectra[:, :, 1].median(dim=1).values.to(device=device, dtype=dtype).unsqueeze(-1)
    init_rt_aug = torch.zeros(B, 1, device=device, dtype=dtype)
    init_level_aug = torch.full((B, 1), 2.0, device=device, dtype=dtype)

    init_aug_feat = torch.cat(
        [init_mass_aug, init_intensity_aug, init_rt_aug, init_level_aug],
        dim=-1
    ).unsqueeze(1)  # (B,2,4)

    return init_aug_feat


def scale_intensity(intensity: np.array) -> np.array:
    return np.power(intensity, 1 / 2).astype(np.float32)


def scale_to_unit_norm(intensity: np.array) -> np.array:
    return intensity / np.sqrt(intensity**2).sum()


def aa_acc_matrics(preds_seqs, true_seqs):
    correct_aas = torch.logical_or(preds_seqs == true_seqs, true_seqs == 0)
    padding = torch.sum(true_seqs == 0)
    aa_acc = (torch.sum(correct_aas.float()) - padding) / torch.sum(true_seqs != 0)

    correct_peps = torch.mean(correct_aas.float(), dim=1) == 1
    pep_acc = torch.mean(correct_peps.float())

    return aa_acc, pep_acc


def frag_match_matrics(pred_frags, frag_labels):

    pred  = torch.argmax(pred_frags, dim=1)
    label = frag_labels.flatten()

    TP = ((pred==1) & (label==1)).sum()
    FP = ((pred==1) & (label==0)).sum()
    FN = ((pred==0) & (label==1)).sum()

    precision = TP / (TP + FP)
    recall    = TP / (TP + FN)

    return precision, recall


def aa_match_prefix(
    peptide1: List[str],
    peptide2: List[str],
    aa_dict: Dict[str, float],
    cum_mass_threshold: float = 0.5,
    ind_mass_threshold: float = 0.1,
) -> Tuple[np.ndarray, bool]:
    """
    Find the matching prefix amino acids between two peptide sequences.

    This is a similar evaluation criterion as used by DeepNovo.

    Parameters
    ----------
    peptide1 : List[str]
        The first tokenized peptide sequence to be compared.
    peptide2 : List[str]
        The second tokenized peptide sequence to be compared.
    aa_dict : Dict[str, float]
        Mapping of amino acid tokens to their mass values.
    cum_mass_threshold : float
        Mass threshold in Dalton to accept cumulative mass-matching amino acid
        sequences.
    ind_mass_threshold : float
        Mass threshold in Dalton to accept individual mass-matching amino acids.

    Returns
    -------
    aa_matches : np.ndarray of length max(len(peptide1), len(peptide2))
        Boolean flag indicating whether each paired-up amino acid matches across
        both peptide sequences.
    pep_match : bool
        Boolean flag to indicate whether the two peptide sequences fully match.
    """
    aa_matches = np.zeros(max(len(peptide1), len(peptide2)), np.bool_)
    # Find longest mass-matching prefix.
    i1, i2, cum_mass1, cum_mass2 = 0, 0, 0.0, 0.0
    while i1 < len(peptide1) and i2 < len(peptide2):
        aa_mass1 = aa_dict.get(peptide1[i1], 0)
        aa_mass2 = aa_dict.get(peptide2[i2], 0)
        if (
            abs(mass_diff(cum_mass1 + aa_mass1, cum_mass2 + aa_mass2, True))
            < cum_mass_threshold
        ):
            aa_matches[max(i1, i2)] = (
                abs(mass_diff(aa_mass1, aa_mass2, True)) < ind_mass_threshold
            )
            i1, i2 = i1 + 1, i2 + 1
            cum_mass1, cum_mass2 = cum_mass1 + aa_mass1, cum_mass2 + aa_mass2
        elif cum_mass2 + aa_mass2 > cum_mass1 + aa_mass1:
            i1, cum_mass1 = i1 + 1, cum_mass1 + aa_mass1
        else:
            i2, cum_mass2 = i2 + 1, cum_mass2 + aa_mass2
    return aa_matches, aa_matches.all()


def aa_match_prefix_suffix(
    peptide1: List[str],
    peptide2: List[str],
    aa_dict: Dict[str, float],
    cum_mass_threshold: float = 0.5,
    ind_mass_threshold: float = 0.1,
) -> Tuple[np.ndarray, bool]:
    """
    Find the matching prefix and suffix amino acids between two peptide
    sequences.

    Parameters
    ----------
    peptide1 : List[str]
        The first tokenized peptide sequence to be compared.
    peptide2 : List[str]
        The second tokenized peptide sequence to be compared.
    aa_dict : Dict[str, float]
        Mapping of amino acid tokens to their mass values.
    cum_mass_threshold : float
        Mass threshold in Dalton to accept cumulative mass-matching amino acid
        sequences.
    ind_mass_threshold : float
        Mass threshold in Dalton to accept individual mass-matching amino acids.

    Returns
    -------
    aa_matches : np.ndarray of length max(len(peptide1), len(peptide2))
        Boolean flag indicating whether each paired-up amino acid matches across
        both peptide sequences.
    pep_match : bool
        Boolean flag to indicate whether the two peptide sequences fully match.
    """
    # Find longest mass-matching prefix.
    aa_matches, pep_match = aa_match_prefix(
        peptide1, peptide2, aa_dict, cum_mass_threshold, ind_mass_threshold
    )
    # No need to evaluate the suffixes if the sequences already fully match.
    if pep_match:
        return aa_matches, pep_match
    # Find longest mass-matching suffix.
    i1, i2 = len(peptide1) - 1, len(peptide2) - 1
    i_stop = np.argwhere(~aa_matches)[0]
    cum_mass1, cum_mass2 = 0.0, 0.0
    while i1 >= i_stop and i2 >= i_stop:
        aa_mass1 = aa_dict.get(peptide1[i1], 0)
        aa_mass2 = aa_dict.get(peptide2[i2], 0)
        if (
            abs(mass_diff(cum_mass1 + aa_mass1, cum_mass2 + aa_mass2, True))
            < cum_mass_threshold
        ):
            aa_matches[max(i1, i2)] = (
                abs(mass_diff(aa_mass1, aa_mass2, True)) < ind_mass_threshold
            )
            i1, i2 = i1 - 1, i2 - 1
            cum_mass1, cum_mass2 = cum_mass1 + aa_mass1, cum_mass2 + aa_mass2
        elif cum_mass2 + aa_mass2 > cum_mass1 + aa_mass1:
            i1, cum_mass1 = i1 - 1, cum_mass1 + aa_mass1
        else:
            i2, cum_mass2 = i2 - 1, cum_mass2 + aa_mass2
    return aa_matches, aa_matches.all()


def aa_match(
    peptide1: List[str],
    peptide2: List[str],
    aa_dict: Dict[str, float],
    cum_mass_threshold: float = 0.5,
    ind_mass_threshold: float = 0.1,
    mode: str = "best",
) -> Tuple[np.ndarray, bool]:
    """
    Find the matching amino acids between two peptide sequences.

    Parameters
    ----------
    peptide1 : List[str]
        The first tokenized peptide sequence to be compared.
    peptide2 : List[str]
        The second tokenized peptide sequence to be compared.
    aa_dict : Dict[str, float]
        Mapping of amino acid tokens to their mass values.
    cum_mass_threshold : float
        Mass threshold in Dalton to accept cumulative mass-matching amino acid
        sequences.
    ind_mass_threshold : float
        Mass threshold in Dalton to accept individual mass-matching amino acids.
    mode : {"best", "forward", "backward"}
        The direction in which to find matching amino acids.

    Returns
    -------
    aa_matches : np.ndarray of length max(len(peptide1), len(peptide2))
        Boolean flag indicating whether each paired-up amino acid matches across
        both peptide sequences.
    pep_match : bool
        Boolean flag to indicate whether the two peptide sequences fully match.
    """
    if mode == "best":
        return aa_match_prefix_suffix(
            peptide1, peptide2, aa_dict, cum_mass_threshold, ind_mass_threshold
        )
    elif mode == "forward":
        return aa_match_prefix(
            peptide1, peptide2, aa_dict, cum_mass_threshold, ind_mass_threshold
        )
    elif mode == "backward":
        aa_matches, pep_match = aa_match_prefix(
            list(reversed(peptide1)),
            list(reversed(peptide2)),
            aa_dict,
            cum_mass_threshold,
            ind_mass_threshold,
        )
        return aa_matches[::-1], pep_match
    else:
        raise ValueError("Unknown evaluation mode")


def aa_match_batch(
    peptides1: Iterable,
    peptides2: Iterable,
    aa_dict: Dict[str, float],
    cum_mass_threshold: float = 0.5,
    ind_mass_threshold: float = 0.1,
    mode: str = "best",
) -> Tuple[List[Tuple[np.ndarray, bool]], int, int]:
    """
    Find the matching amino acids between multiple pairs of peptide sequences.

    Parameters
    ----------
    peptides1 : Iterable
        The first list of peptide sequences to be compared.
    peptides2 : Iterable
        The second list of peptide sequences to be compared.
    aa_dict : Dict[str, float]
        Mapping of amino acid tokens to their mass values.
    cum_mass_threshold : float
        Mass threshold in Dalton to accept cumulative mass-matching amino acid
        sequences.
    ind_mass_threshold : float
        Mass threshold in Dalton to accept individual mass-matching amino acids.
    mode : {"best", "forward", "backward"}
        The direction in which to find matching amino acids.

    Returns
    -------
    aa_matches_batch : List[Tuple[np.ndarray, bool]]
        For each pair of peptide sequences: (i) boolean flags indicating whether
        each paired-up amino acid matches across both peptide sequences, (ii)
        boolean flag to indicate whether the two peptide sequences fully match.
    n_aa1: int
        Total number of amino acids in the first list of peptide sequences.
    n_aa2: int
        Total number of amino acids in the second list of peptide sequences.
    """
    aa_matches_batch, n_aa1, n_aa2 = [], 0, 0
    for peptide1, peptide2 in zip(peptides1, peptides2):
        # Split peptides into individual AAs if necessary.
        if isinstance(peptide1, str):
            peptide1 = re.split(r"(?<=.)(?=[A-Z])", peptide1)
        if isinstance(peptide2, str):
            peptide2 = re.split(r"(?<=.)(?=[A-Z])", peptide2)
        n_aa1, n_aa2 = n_aa1 + len(peptide1), n_aa2 + len(peptide2)
        aa_matches_batch.append(
            aa_match(
                peptide1,
                peptide2,
                aa_dict,
                cum_mass_threshold,
                ind_mass_threshold,
                mode,
            )
        )
    return aa_matches_batch, n_aa1, n_aa2


def aa_match_metrics(
    aa_matches_batch: List[Tuple[np.ndarray, bool]],
    n_aa_true: int,
    n_aa_pred: int,
) -> Tuple[float, float, float]:
    """
    Calculate amino acid and peptide-level evaluation metrics.

    Parameters
    ----------
    aa_matches_batch : List[Tuple[np.ndarray, bool]]
        For each pair of peptide sequences: (i) boolean flags indicating whether
        each paired-up amino acid matches across both peptide sequences, (ii)
        boolean flag to indicate whether the two peptide sequences fully match.
    n_aa_true: int
        Total number of amino acids in the true peptide sequences.
    n_aa_pred: int
        Total number of amino acids in the predicted peptide sequences.

    Returns
    -------
    aa_precision: float
        The number of correct AA predictions divided by the number of predicted
        AAs.
    aa_recall: float
        The number of correct AA predictions divided by the number of true AAs.
    pep_recall: float
        The number of correct peptide predictions divided by the number of
        peptides.
    """
    n_aa_correct = sum(
        [aa_matches[0].sum() for aa_matches in aa_matches_batch]
    )
    aa_precision = n_aa_correct / (n_aa_pred + 1e-8)
    aa_recall = n_aa_correct / (n_aa_true + 1e-8)
    pep_recall = sum([aa_matches[1] for aa_matches in aa_matches_batch]) / (
        len(aa_matches_batch) + 1e-8
    )
    return aa_precision, aa_recall, pep_recall


def aa_precision_recall(
    aa_scores_correct: List[float],
    aa_scores_all: List[float],
    n_aa_total: int,
    threshold: float,
) -> Tuple[float, float]:
    """
    Calculate amino acid level precision and recall at a given score threshold.

    Parameters
    ----------
    aa_scores_correct : List[float]
        Amino acids scores for the correct amino acids predictions.
    aa_scores_all : List[float]
        Amino acid scores for all amino acids predictions.
    n_aa_total : int
        The total number of amino acids in the predicted peptide sequences.
    threshold : float
        The amino acid score threshold.

    Returns
    -------
    aa_precision: float
        The number of correct amino acid predictions divided by the number of
        predicted amino acids.
    aa_recall: float
        The number of correct amino acid predictions divided by the total number
        of amino acids.
    """
    n_aa_correct = sum([score > threshold for score in aa_scores_correct])
    n_aa_predicted = sum([score > threshold for score in aa_scores_all])
    return n_aa_correct / n_aa_predicted, n_aa_correct / n_aa_total




# PTM #

def _split_if_string(peptide, aa_dict: Dict[str, float]) -> List[str]:
    """
    Your pipeline uses detokenize(..., join=False), so peptide is usually a List[str].
    If somehow a string comes in, split by capital letters (fallback).
    """
    if isinstance(peptide, list):
        return peptide
    if isinstance(peptide, str):
        # fallback: split by capital letters; assumes tokens are single-letter or X[...]
        # If you might have multi-letter tokens, replace this with your split_peptide().
        return re.findall(r"[A-Z](?:\[[^\]]+\])?", peptide)
    return list(peptide)


def _infer_ptm_list_from_aa_dict(aa_dict: Dict[str, float]) -> List[str]:
    """
    PTM tokens are values like 'M[Oxidation]' or 'C[Carbamidomethyl]'.
    So we treat any token containing '[' and ']' as PTM token.
    """
    return [k for k in aa_dict.keys() if ("[" in k and "]" in k)]


def _ptm_match_prefix_suffix(
    peptide_true: List[str],
    peptide_pred: List[str],
    aa_dict: Dict[str, float],
    ptm_list: List[str],
    cum_mass_threshold: float = 0.5,
    ind_mass_threshold: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns boolean arrays (length=max(len(true),len(pred))):
      - ptm_correct_flags_for_recall: counts correct PTM among true PTMs
      - ptm_correct_flags_for_precision: counts correct PTM among predicted PTMs

    "Correct PTM" is STRICT: token equality (e.g., 'M[Oxidation]' == 'M[Oxidation]'),
    and only counted at AA-matched positions (mass-based alignment).
    """
    L = max(len(peptide_true), len(peptide_pred))
    aa_matched = np.zeros(L, dtype=np.bool_)
    ptm_correct_true = np.zeros(L, dtype=np.bool_)
    ptm_correct_pred = np.zeros(L, dtype=np.bool_)

    # ---------- prefix ----------
    i1, i2, cum1, cum2 = 0, 0, 0.0, 0.0
    while i1 < len(peptide_true) and i2 < len(peptide_pred):
        t_tok = peptide_true[i1]
        p_tok = peptide_pred[i2]
        m1 = aa_dict.get(t_tok, 0.0)
        m2 = aa_dict.get(p_tok, 0.0)

        if abs(mass_diff(cum1 + m1, cum2 + m2, True)) < cum_mass_threshold:
            idx = max(i1, i2)
            aa_ok = abs(mass_diff(m1, m2, True)) < ind_mass_threshold
            aa_matched[idx] = aa_ok

            if aa_ok:
                # strict PTM correctness: exact token match + both are PTM tokens
                ptm_ok = (t_tok == p_tok) and (t_tok in ptm_list)
                ptm_correct_true[idx] = ptm_ok
                ptm_correct_pred[idx] = ptm_ok

            i1 += 1
            i2 += 1
            cum1 += m1
            cum2 += m2
        elif cum2 + m2 > cum1 + m1:
            cum1 += m1
            i1 += 1
        else:
            cum2 += m2
            i2 += 1

    if aa_matched.all():
        return ptm_correct_true, ptm_correct_pred

    # ---------- suffix ----------
    stop_idx = int(np.argwhere(~aa_matched)[0])  # earliest mismatch boundary
    i1, i2 = len(peptide_true) - 1, len(peptide_pred) - 1
    cum1, cum2 = 0.0, 0.0

    while i1 >= stop_idx and i2 >= stop_idx:
        t_tok = peptide_true[i1]
        p_tok = peptide_pred[i2]
        m1 = aa_dict.get(t_tok, 0.0)
        m2 = aa_dict.get(p_tok, 0.0)

        if abs(mass_diff(cum1 + m1, cum2 + m2, True)) < cum_mass_threshold:
            idx = max(i1, i2)
            aa_ok = abs(mass_diff(m1, m2, True)) < ind_mass_threshold
            aa_matched[idx] = aa_ok

            if aa_ok:
                ptm_ok = (t_tok == p_tok) and (t_tok in ptm_list)
                ptm_correct_true[idx] = ptm_ok
                ptm_correct_pred[idx] = ptm_ok

            i1 -= 1
            i2 -= 1
            cum1 += m1
            cum2 += m2
        elif cum2 + m2 > cum1 + m1:
            cum1 += m1
            i1 -= 1
        else:
            cum2 += m2
            i2 -= 1

    return ptm_correct_true, ptm_correct_pred


def ptm_aa_match_batch(
    peptides_pred: Iterable,
    peptides_true: Iterable,
    aa_dict: Dict[str, float],
    ptm_list: List[str] | None = None,
    cum_mass_threshold: float = 0.5,
    ind_mass_threshold: float = 0.1,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], int, int]:
    """
    Returns:
      - ptm_batch: list of (ptm_correct_true_flags, ptm_correct_pred_flags)
      - n_ptm_true: total number of PTM tokens in true peptides
      - n_ptm_pred: total number of PTM tokens in predicted peptides
    """
    if ptm_list is None:
        ptm_list = _infer_ptm_list_from_aa_dict(aa_dict)

    ptm_batch = []
    n_ptm_true = 0
    n_ptm_pred = 0

    for pred, true in zip(peptides_pred, peptides_true):
        pred_tok = _split_if_string(pred, aa_dict)
        true_tok = _split_if_string(true, aa_dict)

        n_ptm_true += sum(t in ptm_list for t in true_tok)
        n_ptm_pred += sum(p in ptm_list for p in pred_tok)

        if len(pred_tok) == 0 or len(true_tok) == 0:
            L = max(len(true_tok), len(pred_tok), 1)
            ptm_batch.append((np.zeros(L, dtype=np.bool_), np.zeros(L, dtype=np.bool_)))
            continue

        ptm_true_flags, ptm_pred_flags = _ptm_match_prefix_suffix(
            peptide_true=true_tok,
            peptide_pred=pred_tok,
            aa_dict=aa_dict,
            ptm_list=ptm_list,
            cum_mass_threshold=cum_mass_threshold,
            ind_mass_threshold=ind_mass_threshold,
        )
        ptm_batch.append((ptm_true_flags, ptm_pred_flags))

    return ptm_batch, n_ptm_true, n_ptm_pred


def ptm_aa_match_metrics(
    ptm_batch: List[Tuple[np.ndarray, np.ndarray]],
    n_ptm_true: int,
    n_ptm_pred: int,
) -> Tuple[float, float]:
    """
    ptm_aa_precision = (#correct PTM predictions) / (#pred PTM tokens)
    ptm_aa_recall    = (#correct PTM predictions) / (#true PTM tokens)
    """
    correct_true = int(sum(x[0].sum() for x in ptm_batch))
    correct_pred = int(sum(x[1].sum() for x in ptm_batch))
    ptm_aa_precision = correct_pred / (n_ptm_pred + 1e-8)
    ptm_aa_recall = correct_true / (n_ptm_true + 1e-8)
    return ptm_aa_precision, ptm_aa_recall
