"""Extract the single verified white-box JSON bundle for Phase 2 Pair B (`/`).

Computes all cohort and per-frame quantities over real model weights, activations,
gradients, spatially local Layer 49 (`layer4`) ablations, equal-height 24 px strip
occlusion, and exact full-129,809-train cosine kNN for both the fine-tuned WILDS
ResNet-50 and the frozen ImageNet-1K linear-probe baseline.

Output: `phase2/pair_b/results/camera156_whitebox.json`
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from common import load_config, resolve
from common.data import get_transform, load_class_names, normalize, set_seed
from common.model import get_device, load_model, load_probe_model


SA_WHITE_BAR_RAW_LOCS = {82, 121, 484, 174, 434}
CAM0_RAW_LOC = 0
CAM186_RAW_LOC = 333
C156_RAW_LOC = 282

COMMON_NAMES = {
    "empty": "Empty",
    "sylvilagus brasiliensis": "Brazilian rabbit",
    "cuniculus paca": "Lowland paca",
    "cerdocyon thous": "Crab-eating fox",
    "mazama sp": "Brocket deer",
    "mazama americana": "Red brocket deer",
    "mazama gouazoubira": "Grey brocket deer",
    "mazama temama": "Central American red brocket",
    "mus minutoides": "African pygmy mouse",
    "ortalis vetula": "Plain chachalaca",
    "meleagris ocellata": "Ocellated turkey",
    "argusianus argus": "Great argus",
    "didelphis sp": "Opossum",
    "leopardus pardalis": "Ocelot",
    "leopardus wiedii": "Margay",
    "pecari tajacu": "Collared peccary",
    "tayassu pecari": "White-lipped peccary",
    "francolinus nobilis": "Handsome francolin",
    "cricetomys gambianus": "Gambian pouched rat",
    "cercopithecus mitis": "Blue monkey",
    "cercopithecus lhoesti": "L'Hoest's monkey",
    "pan troglodytes": "Chimpanzee",
    "claravis pretiosa": "Blue ground dove",
    "dasyprocta punctata": "Central American agouti",
    "dasyprocta fuliginosa": "Black agouti",
    "panthera onca": "Jaguar",
    "tapirus terrestris": "South American tapir",
    "didelphis marsupialis": "Common opossum",
    "nasua nasua": "South American coati",
    "tinamus major": "Great tinamou",
    "funisciurus carruthersi": "Mountain squirrel",
    "cephalophus nigrifrons": "Black-fronted duiker",
    "cephalophus silvicultor": "Yellow-backed duiker",
    "muntiacus muntjak": "Red muntjac",
    "prionailurus bengalensis": "Leopard cat",
    "helarctos malayanus": "Sun bear",
    "panthera tigris": "Tiger",
    "paguma larvata": "Masked palm civet",
    "leptotila plumbeiceps": "Grey-headed dove",
    "geotrygon montana": "Ruddy quail-dove",
    "crax rubra": "Great curassow",
    "penelope purpurascens": "Crested guan",
    "eira barbara": "Tayra",
    "puma concolor": "Puma",
    "procyon cancrivorus": "Crab-eating raccoon",
    "dasypus novemcinctus": "Nine-banded armadillo",
    "myoprocta pratti": "Green acouchi",
    "sciurus granatensis": "Red-tailed squirrel",
    "sciurus spadiceus": "Southern Amazon red squirrel",
    "atherurus africanus": "African brush-tailed porcupine",
    "genetta servalina": "Servaline genet",
    "bovidae": "Wild bovid",
    "rodentia": "Rodent",
    "aves": "Bird",
}

# Visually verified normalized [x0, y0, x1, y1] bounding boxes for non-empty neighbor frames
# where the animal is visibly present (empty sequence tails / obscured frames intentionally omitted).
VERIFIED_ANIMAL_BBOXES = {
    "8dbaf818-21bc-11ea-a13a-137349068a90.jpg": [0.00, 0.45, 0.09, 0.61],
    "9147cf38-21bc-11ea-a13a-137349068a90.jpg": [0.00, 0.50, 0.14, 0.66],
    "922286aa-21bc-11ea-a13a-137349068a90.jpg": [0.58, 0.45, 0.70, 0.60],
    "97517dde-21bc-11ea-a13a-137349068a90.jpg": [0.16, 0.45, 0.30, 0.60],
    "8bec2d54-21bc-11ea-a13a-137349068a90.jpg": [0.00, 0.44, 0.15, 0.61],
    "94fa6f1e-21bc-11ea-a13a-137349068a90.jpg": [0.00, 0.43, 0.12, 0.61],
    "95597450-21bc-11ea-a13a-137349068a90.jpg": [0.20, 0.66, 0.37, 0.90],
    "92405a36-21bc-11ea-a13a-137349068a90.jpg": [0.73, 0.47, 0.87, 0.65],
    "8bbfda92-21bc-11ea-a13a-137349068a90.jpg": [0.67, 0.42, 0.82, 0.59],
    "8daf3a3c-21bc-11ea-a13a-137349068a90.jpg": [0.32, 0.47, 0.51, 0.69],
    "8fe578de-21bc-11ea-a13a-137349068a90.jpg": [0.50, 0.36, 0.68, 0.51],
    "92b8ed20-21bc-11ea-a13a-137349068a90.jpg": [0.89, 0.39, 1.00, 0.55],
    "97a9afd6-21bc-11ea-a13a-137349068a90.jpg": [0.17, 0.39, 0.36, 0.58],
    "98619362-21bc-11ea-a13a-137349068a90.jpg": [0.00, 0.61, 0.25, 0.88],
    "98c0ae7e-21bc-11ea-a13a-137349068a90.jpg": [0.39, 0.36, 0.47, 0.51],
    "88bf258c-21bc-11ea-a13a-137349068a90.jpg": [0.40, 0.37, 1.00, 0.84],
    "93add902-21bc-11ea-a13a-137349068a90.jpg": [0.60, 0.38, 1.00, 0.94],
    "94b9be06-21bc-11ea-a13a-137349068a90.jpg": [0.29, 0.00, 1.00, 0.94],
    "97df1770-21bc-11ea-a13a-137349068a90.jpg": [0.64, 0.31, 1.00, 0.72],
    "8ff3cae2-21bc-11ea-a13a-137349068a90.jpg": [0.00, 0.16, 0.31, 0.68],
    "9240106c-21bc-11ea-a13a-137349068a90.jpg": [0.84, 0.76, 1.00, 0.94],
    "869da008-21bc-11ea-a13a-137349068a90.jpg": [0.04, 0.04, 0.75, 0.77],
    "91d773b8-21bc-11ea-a13a-137349068a90.jpg": [0.62, 0.19, 1.00, 0.94],
    "97c3f88c-21bc-11ea-a13a-137349068a90.jpg": [0.76, 0.14, 0.99, 0.30],
    "89dd0196-21bc-11ea-a13a-137349068a90.jpg": [0.11, 0.44, 0.25, 0.59],
    "8a22629a-21bc-11ea-a13a-137349068a90.jpg": [0.15, 0.44, 0.36, 0.60],
    "9128b918-21bc-11ea-a13a-137349068a90.jpg": [0.11, 0.46, 0.23, 0.59],
    "925ee898-21bc-11ea-a13a-137349068a90.jpg": [0.11, 0.45, 0.23, 0.60],
    "9318bfca-21bc-11ea-a13a-137349068a90.jpg": [0.40, 0.46, 0.62, 0.62],
}


def common_name(sci: str) -> str:
    s = sci.strip().lower()
    if s in COMMON_NAMES:
        return COMMON_NAMES[s]
    return sci.strip().capitalize()


def sanitize_json(obj):
    if isinstance(obj, dict):
        return {k: sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_json(v) for v in obj]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if (math.isnan(v) or math.isinf(v)) else round(v, 5)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    return obj


def compute_linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    # Gram-free O(N * D^2) or O(N^2 * D) linear CKA; N <= 200 here so N x N is tiny & exact
    K = Xc @ Xc.T
    L = Yc @ Yc.T
    hsic = float(np.sum(K * L))
    denom = float(np.linalg.norm(K) * np.linalg.norm(L))
    return hsic / denom if denom > 1e-12 else 0.0


def main() -> None:
    t0 = time.time()
    cfg = load_config()
    set_seed(cfg["seed"])
    device = get_device(cfg)
    print(f"[extract] Device: {device}", flush=True)

    preds = pd.read_parquet(resolve(cfg["paths"]["predictions"]))
    names = load_class_names(cfg)
    img_dir = resolve(cfg["paths"]["data_dir"]) / "train"
    tf = get_transform(cfg)

    ft_model = load_model(cfg, device).eval()
    probe_model = load_probe_model(cfg, device).eval()
    for mod in (ft_model, probe_model):
        for m in mod.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = False

    w_fc = ft_model.fc.weight.detach().to(device)  # [182, 2048]
    b_fc = ft_model.fc.bias.detach().to(device)    # [182]

    # Datasets
    c156_emp = preds[(preds["location"] == C156_RAW_LOC) & (preds["label"] == 0)].reset_index(drop=True)
    c156_ani = preds[(preds["location"] == C156_RAW_LOC) & (preds["label"] != 0)].reset_index(drop=True)
    test_other = preds[(preds["split"] == "test") & (preds["location"] != C156_RAW_LOC)]
    other_ani = test_other[test_other["label"] != 0].sample(n=500, random_state=42).reset_index(drop=True)
    other_emp = test_other[test_other["label"] == 0].sample(n=200, random_state=42).reset_index(drop=True)

    def load_u8(df: pd.DataFrame) -> tuple[torch.Tensor, np.ndarray]:
        arr = torch.empty((len(df), 3, 448, 448), dtype=torch.uint8)
        ir = np.zeros(len(df), dtype=bool)
        for i, fn in enumerate(df["filename"]):
            im = tf(Image.open(img_dir / fn).convert("RGB"))
            arr[i] = im
            mid = im[:, 24:416, :].float()
            ir[i] = ((mid[0] - mid[1]).abs().mean() + (mid[1] - mid[2]).abs().mean()).item() < 2.0
        return arr, ir

    u8_emp, ir_emp = load_u8(c156_emp)
    u8_ani, ir_ani = load_u8(c156_ani)
    u8_oani, _ = load_u8(other_ani)
    u8_oemp, _ = load_u8(other_emp)

    u8_emp_cov = u8_emp.clone()
    u8_emp_cov[:, :, 424:, :] = 0
    u8_ani_cov = u8_ani.clone()
    u8_ani_cov[:, :, 424:, :] = 0
    u8_oani_cov = u8_oani.clone()
    u8_oani_cov[:, :, 424:, :] = 0
    u8_oemp_cov = u8_oemp.clone()
    u8_oemp_cov[:, :, 424:, :] = 0

    cohort_masks = {
        "all": np.ones(len(c156_emp), dtype=bool),
        "night": ir_emp.copy(),
        "day": (~ir_emp).copy(),
    }
    print(
        f"[extract] Loaded images in {time.time() - t0:.1f}s "
        f"(all={len(c156_emp)}, night={ir_emp.sum()}, day={(~ir_emp).sum()}, c156_ani={len(c156_ani)})",
        flush=True,
    )

    # =========================================================================
    # 1. LAYER 49 (layer4, 14x14) MAPS & ALL 5 INTERVENTION MODES + FROZEN PROBE
    # =========================================================================
    l4_holder = {}
    h_l4 = ft_model.layer4.register_forward_hook(lambda _m, _i, o: l4_holder.__setitem__("x", o.detach()))

    @torch.no_grad()
    def extract_l4_and_logits(u8_t: torch.Tensor, bs: int = 32) -> tuple[torch.Tensor, np.ndarray]:
        a_list, z_list = [], []
        for s in range(0, len(u8_t), bs):
            z = ft_model(normalize(u8_t[s : s + bs].to(device)))
            a_list.append(l4_holder["x"].cpu())
            z_list.append(z.cpu().numpy())
        return torch.cat(a_list, 0), np.concatenate(z_list, 0)

    A4_emp, z_emp_base = extract_l4_and_logits(u8_emp)
    A4_emp_cov, z_emp_cov = extract_l4_and_logits(u8_emp_cov)
    A4_ani, z_ani_base = extract_l4_and_logits(u8_ani)
    A4_ani_cov, z_ani_cov = extract_l4_and_logits(u8_ani_cov)
    A4_oani, z_oani_base = extract_l4_and_logits(u8_oani)
    _, z_oani_cov = extract_l4_and_logits(u8_oani_cov)
    A4_oemp, z_oemp_base = extract_l4_and_logits(u8_oemp)
    _, z_oemp_cov = extract_l4_and_logits(u8_oemp_cov)
    h_l4.remove()

    # Frozen ImageNet probe logits & embeddings
    p_bb = probe_model[0]
    p_l4_holder = {}
    h_pl4 = p_bb[7].register_forward_hook(lambda _m, _i, o: p_l4_holder.__setitem__("x", o.detach()))

    @torch.no_grad()
    def extract_probe(u8_t: torch.Tensor, bs: int = 32) -> tuple[torch.Tensor, np.ndarray]:
        f_list, z_list = [], []
        for s in range(0, len(u8_t), bs):
            z = probe_model(normalize(u8_t[s : s + bs].to(device)))
            f_list.append(p_l4_holder["x"].mean((2, 3)).cpu())
            z_list.append(z.cpu().numpy())
        return torch.cat(f_list, 0), np.concatenate(z_list, 0)

    f_emp_frozen, z_emp_frozen = extract_probe(u8_emp)
    _, z_ani_frozen = extract_probe(u8_ani)
    _, z_oani_frozen = extract_probe(u8_oani)
    _, z_oemp_frozen = extract_probe(u8_oemp)
    h_pl4.remove()

    # Exact 14-row additive logit profile (Layer 49) per cohort
    with torch.no_grad():
        row_vec_emp = A4_emp.mean(dim=3).to(device) / 14.0  # [698, 2048, 14]
        row_logits_emp = torch.einsum("ncu,kc->nuk", row_vec_emp, w_fc).cpu().numpy()  # [698, 14, 182]
        row_norm_emp = A4_emp.norm(dim=1).mean(dim=2).numpy()  # [698, 14]
        row_norm_oemp = A4_oemp.norm(dim=1).mean(dim=2).numpy()  # [200, 14]

    def pool_rows(A4_cpu: torch.Tensor, zero_rows: list[int], renorm: bool = True) -> torch.Tensor:
        A = A4_cpu.to(device)
        keep = torch.ones(14, dtype=torch.bool, device=device)
        for r in zero_rows:
            keep[r] = False
        if renorm:
            return A[:, :, keep, :].mean(dim=(2, 3))
        return A[:, :, keep, :].sum(dim=(2, 3)) / 196.0

    # Single-row zeroing (renorm) empty% for tooltips on each of the 14 rows
    single_row_emp_preds = {}
    with torch.no_grad():
        for u in range(14):
            f_u = pool_rows(A4_emp, [u], renorm=True)
            p_u = (f_u @ w_fc.T + b_fc).argmax(1).cpu().numpy()
            single_row_emp_preds[u] = p_u

    layer49_rows_by_cohort = {}
    for cname, mask in cohort_masks.items():
        rows_out = []
        for u in range(14):
            z0u = float(row_logits_emp[mask, u, 0].mean())
            z18u = float(row_logits_emp[mask, u, 18].mean())
            z3u = float(row_logits_emp[mask, u, 3].mean())
            n1 = float(row_norm_emp[mask, u].mean())
            n2 = float(row_norm_oemp[:, u].mean())
            rows_out.append({
                "row": u,
                "y0": u * 32,
                "y1": (u + 1) * 32,
                "z_empty": z0u,
                "z_rabbit": z18u,
                "z_paca": z3u,
                "margin_rabbit": z18u - z0u,
                "margin_paca": z3u - z0u,
                "norm_c156": n1,
                "norm_ctrl": n2,
                "norm_ratio": n1 / n2 if n2 > 1e-8 else 1.0,
                "zero_single_empty_pct": 100.0 * float((single_row_emp_preds[u][mask] == 0).mean()),
            })
        layer49_rows_by_cohort[cname] = rows_out

    # Compute the 5 unified intervention states (using renorm mean for layer49 row zeroing; also check avgpl for tooltip)
    INTERVENTIONS = [
        ("baseline", "Baseline", "Original photo & untouched Layer 49", []),
        ("cover_bar", "Cover bar (24 px input)", "Input rows 424–448 blacked out", None),
        ("zero_r13", "Zero row 13", "Layer 49 row 13 set to 0 (renorm mean, photo untouched)", [13]),
        ("zero_r12_13", "Zero rows 12–13", "Layer 49 rows 12–13 set to 0 (renorm mean, photo untouched)", [12, 13]),
        ("zero_r6_7", "Control: rows 6–7", "Layer 49 middle rows 6–7 set to 0 (renorm mean, photo untouched)", [6, 7]),
    ]

    y_c156_ani = c156_ani["label"].to_numpy()
    y_oani = other_ani["label"].to_numpy()

    int_f_emp: dict[str, torch.Tensor] = {}
    int_z_emp: dict[str, np.ndarray] = {}
    int_z_emp_avgpl: dict[str, np.ndarray] = {}
    int_z_ani: dict[str, np.ndarray] = {}
    int_z_oani: dict[str, np.ndarray] = {}
    int_z_oemp: dict[str, np.ndarray] = {}

    with torch.no_grad():
        for key, _lbl, _desc, zrows in INTERVENTIONS:
            if key == "baseline":
                int_f_emp[key] = A4_emp.mean((2, 3))
                int_z_emp[key] = z_emp_base
                int_z_emp_avgpl[key] = z_emp_base
                int_z_ani[key] = z_ani_base
                int_z_oani[key] = z_oani_base
                int_z_oemp[key] = z_oemp_base
            elif key == "cover_bar":
                int_f_emp[key] = A4_emp_cov.mean((2, 3))
                int_z_emp[key] = z_emp_cov
                int_z_emp_avgpl[key] = z_emp_cov
                int_z_ani[key] = z_ani_cov
                int_z_oani[key] = z_oani_cov
                int_z_oemp[key] = z_oemp_cov
            else:
                fe = pool_rows(A4_emp, zrows, renorm=True)
                fe_avg = pool_rows(A4_emp, zrows, renorm=False)
                fa = pool_rows(A4_ani, zrows, renorm=True)
                foa = pool_rows(A4_oani, zrows, renorm=True)
                foe = pool_rows(A4_oemp, zrows, renorm=True)
                int_f_emp[key] = fe.cpu()
                int_z_emp[key] = (fe @ w_fc.T + b_fc).cpu().numpy()
                int_z_emp_avgpl[key] = (fe_avg @ w_fc.T + b_fc).cpu().numpy()
                int_z_ani[key] = (fa @ w_fc.T + b_fc).cpu().numpy()
                int_z_oani[key] = (foa @ w_fc.T + b_fc).cpu().numpy()
                int_z_oemp[key] = (foe @ w_fc.T + b_fc).cpu().numpy()

    base_oani_ok = z_oani_base.argmax(1) == y_oani

    def summarize_ani_and_controls(z_a: np.ndarray, z_oa: np.ndarray, z_oe: np.ndarray) -> dict:
        pa = z_a.argmax(1)
        ranks = [int((z_a[i] > z_a[i, y_c156_ani[i]]).sum()) + 1 for i in range(len(y_c156_ani))]
        z_no0 = z_a.copy()
        z_no0[:, 0] = -1e9
        p_no0 = z_no0.argmax(1)
        top_no0_vc = pd.Series(p_no0).value_counts().head(3)
        top_no0_list = [f"{common_name(names[int(k)])} ({int(v)})" for k, v in top_no0_vc.items()]

        poa = z_oa.argmax(1)
        ok = poa == y_oani
        lost = int((base_oani_ok & (~ok)).sum())
        gained = int(((~base_oani_ok) & ok).sum())
        poe = z_oe.argmax(1)
        return {
            "c156_ani_n": int(len(y_c156_ani)),
            "c156_ani_rab_paca": int(np.isin(pa, [18, 3]).sum()),
            "c156_ani_empty": int((pa == 0).sum()),
            "c156_ani_true": int((pa == y_c156_ani).sum()),
            "c156_ani_med_true_rank": float(np.median(ranks)),
            "c156_ani_nonempty_rab_paca": int(np.isin(p_no0, [18, 3]).sum()),
            "c156_ani_top_nonempty": top_no0_list,
            "other_ani_n": int(len(y_oani)),
            "other_ani_cams": int(other_ani["location"].nunique()),
            "other_ani_acc_pct": 100.0 * float(ok.mean()),
            "other_ani_delta_pp": 100.0 * float(ok.mean() - base_oani_ok.mean()),
            "other_ani_lost": lost,
            "other_ani_gained": gained,
            "other_ani_to_empty_pct": 100.0 * float((poa == 0).mean()),
            "other_emp_acc_pct": 100.0 * float((poe == 0).mean()),
        }

    # =========================================================================
    # 2. FULL 129,809-TRAIN COSINE kNN (FT embeddings.npy & Frozen embeddings_imagenet.npy)
    # =========================================================================
    print("[extract] Querying full 129,809-train cosine kNN...", flush=True)
    train_mask = (preds["split"] == "train").to_numpy()
    train_idx = np.where(train_mask)[0]
    train_meta = preds.iloc[train_idx].reset_index(drop=True)
    tr_locs = train_meta["location"].to_numpy()
    tr_loc_rem = train_meta["location_remapped"].to_numpy()
    tr_y = train_meta["label"].to_numpy()
    tr_fn = train_meta["filename"].to_numpy()

    # Precompute camera metadata table for all train cameras
    cam_empty_rate = (
        train_meta.groupby("location")["label"].apply(lambda s: 100.0 * float((s == 0).mean())).to_dict()
    )
    cam_top_species = (
        train_meta.groupby("location")["label"].apply(lambda s: int(s.value_counts().index[0])).to_dict()
    )
    cam_counts_total = train_meta["location"].value_counts().to_dict()

    tr_is_sa_wb = np.isin(tr_locs, list(SA_WHITE_BAR_RAW_LOCS))
    tr_is_c0 = tr_locs == CAM0_RAW_LOC
    tr_is_c186 = tr_locs == CAM186_RAW_LOC
    tr_is_empty = tr_y == 0
    tr_is_wb = tr_is_sa_wb | tr_is_c0

    def run_full_knn(emb_path: Path, query_dict: dict[str, torch.Tensor]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        mmap = np.load(emb_path, mmap_mode="r")
        K = torch.from_numpy(np.array(mmap[train_idx], dtype=np.float32)).to(device)
        K = F.normalize(K, dim=1)
        out = {}
        with torch.no_grad():
            for qk, qf in query_dict.items():
                Q = F.normalize(qf.to(device), dim=1)
                idx_chunks, sim_chunks = [], []
                for s in range(0, len(Q), 128):
                    sims = Q[s : s + 128] @ K.T
                    vals, inds = torch.topk(sims, k=5, dim=1)
                    idx_chunks.append(inds.cpu().numpy())
                    sim_chunks.append(vals.cpu().numpy())
                out[qk] = (np.concatenate(idx_chunks, 0), np.concatenate(sim_chunks, 0))
        del K
        torch.cuda.empty_cache()
        return out

    tr_by_cls = {int(c): np.where(tr_y == c)[0] for c in np.unique(tr_y)}

    def run_pred_class_knn(
        emb_path: Path, query_dict: dict[str, torch.Tensor], pred_dict: dict[str, np.ndarray]
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """ExMatchina-style top-5 cosine neighbors restricted to each query's predicted class."""
        mmap = np.load(emb_path, mmap_mode="r")
        K_all = F.normalize(torch.from_numpy(np.array(mmap[train_idx], dtype=np.float32)).to(device), dim=1)
        out = {}
        with torch.no_grad():
            for qk, qf in query_dict.items():
                Q = F.normalize(qf.to(device), dim=1)
                p_arr = pred_dict[qk]
                N = len(p_arr)
                ti = np.zeros((N, 5), dtype=np.int64)
                ts = np.zeros((N, 5), dtype=np.float32)
                for cls_id in np.unique(p_arr):
                    q_rows = np.where(p_arr == cls_id)[0]
                    cand = tr_by_cls.get(int(cls_id))
                    if cand is None or len(cand) < 5:
                        cand_t, K_sub = None, K_all
                    else:
                        cand_t, K_sub = cand, K_all[torch.from_numpy(cand).to(device)]
                    Q_sub = Q[torch.from_numpy(q_rows).to(device)]
                    for s in range(0, len(Q_sub), 128):
                        sims = Q_sub[s : s + 128] @ K_sub.T
                        vals, inds = torch.topk(sims, k=5, dim=1)
                        inds_np = inds.cpu().numpy()
                        if cand_t is not None:
                            inds_np = cand_t[inds_np]
                        ti[q_rows[s : s + 128]] = inds_np
                        ts[q_rows[s : s + 128]] = vals.cpu().numpy()
                out[qk] = (ti, ts)
        del K_all
        torch.cuda.empty_cache()
        return out

    ft_knn = run_full_knn(resolve(cfg["paths"]["embeddings"]), int_f_emp)
    frozen_knn = run_full_knn(resolve(cfg["paths"]["embeddings_imagenet"]), {"frozen": f_emp_frozen})
    ft_knn_pred = run_pred_class_knn(
        resolve(cfg["paths"]["embeddings"]), int_f_emp, {k: z.argmax(1) for k, z in int_z_emp.items()}
    )
    frozen_knn_pred = run_pred_class_knn(
        resolve(cfg["paths"]["embeddings_imagenet"]), {"frozen": f_emp_frozen}, {"frozen": z_emp_frozen.argmax(1)}
    )

    def summarize_knn(ti: np.ndarray, ts: np.ndarray, mask: np.ndarray) -> dict:
        sub_i = ti[mask]
        sub_s = ts[mask]
        flat = sub_i.ravel()
        n_votes = len(flat)
        sa_wb = 100.0 * float(tr_is_sa_wb[flat].mean())
        c0 = 100.0 * float(tr_is_c0[flat].mean())
        c186 = 100.0 * float(tr_is_c186[flat].mean())
        norm_emp = 100.0 * float((tr_is_empty[flat] & (~tr_is_wb[flat])).mean())
        tot_emp = 100.0 * float(tr_is_empty[flat].mean())
        other_nonemp = max(0.0, 100.0 - sa_wb - c0 - c186 - norm_emp)
        y18 = 100.0 * float((tr_y[flat] == 18).mean())
        y3 = 100.0 * float((tr_y[flat] == 3).mean())

        vc = pd.Series(tr_locs[flat]).value_counts().head(6)
        top_cams = []
        for raw_loc, cnt in vc.items():
            rl = int(raw_loc)
            rem = int(train_meta.loc[train_meta["location"] == rl, "location_remapped"].iloc[0])
            top_cams.append({
                "camera_name": f"Camera {rem}",
                "location_remapped": rem,
                "location_raw": rl,
                "share_pct": 100.0 * float(cnt) / n_votes,
                "train_n": int(cam_counts_total[rl]),
                "empty_pct": float(cam_empty_rate[rl]),
                "has_white_bar": bool((rl in SA_WHITE_BAR_RAW_LOCS) or (rl == CAM0_RAW_LOC)),
                "is_sa_wb": bool(rl in SA_WHITE_BAR_RAW_LOCS),
                "top_species": common_name(names[cam_top_species[rl]]),
            })
        return {
            "n_votes": int(n_votes),
            "top1_cos": float(sub_s[:, 0].mean()),
            "sa_wb_pct": sa_wb,
            "cam0_pct": c0,
            "cam186_pct": c186,
            "normal_empty_pct": norm_emp,
            "total_empty_pct": tot_emp,
            "other_nonempty_pct": other_nonemp,
            "y18_vote_pct": y18,
            "y3_vote_pct": y3,
            "top_cameras": top_cams,
        }

    # Assemble interventions table per cohort
    interventions_meta = [
        {"id": key, "label": lbl, "description": desc, "zero_rows": zrows}
        for key, lbl, desc, zrows in INTERVENTIONS
    ]
    interventions_data = {}
    for key, _lbl, _desc, _zrows in INTERVENTIONS:
        shared_ctrl = summarize_ani_and_controls(int_z_ani[key], int_z_oani[key], int_z_oemp[key])
        per_cohort = {}
        Z = int_z_emp[key]
        Z_avg = int_z_emp_avgpl[key]
        P = Z.argmax(1)
        P_avg = Z_avg.argmax(1)
        ti, ts = ft_knn[key]
        tip, tsp = ft_knn_pred[key]
        for cname, mask in cohort_masks.items():
            per_cohort[cname] = {
                "n": int(mask.sum()),
                "empty_pct": 100.0 * float((P[mask] == 0).mean()),
                "rabbit_pct": 100.0 * float((P[mask] == 18).mean()),
                "paca_pct": 100.0 * float((P[mask] == 3).mean()),
                "other_pct": 100.0 * float((~np.isin(P[mask], [0, 18, 3])).mean()),
                "z_empty": float(Z[mask, 0].mean()),
                "z_rabbit": float(Z[mask, 18].mean()),
                "z_paca": float(Z[mask, 3].mean()),
                "avgpl_empty_pct": 100.0 * float((P_avg[mask] == 0).mean()),
                "avgpl_z_empty": float(Z_avg[mask, 0].mean()),
                "knn": summarize_knn(ti, ts, mask),
                "knn_pred": summarize_knn(tip, tsp, mask),
            }
        interventions_data[key] = {"controls": shared_ctrl, "cohorts": per_cohort}

    # Frozen ImageNet baseline summary (for bottom strip toggle & quiet reference)
    frozen_ctrl = summarize_ani_and_controls(z_ani_frozen, z_oani_frozen, z_oemp_frozen)
    frozen_cohorts = {}
    P_fr = z_emp_frozen.argmax(1)
    ti_fr, ts_fr = frozen_knn["frozen"]
    tip_fr, tsp_fr = frozen_knn_pred["frozen"]
    for cname, mask in cohort_masks.items():
        top_sp_vc = pd.Series(P_fr[mask]).value_counts().head(3)
        frozen_cohorts[cname] = {
            "n": int(mask.sum()),
            "empty_pct": 100.0 * float((P_fr[mask] == 0).mean()),
            "rabbit_pct": 100.0 * float((P_fr[mask] == 18).mean()),
            "paca_pct": 100.0 * float((P_fr[mask] == 3).mean()),
            "top_preds": [
                {"species": common_name(names[int(k)]), "pct": 100.0 * float(v) / int(mask.sum())}
                for k, v in top_sp_vc.items()
            ],
            "knn": summarize_knn(ti_fr, ts_fr, mask),
            "knn_pred": summarize_knn(tip_fr, tsp_fr, mask),
        }
    frozen_baseline = {"controls": frozen_ctrl, "cohorts": frozen_cohorts}

    # =========================================================================
    # 3. EQUAL-HEIGHT 24 PX HORIZONTAL STRIP-OCCLUSION SWEEP (18 STRIPS x 2 FILLS)
    # =========================================================================
    print("[extract] Running 18-strip equal-height 24px occlusion sweep on N=698...", flush=True)
    # Anchored to exact bottom bar y=424..448: 18 strips of 24px covering y=16..448
    img_means = u8_emp.float().mean(dim=(2, 3), keepdim=True).round().to(torch.uint8)  # [698, 3, 1, 1]
    strip_results_raw = {"black": [], "mean": []}

    @torch.no_grad()
    def eval_strip(y0: int, y1: int, fill_mode: str) -> tuple[np.ndarray, np.ndarray]:
        zs = []
        for s in range(0, len(u8_emp), 32):
            xb = u8_emp[s : s + 32].clone()
            if fill_mode == "black":
                xb[:, :, y0:y1, :] = 0
            else:
                xb[:, :, y0:y1, :] = img_means[s : s + 32]
            zs.append(ft_model(normalize(xb.to(device))).cpu().numpy())
        Z = np.concatenate(zs, 0)
        return Z[:, 0] - z_emp_base[:, 0], Z.argmax(1)

    for idx in range(18):
        y0 = 16 + idx * 24
        y1 = y0 + 24
        dz_b, p_b = eval_strip(y0, y1, "black")
        dz_m, p_m = eval_strip(y0, y1, "mean")
        strip_results_raw["black"].append((y0, y1, dz_b, p_b))
        strip_results_raw["mean"].append((y0, y1, dz_m, p_m))

    strips_by_cohort = {}
    for cname, mask in cohort_masks.items():
        s_list = []
        for idx in range(18):
            y0, y1, dz_b, p_b = strip_results_raw["black"][idx]
            _, _, dz_m, p_m = strip_results_raw["mean"][idx]
            s_list.append({
                "strip": idx,
                "y0": y0,
                "y1": y1,
                "is_bar": bool(idx == 17),
                "dz0_black": float(dz_b[mask].mean()),
                "dz0_mean": float(dz_m[mask].mean()),
                "empty_pct_black": 100.0 * float((p_b[mask] == 0).mean()),
                "empty_pct_mean": 100.0 * float((p_m[mask] == 0).mean()),
                "rabbit_pct_black": 100.0 * float((p_b[mask] == 18).mean()),
                "paca_pct_black": 100.0 * float((p_b[mask] == 3).mean()),
            })
        strips_by_cohort[cname] = s_list

    # =========================================================================
    # 4. COL 1: LAYERWISE CONTRASTIVE GRAD-CAM ACROSS LAYER 1, 10, 22, 40, 49
    #    (Fine-Tuned vs Frozen ImageNet Probe vs Normal Empty Control; Night vs Day!)
    #    At Layer 49: primary = rows 12..13 (exact RF spillover zone!), plus row 13 & rows 11..13 in tooltip.
    # =========================================================================
    print("[extract] Computing Layer 1, 10, 22, 40, 49 Contrastive Grad-CAM + CKA (FT vs Frozen Probe)...", flush=True)
    STAGE_SPECS = [
        {"id": "stem", "depth": 1, "label": "Layer 1", "torch_name": "stem", "grid": 112, "r0": 106, "r1": 112, "rows_label": "rows 106–111"},
        {"id": "layer1", "depth": 10, "label": "Layer 10", "torch_name": "layer1", "grid": 112, "r0": 106, "r1": 112, "rows_label": "rows 106–111"},
        {"id": "layer2", "depth": 22, "label": "Layer 22", "torch_name": "layer2", "grid": 56, "r0": 53, "r1": 56, "rows_label": "rows 53–55"},
        {"id": "layer3", "depth": 40, "label": "Layer 40", "torch_name": "layer3", "grid": 28, "r0": 26, "r1": 28, "rows_label": "rows 26–27"},
        {"id": "layer4", "depth": 49, "label": "Layer 49", "torch_name": "layer4", "grid": 14, "r0": 12, "r1": 14, "rows_label": "rows 12–13"},
    ]
    STAGE_IDS = [s["id"] for s in STAGE_SPECS]

    # Linear CKA FT vs Frozen ImageNet on a diverse 300-frame multi-camera cohort
    # (100 C156 empty + 100 normal empty + 100 other test animals; avoids single-camera Day/Night rank-1 collapse)
    cka_sub = torch.cat([u8_emp[::6][:100], u8_oemp[::2][:100], u8_oani[::5][:100]], dim=0)
    ft_pool_acts = {s: [] for s in STAGE_IDS}
    fr_pool_acts = {s: [] for s in STAGE_IDS}
    cur_ft, cur_fr = {}, {}
    hs_ft_cka = [
        ft_model.maxpool.register_forward_hook(lambda _m, _i, o: cur_ft.__setitem__("stem", o.detach().mean((2, 3)).cpu().numpy())),
        ft_model.layer1.register_forward_hook(lambda _m, _i, o: cur_ft.__setitem__("layer1", o.detach().mean((2, 3)).cpu().numpy())),
        ft_model.layer2.register_forward_hook(lambda _m, _i, o: cur_ft.__setitem__("layer2", o.detach().mean((2, 3)).cpu().numpy())),
        ft_model.layer3.register_forward_hook(lambda _m, _i, o: cur_ft.__setitem__("layer3", o.detach().mean((2, 3)).cpu().numpy())),
        ft_model.layer4.register_forward_hook(lambda _m, _i, o: cur_ft.__setitem__("layer4", o.detach().mean((2, 3)).cpu().numpy())),
    ]
    hs_fr_cka = [
        p_bb[3].register_forward_hook(lambda _m, _i, o: cur_fr.__setitem__("stem", o.detach().mean((2, 3)).cpu().numpy())),
        p_bb[4].register_forward_hook(lambda _m, _i, o: cur_fr.__setitem__("layer1", o.detach().mean((2, 3)).cpu().numpy())),
        p_bb[5].register_forward_hook(lambda _m, _i, o: cur_fr.__setitem__("layer2", o.detach().mean((2, 3)).cpu().numpy())),
        p_bb[6].register_forward_hook(lambda _m, _i, o: cur_fr.__setitem__("layer3", o.detach().mean((2, 3)).cpu().numpy())),
        p_bb[7].register_forward_hook(lambda _m, _i, o: cur_fr.__setitem__("layer4", o.detach().mean((2, 3)).cpu().numpy())),
    ]
    with torch.no_grad():
        for st in range(0, len(cka_sub), 32):
            xb = normalize(cka_sub[st : st + 32].to(device))
            _ = ft_model(xb)
            _ = probe_model(xb)
            for s in STAGE_IDS:
                ft_pool_acts[s].append(cur_ft[s])
                fr_pool_acts[s].append(cur_fr[s])
    for h in hs_ft_cka + hs_fr_cka:
        h.remove()
    cka_by_stage = {
        s: compute_linear_cka(np.concatenate(ft_pool_acts[s], 0), np.concatenate(fr_pool_acts[s], 0))
        for s in STAGE_IDS
    }

    # Gradient hook helper for both FT and Frozen Probe
    def compute_stage_gradcam_metrics(model: nn.Module, stage_modules: dict[str, nn.Module], u8_t: torch.Tensor, target_cls: int):
        """Compute per-frame Contrastive Grad-CAM (z_target - z_0), Standard ReLU clip%, and L2 energy share."""
        acts = {}

        def mk(sid: str):
            def _h(_m, _i, out):
                if out.requires_grad:
                    out.retain_grad()
                acts[sid] = out
            return _h

        hooks = [mod.register_forward_hook(mk(sid)) for sid, mod in stage_modules.items()]
        N = len(u8_t)
        res = {
            s["id"]: {
                "ctr_bar_num": np.zeros(N, dtype=np.float64),
                "ctr_tot_den": np.zeros(N, dtype=np.float64),
                "ctr_r13_num": np.zeros(N, dtype=np.float64),
                "ctr_r11_13_num": np.zeros(N, dtype=np.float64),
                "l2_bar_pct": np.zeros(N, dtype=np.float64),
                "std_clip_pct": np.zeros(N, dtype=np.float64),
            }
            for s in STAGE_SPECS
        }
        try:
            # 1) Contrastive pass: z_target - z_0 (also collects L2 energy share)
            for st in range(0, N, 16):
                ed = min(N, st + 16)
                xb = normalize(u8_t[st:ed].to(device)).detach().requires_grad_(True)
                model.zero_grad(set_to_none=True)
                z = model(xb)
                (z[:, target_cls] - z[:, 0]).sum().backward()
                for spec in STAGE_SPECS:
                    sid, r0, r1 = spec["id"], spec["r0"], spec["r1"]
                    A = acts[sid].detach()
                    G = acts[sid].grad.detach()
                    raw = (G.mean((2, 3), keepdim=True) * A).sum(1)  # [B, H, W]
                    rel = F.relu(raw)
                    res[sid]["ctr_bar_num"][st:ed] = rel[:, r0:r1, :].sum((1, 2)).cpu().numpy()
                    res[sid]["ctr_tot_den"][st:ed] = rel.sum((1, 2)).cpu().numpy()
                    if sid == "layer4":
                        res[sid]["ctr_r13_num"][st:ed] = rel[:, 13:14, :].sum((1, 2)).cpu().numpy()
                        res[sid]["ctr_r11_13_num"][st:ed] = rel[:, 11:14, :].sum((1, 2)).cpu().numpy()
                    l2_bar = A[:, :, r0:r1, :].pow(2).sum((1, 2, 3))
                    l2_tot = A.pow(2).sum((1, 2, 3)) + 1e-8
                    res[sid]["l2_bar_pct"][st:ed] = (100.0 * l2_bar / l2_tot).cpu().numpy()

            # 2) Standard single-class pass: z_target (just to measure ReLU clip% honestly!)
            for st in range(0, N, 16):
                ed = min(N, st + 16)
                xb = normalize(u8_t[st:ed].to(device)).detach().requires_grad_(True)
                model.zero_grad(set_to_none=True)
                z = model(xb)
                z[:, target_cls].sum().backward()
                for spec in STAGE_SPECS:
                    sid = spec["id"]
                    A = acts[sid].detach()
                    G = acts[sid].grad.detach()
                    raw = (G.mean((2, 3), keepdim=True) * A).sum(1)
                    res[sid]["std_clip_pct"][st:ed] = (100.0 * (raw <= 0).float().mean((1, 2))).cpu().numpy()
        finally:
            for h in hooks:
                h.remove()
        return res

    ft_mods = {
        "stem": ft_model.maxpool,
        "layer1": ft_model.layer1,
        "layer2": ft_model.layer2,
        "layer3": ft_model.layer3,
        "layer4": ft_model.layer4,
    }
    fr_mods = {
        "stem": p_bb[3],
        "layer1": p_bb[4],
        "layer2": p_bb[5],
        "layer3": p_bb[6],
        "layer4": p_bb[7],
    }

    # Compute for target=18 (Brazilian rabbit — dominant on All & Night IR) AND target=3 (Lowland paca — Day RGB)
    ft_c156_y18 = compute_stage_gradcam_metrics(ft_model, ft_mods, u8_emp, target_cls=18)
    ft_c156_y3 = compute_stage_gradcam_metrics(ft_model, ft_mods, u8_emp, target_cls=3)
    fr_c156_y18 = compute_stage_gradcam_metrics(probe_model, fr_mods, u8_emp, target_cls=18)
    fr_c156_y3 = compute_stage_gradcam_metrics(probe_model, fr_mods, u8_emp, target_cls=3)
    ft_ctrl_y18 = compute_stage_gradcam_metrics(ft_model, ft_mods, u8_oemp, target_cls=18)
    ft_ctrl_y3 = compute_stage_gradcam_metrics(ft_model, ft_mods, u8_oemp, target_cls=3)

    def agg_ratio(d: dict, mask: np.ndarray, num_key: str = "ctr_bar_num") -> float:
        num = float(d[num_key][mask].sum())
        den = float(d["ctr_tot_den"][mask].sum())
        return 100.0 * num / den if den > 1e-12 else 0.0

    # For "all" & "night": rabbit (y=18) vs empty; for "day": paca (y=3) vs empty (and store rabbit too in tooltip!)
    layer_trajectories = {}
    for cname, mask in cohort_masks.items():
        use_paca = cname == "day"
        ft_primary = ft_c156_y3 if use_paca else ft_c156_y18
        fr_primary = fr_c156_y3 if use_paca else fr_c156_y18
        ctrl_primary = ft_ctrl_y3 if use_paca else ft_ctrl_y18
        ft_alt = ft_c156_y18 if use_paca else ft_c156_y3
        ctrl_mask_all = np.ones(len(u8_oemp), dtype=bool)
        stages_out = []
        for spec in STAGE_SPECS:
            sid = spec["id"]
            unif = 100.0 * (spec["r1"] - spec["r0"]) / spec["grid"]
            row = {
                "id": sid,
                "depth": spec["depth"],
                "label": spec["label"],
                "grid": spec["grid"],
                "rows_label": spec["rows_label"],
                "uniform_pct": unif,
                "ft_ctr_pct": agg_ratio(ft_primary[sid], mask),
                "ft_ctr_alt_pct": agg_ratio(ft_alt[sid], mask),
                "frozen_ctr_pct": agg_ratio(fr_primary[sid], mask),
                "ctrl_ctr_pct": agg_ratio(ctrl_primary[sid], ctrl_mask_all),
                "ft_std_clip_pct": float(ft_primary[sid]["std_clip_pct"][mask].mean()),
                "ft_l2_bar_pct": float(ft_primary[sid]["l2_bar_pct"][mask].mean()),
                "frozen_l2_bar_pct": float(fr_primary[sid]["l2_bar_pct"][mask].mean()),
                "cka_vs_frozen": float(cka_by_stage[sid]),
            }
            if sid == "layer4":
                row["ft_ctr_r13_pct"] = agg_ratio(ft_primary[sid], mask, "ctr_r13_num")
                row["ft_ctr_r11_13_pct"] = agg_ratio(ft_primary[sid], mask, "ctr_r11_13_num")
                row["frozen_ctr_r13_pct"] = agg_ratio(fr_primary[sid], mask, "ctr_r13_num")
                row["frozen_ctr_r11_13_pct"] = agg_ratio(fr_primary[sid], mask, "ctr_r11_13_num")
                row["ctrl_ctr_r13_pct"] = agg_ratio(ctrl_primary[sid], ctrl_mask_all, "ctr_r13_num")
            stages_out.append(row)
        layer_trajectories[cname] = {
            "objective_label": "lowland paca − empty" if use_paca else "Brazilian rabbit − empty",
            "alt_objective_label": "Brazilian rabbit − empty" if use_paca else "lowland paca − empty",
            "stages": stages_out,
        }

    # =========================================================================
    # 5. CURATED CAMERA 156 SAMPLE FRAMES (Maps + Per-Frame Neighbors for ALL 6 States!)
    # =========================================================================
    print("[extract] Selecting curated Camera 156 frames & computing spatial maps + exact 129,809-train neighbors...", flush=True)
    # Pick 5 crisp representative frames:
    #   2 Night IR empty -> Brazilian rabbit (high confidence, cleanly flip under bot2 & cover_bar)
    #   1 Day RGB empty -> Lowland paca (high confidence, cleanly flips)
    #   1 Day RGB empty -> Brazilian rabbit (high confidence, cleanly flips)
    #   1 Real African animal on Camera 156 (hallucinates paca/rabbit -> flips to 0 hallucinations)
    probs_base = F.softmax(torch.from_numpy(z_emp_base), dim=1).numpy()
    p_base = z_emp_base.argmax(1)
    p_cov = z_emp_cov.argmax(1)
    p_b2 = int_z_emp["zero_r12_13"].argmax(1)
    clean_flip = (p_cov == 0) & (p_b2 == 0)

    def pick_emp_idx(cond_mask: np.ndarray, cls_id: int, k: int = 1, ranks: list[int] | None = None) -> list[int]:
        cand = np.where(cond_mask & clean_flip & (p_base == cls_id))[0]
        order = cand[np.argsort(-probs_base[cand, cls_id])]
        if ranks is not None:
            return [int(order[r]) for r in ranks if r < len(order)]
        return [int(x) for x in order[:k]]

    # Ranks 12 & 45 among clean-flip Night IR rabbit frames both exhibit ~15.5% Layer 22 bar focus
    # (above cohort mean 13.0%) alongside >50% rabbit confidence and 99.0-99.5% Layer 49 ReLU clipping.
    night_picks = pick_emp_idx(ir_emp, 18, k=2, ranks=[12, 45])
    day_paca_picks = pick_emp_idx(~ir_emp, 3, k=1)
    day_rab_picks = pick_emp_idx(~ir_emp, 18, k=1)

    # Pick 1 real animal frame on C156 that hallucinates paca/rabbit and flips off R+P
    probs_ani_base = F.softmax(torch.from_numpy(z_ani_base), dim=1).numpy()
    pa_base = z_ani_base.argmax(1)
    ani_cand = np.where(np.isin(pa_base, [18, 3]))[0]
    # Prefer francolinus nobilis (handsome francolin, visible bird) or cricetomys gambianus
    ani_scores = [
        (1 if y_c156_ani[i] == 114 else 0, float(probs_ani_base[i, pa_base[i]]), int(i))
        for i in ani_cand
    ]
    ani_scores.sort(reverse=True)
    ani_pick = ani_scores[0][2]

    curated_specs = [
        ("emp", night_picks[0], "Night IR · Empty #1", "night"),
        ("emp", night_picks[1], "Night IR · Empty #2", "night"),
        ("emp", day_paca_picks[0], "Day RGB · Empty #1", "day"),
        ("emp", day_rab_picks[0], "Day RGB · Empty #2", "day"),
        ("ani", ani_pick, f"{'Night IR' if ir_ani[ani_pick] else 'Day RGB'} · Real animal", "night" if ir_ani[ani_pick] else "day"),
    ]

    # Need exact 129,809-train kNN for the 1 real animal frame too (for the 5 FT modes + Frozen)
    ani_f_by_mode = {
        "baseline": A4_ani[ani_pick : ani_pick + 1].mean((2, 3)),
        "cover_bar": A4_ani_cov[ani_pick : ani_pick + 1].mean((2, 3)),
        "zero_r13": pool_rows(A4_ani[ani_pick : ani_pick + 1], [13], renorm=True).cpu(),
        "zero_r12_13": pool_rows(A4_ani[ani_pick : ani_pick + 1], [12, 13], renorm=True).cpu(),
        "zero_r6_7": pool_rows(A4_ani[ani_pick : ani_pick + 1], [6, 7], renorm=True).cpu(),
    }
    ani_ft_knn = run_full_knn(resolve(cfg["paths"]["embeddings"]), ani_f_by_mode)
    ani_ft_knn_pred = run_pred_class_knn(
        resolve(cfg["paths"]["embeddings"]),
        ani_f_by_mode,
        {k: int_z_ani[k][ani_pick : ani_pick + 1].argmax(1) for k in ani_f_by_mode},
    )
    # Frozen feature for the 1 real animal frame
    h_pl4_1 = p_bb[7].register_forward_hook(lambda _m, _i, o: p_l4_holder.__setitem__("x", o.detach()))
    with torch.no_grad():
        _ = probe_model(normalize(u8_ani[ani_pick : ani_pick + 1].to(device)))
        f_ani_frozen_1 = p_l4_holder["x"].mean((2, 3)).cpu()
    h_pl4_1.remove()
    ani_fr_knn = run_full_knn(resolve(cfg["paths"]["embeddings_imagenet"]), {"frozen": f_ani_frozen_1})
    ani_fr_knn_pred = run_pred_class_knn(
        resolve(cfg["paths"]["embeddings_imagenet"]),
        {"frozen": f_ani_frozen_1},
        {"frozen": z_ani_frozen[ani_pick : ani_pick + 1].argmax(1)},
    )

    def format_neighbors(inds: np.ndarray, sims: np.ndarray) -> list[dict]:
        out = []
        for idx_tr, sim in zip(inds, sims):
            i = int(idx_tr)
            rl = int(tr_locs[i])
            rem = int(tr_loc_rem[i])
            sci = names[int(tr_y[i])]
            out.append({
                "filename": str(tr_fn[i]),
                "camera_name": f"Camera {rem}",
                "location_remapped": rem,
                "location_raw": rl,
                "species_common": common_name(sci),
                "species_sci": sci,
                "is_empty": bool(tr_y[i] == 0),
                "has_white_bar": bool((rl in SA_WHITE_BAR_RAW_LOCS) or (rl == CAM0_RAW_LOC)),
                "is_sa_wb": bool(rl in SA_WHITE_BAR_RAW_LOCS),
                "is_cam0": bool(rl == CAM0_RAW_LOC),
                "is_cam186": bool(rl == CAM186_RAW_LOC),
                "cam_empty_pct": float(cam_empty_rate[rl]),
                "cos_sim": float(sim),
                "bbox": None if tr_y[i] == 0 else VERIFIED_ANIMAL_BBOXES.get(str(tr_fn[i])),
            })
        return out

    # Spatial maps helper for a single image
    def compute_frame_maps(u8_single: torch.Tensor, target_cls: int) -> dict:
        acts_s = {}

        def mk(s: str):
            def _h(_m, _i, o):
                if o.requires_grad:
                    o.retain_grad()
                acts_s[s] = o
            return _h

        hs = [
            ft_model.layer2.register_forward_hook(mk("l2")),
            ft_model.layer4.register_forward_hook(mk("l4")),
        ]
        try:
            xb = normalize(u8_single.unsqueeze(0).to(device)).detach().requires_grad_(True)
            ft_model.zero_grad(set_to_none=True)
            z = ft_model(xb)
            z[0, target_cls].backward()

            A2 = acts_s["l2"].detach()[0]
            G2 = acts_s["l2"].grad.detach()[0]
            raw2 = (G2.mean((1, 2), keepdim=True) * A2).sum(0)  # [56, 56]
            rel2 = F.relu(raw2).cpu().numpy()
            l2_bar_pct = 100.0 * float(rel2[53:56, :].sum() / (rel2.sum() + 1e-12))
            l2_norm = rel2 / (rel2.max() + 1e-12)
            l2_q = np.round(l2_norm * 255.0).astype(int).tolist()

            A4 = acts_s["l4"].detach()[0]
            G4 = acts_s["l4"].grad.detach()[0]
            raw4_std = (G4.mean((1, 2), keepdim=True) * A4).sum(0).cpu().numpy()  # [14, 14]
            rel4_std = np.maximum(raw4_std, 0.0)
            l4_std_clip_pct = 100.0 * float((raw4_std <= 0).mean())
            l4_std_max = float(rel4_std.max())
            l4_std_q = (
                np.round((rel4_std / l4_std_max) * 255.0).astype(int).tolist()
                if l4_std_max > 1e-9
                else np.zeros((14, 14), dtype=int).tolist()
            )

            # Exact Layer 49 signed contrastive margin: (w_target - w_0)^T A4[u,v] / 196
            w_diff = (w_fc[target_cls] - w_fc[0]).detach()
            raw4_ctr = (torch.einsum("chw,c->hw", A4, w_diff) / 196.0).cpu().numpy()
            rel4_ctr = np.maximum(raw4_ctr, 0.0)
            ctr_r12_13_pct = 100.0 * float(rel4_ctr[12:14, :].sum() / (rel4_ctr.sum() + 1e-12))
            ctr_r11_13_pct = 100.0 * float(rel4_ctr[11:14, :].sum() / (rel4_ctr.sum() + 1e-12))

            # Exact Layer 49 signed empty map: w_0^T A4[u,v] / 196
            raw4_emp = (torch.einsum("chw,c->hw", A4, w_fc[0].detach()) / 196.0).cpu().numpy()
        finally:
            for h in hs:
                h.remove()

        return {
            "target_cls": int(target_cls),
            "target_common": common_name(names[target_cls]),
            "l22_gradcam_u8": l2_q,
            "l22_bar_share_pct": l2_bar_pct,
            "l49_relu_u8": l4_std_q,
            "l49_relu_clip_pct": l4_std_clip_pct,
            "l49_relu_max": l4_std_max,
            "l49_contrastive": [[round(float(v), 4) for v in row] for row in raw4_ctr],
            "l49_ctr_r12_13_pct": ctr_r12_13_pct,
            "l49_ctr_r11_13_pct": ctr_r11_13_pct,
            "l49_empty_signed": [[round(float(v), 4) for v in row] for row in raw4_emp],
        }

    def pred_summary(z_vec: np.ndarray) -> dict:
        probs = F.softmax(torch.from_numpy(z_vec), dim=0).numpy()
        pid = int(np.argmax(z_vec))
        sci = names[pid]
        return {
            "pred_id": pid,
            "pred_common": common_name(sci),
            "pred_sci": sci,
            "prob_pct": 100.0 * float(probs[pid]),
            "is_empty": bool(pid == 0),
            "is_halluc": bool(pid in (18, 3)),
            "z_empty": float(z_vec[0]),
            "z_rabbit": float(z_vec[18]),
            "z_paca": float(z_vec[3]),
            "z_pred": float(z_vec[pid]),
        }

    curated_frames = []
    for fid, (src, idx, short_label, split_mode) in enumerate(curated_specs):
        if src == "emp":
            row_df = c156_emp.iloc[idx]
            u8_img = u8_emp[idx]
            gt_id = 0
            base_z = z_emp_base[idx]
        else:
            row_df = c156_ani.iloc[idx]
            u8_img = u8_ani[idx]
            gt_id = int(row_df["label"])
            base_z = z_ani_base[idx]

        halluc_cls = int(np.argmax(base_z))
        if halluc_cls not in (18, 3):
            halluc_cls = 18 if base_z[18] >= base_z[3] else 3
        maps = compute_frame_maps(u8_img, halluc_cls)

        states = {}
        for key, _lbl, _desc, _zrows in INTERVENTIONS:
            if src == "emp":
                zv = int_z_emp[key][idx]
                inds, sims = ft_knn[key][0][idx], ft_knn[key][1][idx]
                indsp, simsp = ft_knn_pred[key][0][idx], ft_knn_pred[key][1][idx]
            else:
                zv = int_z_ani[key][idx]
                inds, sims = ani_ft_knn[key][0][0], ani_ft_knn[key][1][0]
                indsp, simsp = ani_ft_knn_pred[key][0][0], ani_ft_knn_pred[key][1][0]
            st = pred_summary(zv)
            st["neighbors"] = format_neighbors(inds, sims)
            st["neighbors_pred"] = format_neighbors(indsp, simsp)
            states[key] = st

        # Frozen ImageNet baseline state for this frame
        zv_fr = z_emp_frozen[idx] if src == "emp" else z_ani_frozen[idx]
        if src == "emp":
            inds_fr, sims_fr = frozen_knn["frozen"][0][idx], frozen_knn["frozen"][1][idx]
            indsp_fr, simsp_fr = frozen_knn_pred["frozen"][0][idx], frozen_knn_pred["frozen"][1][idx]
        else:
            inds_fr, sims_fr = ani_fr_knn["frozen"][0][0], ani_fr_knn["frozen"][1][0]
            indsp_fr, simsp_fr = ani_fr_knn_pred["frozen"][0][0], ani_fr_knn_pred["frozen"][1][0]
        st_fr = pred_summary(zv_fr)
        st_fr["neighbors"] = format_neighbors(inds_fr, sims_fr)
        st_fr["neighbors_pred"] = format_neighbors(indsp_fr, simsp_fr)
        states["frozen"] = st_fr

        curated_frames.append({
            "id": f"frame_{fid}",
            "filename": str(row_df["filename"]),
            "label": short_label,
            "lighting": split_mode,
            "is_real_animal": bool(src == "ani"),
            "gt_id": gt_id,
            "gt_common": common_name(names[gt_id]),
            "gt_sci": names[gt_id],
            "maps": maps,
            "states": states,
        })

    # =========================================================================
    # 6. ASSEMBLE & SAVE ONE CLEAN JSON BUNDLE
    # =========================================================================
    bundle = {
        "meta": {
            "camera_name": "Camera 156",
            "location_remapped": 156,
            "location_raw": C156_RAW_LOC,
            "n_empty_all": int(len(c156_emp)),
            "n_empty_night": int(ir_emp.sum()),
            "n_empty_day": int((~ir_emp).sum()),
            "n_real_animals": int(len(c156_ani)),
            "n_train_embeddings": int(len(train_idx)),
            "n_other_test_animals": int(len(other_ani)),
            "n_other_test_animal_cams": int(other_ani["location"].nunique()),
            "n_other_test_empty": int(len(other_emp)),
            "n_other_test_empty_cams": int(other_emp["location"].nunique()),
            "white_bar_px": [424, 448],
            "fc_bias_max_abs": float(np.abs(b_fc.cpu().numpy()).max()),
            "sa_white_bar_cameras": [
                {"camera_name": "Camera 64", "location_raw": 121, "n": 82, "empty_pct": 0.0},
                {"camera_name": "Camera 278", "location_raw": 484, "n": 39, "empty_pct": 0.0},
                {"camera_name": "Camera 42", "location_raw": 82, "n": 19, "empty_pct": 0.0},
                {"camera_name": "Camera 100", "location_raw": 174, "n": 1, "empty_pct": 0.0},
                {"camera_name": "Camera 246", "location_raw": 434, "n": 1, "empty_pct": 0.0},
            ],
            "cam0_white_bar_empty": {"camera_name": "Camera 0", "location_raw": 0, "n": 3, "empty_pct": 100.0},
            "cam186_african_nonempty": {"camera_name": "Camera 186", "location_raw": 333, "n": 1184, "empty_pct": 0.0},
        },
        "layer_trajectories": layer_trajectories,
        "strips": strips_by_cohort,
        "layer49_rows": layer49_rows_by_cohort,
        "interventions_meta": interventions_meta,
        "interventions": interventions_data,
        "frozen_baseline": frozen_baseline,
        "frames": curated_frames,
    }

    out_path = Path(__file__).resolve().parents[1] / "results" / "camera156_whitebox.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    clean_bundle = sanitize_json(bundle)
    out_path.write_text(json.dumps(clean_bundle, allow_nan=False, separators=(",", ":")), encoding="utf-8")
    size_kb = out_path.stat().st_size / 1024.0
    print(f"\n[extract] Saved {out_path} ({size_kb:.1f} KB) in {time.time() - t0:.1f}s!", flush=True)

    # Print key verification summary for Col 1 (Night vs Day + Frozen Probe at Layer 49 rows 12-13!)
    print("\n=== VERIFICATION SUMMARY: COL 1 LAYER 1..49 CONTRASTIVE GRAD-CAM (rows 12-13 at Layer 49) ===")
    for cname in ("all", "night", "day"):
        tr = clean_bundle["layer_trajectories"][cname]
        print(f"\nCohort: {cname.upper()} ({tr['objective_label']})")
        for s in tr["stages"]:
            extra = ""
            if s["id"] == "layer4":
                extra = f" | r13 FT={s['ft_ctr_r13_pct']:.1f}%, Frz={s['frozen_ctr_r13_pct']:.1f}%"
            print(
                f"  {s['label']:8s} ({s['rows_label']:12s}, unif {s['uniform_pct']:4.1f}%): "
                f"FT={s['ft_ctr_pct']:5.1f}%, Frozen={s['frozen_ctr_pct']:5.1f}%, Ctrl={s['ctrl_ctr_pct']:5.1f}%, "
                f"StdClip={s['ft_std_clip_pct']:5.1f}%, CKA={s['cka_vs_frozen']:.3f}{extra}"
            )

    print("\n=== VERIFICATION SUMMARY: INTERVENTIONS & FULL 129,809 kNN (ALL) ===")
    for key in [m["id"] for m in interventions_meta]:
        c = clean_bundle["interventions"][key]["cohorts"]["all"]
        k = c["knn"]
        ctrl = clean_bundle["interventions"][key]["controls"]
        print(
            f"  {key:12s}: Emp={c['empty_pct']:5.1f}%, Rab={c['rabbit_pct']:4.1f}%, Pac={c['paca_pct']:4.1f}%, z0={c['z_empty']:+6.2f} | "
            f"24Ani R+P={ctrl['c156_ani_rab_paca']}/24 | OthAni={ctrl['other_ani_acc_pct']:5.1f}% | "
            f"kNN: SA_WB={k['sa_wb_pct']:5.1f}%, Cam0={k['cam0_pct']:4.1f}%, Cam186={k['cam186_pct']:4.1f}%, NormEmp={k['normal_empty_pct']:5.1f}%"
        )
    fr_c = clean_bundle["frozen_baseline"]["cohorts"]["all"]
    fr_k = fr_c["knn"]
    print(
        f"  {'frozen':12s}: Emp={fr_c['empty_pct']:5.1f}%, Rab={fr_c['rabbit_pct']:4.1f}%, Pac={fr_c['paca_pct']:4.1f}% | "
        f"kNN: SA_WB={fr_k['sa_wb_pct']:5.1f}%, Cam0={fr_k['cam0_pct']:4.1f}%, NormEmp={fr_k['normal_empty_pct']:5.1f}%, "
        f"TopCams={[x['camera_name'] + ':' + str(round(x['share_pct'], 1)) + '%' for x in fr_k['top_cameras'][:4]]}"
    )

    print("\n=== CURATED FRAMES ===")
    for fr in clean_bundle["frames"]:
        b = fr["states"]["baseline"]
        cb = fr["states"]["cover_bar"]
        r13 = fr["states"]["zero_r13"]
        b2 = fr["states"]["zero_r12_13"]
        print(
            f"  {fr['id']} ({fr['label']}, gt={fr['gt_common']}): "
            f"base={b['pred_common']}({b['prob_pct']:.0f}%) -> cov={cb['pred_common']}({cb['prob_pct']:.0f}%), "
            f"r13={r13['pred_common']}({r13['prob_pct']:.0f}%), r12-13={b2['pred_common']}({b2['prob_pct']:.0f}%) | "
            f"L22 bar={fr['maps']['l22_bar_share_pct']:.1f}%, L49 clip={fr['maps']['l49_relu_clip_pct']:.1f}%, L49 ctr12-13={fr['maps']['l49_ctr_r12_13_pct']:.1f}%"
        )


if __name__ == "__main__":
    main()
