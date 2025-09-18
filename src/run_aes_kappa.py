# run_aes_kappa.py
# -*- coding: utf-8 -*-
"""
要件:
 - DeBERTa-v3-large の CLS ベクトルを入力特徴に使用
 - 指定の MLP で score_norm を回帰（Sigmoid）
 - 予測はターゲット essay_set ごとに get_overall_score_range() で逆正規化→四捨五入→クリップ
 - 評価は scikit-learn の cohen_kappa_score(weights='quadratic')
 - 訓練データは、ターゲット essay_set を含めず、
     {同ジャンル(他セット)}, {他ジャンル1}, {他ジャンル2} の3ビンを 2^3-1 通りでオン/オフ
 - 最終結果を1つのCSVに保存

依存:
 pip install torch torchvision torchaudio -f https://download.pytorch.org/whl/cu121/torch_stable.html  # (必要に応じ)
 pip install transformers accelerate pandas scikit-learn tqdm

Usage:
  python run_aes_kappa.py --input_csv ./essays.csv --out_csv ./results_qwk_mlp_deberta.csv
"""

import os
import math
import argparse
import random
from typing import List, Dict, Tuple, Set

import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import cohen_kappa_score


# ======= 指定: スコア範囲（固定） =======

def get_overall_score_range(prompt_id: int):
    range_dict = {
        1: (2, 12),
        2: (1, 6),
        3: (0, 3),
        4: (0, 3),
        5: (0, 4),
        6: (0, 4),
        7: (0, 30),
        8: (0, 60)
    }
    return range_dict[prompt_id]


# ======= 指定のMLP（そのまま） =======

class MLP(nn.Module):
    """
    Simple Essay Scorer
    """
    def __init__(
        self,
        input_feature: int,
        hidden_dim: int = 512
    ) -> None:
        super(MLP, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_feature, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ======= ユーティリティ =======

def set_seed(seed: int = 12):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def standardize_fit(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (x - mean) / std, mean, std

def standardize_apply(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    std = np.where(std < 1e-6, 1.0, std)
    return (x - mean) / std

def ensure_score_norm(df: pd.DataFrame) -> pd.DataFrame:
    if "score_norm" in df.columns and not df["score_norm"].isna().any():
        return df
    # 無い場合は set毎の min-max で作成（学習用ターゲットは固定レンジで逆正規化するのでOK）
    df = df.copy()
    df["score_norm"] = np.nan
    for s, g in df.groupby("essay_set"):
        smin, smax = g["score"].min(), g["score"].max()
        rng = max(smax - smin, 1)
        df.loc[g.index, "score_norm"] = (g["score"] - smin) / rng
    return df

def denorm_and_round(pred_norm: np.ndarray, min_score: int, max_score: int) -> np.ndarray:
    rng = max_score - min_score
    raw = pred_norm * rng + min_score
    rounded = np.rint(raw).astype(int)
    return np.clip(rounded, min_score, max_score)


# ======= 埋め込み（CLS） =======

class TextDataset(Dataset):
    def __init__(self, texts: List[str]):
        self.texts = texts
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        return self.texts[idx]

def compute_embeddings_cls(df: pd.DataFrame,
                           text_col: str,
                           model_name: str = "microsoft/deberta-v3-large",
                           batch_size: int = 16,
                           max_length: int = 512,
                           cache_path: str = "embeddings_deberta_v3_large_cls.npz",
                           device: str = None) -> Tuple[np.ndarray, int]:
    """
    df[text_col] を DeBERTa-v3-large でエンコードし、CLS トークンの隠れ状態を特徴量にする。
    返り値: (embeddings [N, H], hidden_size)
    """
    if os.path.exists(cache_path):
        cache = np.load(cache_path)
        X = cache["embeddings"]
        hidden_size = X.shape[1]
        return X, hidden_size

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    ds = TextDataset(df[text_col].tolist())
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)

    embs = []
    with torch.no_grad():
        for batch_texts in tqdm(dl, desc="Encoding (CLS) with DeBERTa-v3-large"):
            enc = tokenizer(
                list(batch_texts),
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            # CLS ベクトル（先頭トークンの隠れ状態）
            cls_vec = out.last_hidden_state[:, 0, :]  # (B, H)
            embs.append(cls_vec.cpu().numpy())

    X = np.concatenate(embs, axis=0).astype(np.float32)
    np.savez_compressed(cache_path, embeddings=X)
    hidden_size = X.shape[1]
    return X, hidden_size


# ======= 学習・評価 =======

class ArrayDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32).reshape(-1, 1)
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def train_mlp(X_train: np.ndarray,
              y_train: np.ndarray,
              X_val: np.ndarray,
              y_val: np.ndarray,
              input_dim: int,
              device: str,
              hidden_dim: int = 512,
              epochs: int = 100,
              batch_size: int = 512,
              lr: float = 1e-3) -> MLP:
    model = MLP(input_feature=input_dim, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    ds_train = ArrayDataset(X_train, y_train)
    ds_val = ArrayDataset(X_val, y_val)
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False)

    for ep in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        for xb, yb in dl_train:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * xb.size(0)
        tr_loss /= len(ds_train)

        # 簡易val
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in dl_val:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                val_loss += loss.item() * xb.size(0)
        val_loss /= max(len(ds_val), 1)
        tqdm.write(f"[Epoch {ep:02d}] train MSE={tr_loss:.5f}  val MSE={val_loss:.5f}")

    return model

@torch.no_grad()
def predict_norm(model: MLP, X: np.ndarray, device: str, batch_size: int = 256) -> np.ndarray:
    model.eval()
    ds = ArrayDataset(X, np.zeros((X.shape[0], 1), dtype=np.float32))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    preds = []
    for xb, _ in dl:
        xb = xb.to(device)
        out = model(xb).cpu().numpy().reshape(-1)
        preds.append(out)
    return np.concatenate(preds, axis=0)


# ======= 実験セットアップ =======

GENRE_GROUPS: List[Set[int]] = [
    {1, 2},
    {3, 4, 5, 6},
    {7, 8},
]

def genre_bins_for_target(target_set: int) -> Tuple[Set[int], Set[int], Set[int]]:
    """
    target_set に対し、3つのビン（同ジャンル他セット / 他ジャンル1 / 他ジャンル2）を返す
    """
    target_group_idx = None
    for i, g in enumerate(GENRE_GROUPS):
        if target_set in g:
            target_group_idx = i
            break
    assert target_group_idx is not None, f"essay_set {target_set} not in predefined groups."

    same_group = set(GENRE_GROUPS[target_group_idx])
    same_group.discard(target_set)  # 同ジャンルの"他"セット

    other_groups = [GENRE_GROUPS[i] for i in range(len(GENRE_GROUPS)) if i != target_group_idx]
    return same_group, other_groups[0], other_groups[1]

def all_combinations_of_bins() -> List[Tuple[bool, bool, bool]]:
    """
    3ビンの有無を (b1, b2, b3) で列挙（000を除く）
    """
    combos = []
    for m in range(1, 8):  # 1..7
        b1 = bool(m & 1)
        b2 = bool(m & 2)
        b3 = bool(m & 4)
        combos.append((b1, b2, b3))
    return combos


# ======= メイン =======

def run_experiment(input_csv: str,
                   out_csv: str,
                   cache_path: str = "embeddings_deberta_v3_large_cls.npz",
                   seed: int = 12,
                   encoder_batch: int = 16,
                   encoder_maxlen: int = 512,
                   mlp_hidden: int = 512,
                   mlp_epochs: int = 100,
                   mlp_batch: int = 512,
                   mlp_lr: float = 1e-3,
                   device: str = None):
    set_seed(seed)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # データ読み込み
    df = pl.read_excel('./data/training_set_rel3.xlsx', infer_schema_length=20000)
    df = df.rename({'domain1_score': 'score'})
    df = df[['essay_set', 'essay_id', 'essay', 'score']]
    df = df.drop_nulls('score')


    # essay_setごとの(min, max)を取得 → 正規化
    df = (
        df.with_columns([
            pl.col("essay_set")
            .map_elements(lambda x: get_overall_score_range(int(x))[0], return_dtype=pl.Int64)
            .alias("_min"),
            pl.col("essay_set")
            .map_elements(lambda x: get_overall_score_range(int(x))[1], return_dtype=pl.Int64)
            .alias("_max"),
        ])
        .with_columns(
            pl.when(pl.col("_max") - pl.col("_min") == 0)
            .then(0.0)
            .otherwise(
                (pl.col("score").cast(pl.Float64) - pl.col("_min").cast(pl.Float64)) /
                (pl.col("_max").cast(pl.Float64) - pl.col("_min").cast(pl.Float64))
            )
            .alias("score_norm")
        )
        .drop(["_min", "_max"])
    )
    df = df.to_pandas()
    needed_cols = {"essay_set", "essay_id", "essay", "score", "score_norm"}
    missing = needed_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    # score_norm が無ければ作成
    df = ensure_score_norm(df)

    # 埋め込み（CLS, キャッシュ対応）
    X, hidden_size = compute_embeddings_cls(
        df, "essay",
        model_name="microsoft/deberta-v3-large",
        batch_size=encoder_batch,
        max_length=encoder_maxlen,
        cache_path=cache_path,
        device=device
    )
    print(f"Embeddings shape: {X.shape} (hidden={hidden_size})")

    results = []

    # 各ターゲット essay_set で実験
    unique_sets = sorted(df["essay_set"].unique().tolist())
    for target in unique_sets:
        # 固定レンジを取得
        smin, smax = get_overall_score_range(int(target))
        labels_full = list(range(smin, smax + 1))

        # 3ビンを決定
        bin_same, bin_o1, bin_o2 = genre_bins_for_target(int(target))

        # 7通りの組み合わせ
        for (use_b1, use_b2, use_b3) in all_combinations_of_bins():
            train_sets: Set[int] = set()
            if use_b1 and len(bin_same) > 0:
                train_sets |= set(bin_same)
            if use_b2:
                train_sets |= set(bin_o1)
            if use_b3:
                train_sets |= set(bin_o2)
            # 念のためターゲット除外
            train_sets.discard(int(target))

            if len(train_sets) == 0:
                continue

            # インデックス抽出
            train_idx = df.index[df["essay_set"].isin(train_sets)].to_numpy()
            test_idx = df.index[df["essay_set"] == target].to_numpy()

            X_train_raw = X[train_idx]
            y_train = df.loc[train_idx, "score_norm"].to_numpy().astype(np.float32)

            X_test_raw = X[test_idx]
            y_test_score = df.loc[test_idx, "score"].to_numpy().astype(int)

            # 標準化（訓練集合で fit、val/test に適用）
            X_train, mean, std = standardize_fit(X_train_raw)
            X_test = standardize_apply(X_test_raw, mean, std)

            # 訓練:検証 = 9:1（最低でもvalを1件確保）
            n_train = len(X_train)
            if n_train < 5:
                print(f"Skip: too few training samples for target {target} with sets {sorted(train_sets)}")
                continue
            val_size = max(1, int(0.1 * n_train))
            tr_size = n_train - val_size

            ds_all = ArrayDataset(X_train, y_train)
            ds_tr, ds_val = random_split(ds_all, [tr_size, val_size],
                                         generator=torch.Generator().manual_seed(seed))

            X_tr = ds_tr.dataset.X[ds_tr.indices]
            y_tr = ds_tr.dataset.y[ds_tr.indices].reshape(-1)
            X_val = ds_val.dataset.X[ds_val.indices]
            y_val = ds_val.dataset.y[ds_val.indices].reshape(-1)

            # 学習
            model = train_mlp(
                X_tr, y_tr,
                X_val, y_val,
                input_dim=hidden_size,
                device=device,
                hidden_dim=mlp_hidden,
                epochs=mlp_epochs,
                batch_size=mlp_batch,
                lr=mlp_lr
            )

            # 推論（norm）→ 固定レンジで整数スコアへ
            pred_norm = predict_norm(model, X_test, device=device, batch_size=256)
            pred_scores = denorm_and_round(pred_norm, min_score=smin, max_score=smax)

            # QWK（scikit-learn）
            qwk = cohen_kappa_score(y_test_score, pred_scores,
                                    weights='quadratic', labels=labels_full)

            results.append({
                "target_essay_set": int(target),
                "use_same_gen_others": bool(use_b1),
                "use_other_gen1": bool(use_b2),
                "use_other_gen2": bool(use_b3),
                "train_sets": ",".join(map(str, sorted(train_sets))),
                "n_train": int(n_train),
                "n_test": int(len(X_test)),
                "score_min_fixed": int(smin),
                "score_max_fixed": int(smax),
                "qwk": float(qwk),
            })

            # メモリ削減
            del model
            torch.cuda.empty_cache()

    # 結果保存
    res_df = pd.DataFrame(results).sort_values(
        by=["target_essay_set", "use_same_gen_others", "use_other_gen1", "use_other_gen2"]
    ).reset_index(drop=True)

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    res_df.to_csv(out_csv, index=False)
    print(f"Saved results to: {out_csv}")
    print(res_df.head(12))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, help="入力CSVパス（列: essay_set, essay_id, essay, score, score_norm）")
    parser.add_argument("--out_csv", type=str, required=True, help="結果CSV出力パス")
    parser.add_argument("--cache_npz", type=str, default="./logs/embeddings_deberta_v3_large_cls.npz", help="埋め込みキャッシュNPZパス")
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--encoder_batch", type=int, default=16)
    parser.add_argument("--encoder_maxlen", type=int, default=512)
    parser.add_argument("--mlp_hidden", type=int, default=512)
    parser.add_argument("--mlp_epochs", type=int, default=100)
    parser.add_argument("--mlp_batch", type=int, default=512)
    parser.add_argument("--mlp_lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default='cuda', help="'cuda' もしくは 'cpu'（未指定なら自動判定）")
    args = parser.parse_args()

    run_experiment(
        input_csv=args.input_csv,
        out_csv=args.out_csv,
        cache_path=args.cache_npz,
        seed=args.seed,
        encoder_batch=args.encoder_batch,
        encoder_maxlen=args.encoder_maxlen,
        mlp_hidden=args.mlp_hidden,
        mlp_epochs=args.mlp_epochs,
        mlp_batch=args.mlp_batch,
        mlp_lr=args.mlp_lr,
        device=args.device
    )