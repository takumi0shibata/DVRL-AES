import numpy as np
from typing import Dict, List

from typing import Literal

def select_diverse_subset(
    id_to_embedding: Dict[str, np.ndarray],
    n_samples: int,
    method: Literal["greedy", "kmeans++", "maxmin", "random"] = "greedy",
    seed: int = None
) -> List[str]:
    """
    指定手法で多様またはランダムなサブセットを選択する。

    Parameters:
        id_to_embedding (Dict[str, np.ndarray]): id→embeddingの辞書
        n_samples (int): 選択数
        method (str): 'greedy', 'kmeans++', 'maxmin', 'random'
        seed (int): 乱数シード（任意）

    Returns:
        List[str]: 選ばれたIDリスト
    """
    if method == "greedy":
        return select_greedy_diverse_subset(id_to_embedding, n_samples, seed)
    elif method == "kmeans++":
        return select_kmeanspp_subset(id_to_embedding, n_samples, seed)
    elif method == "maxmin":
        return select_maxmin_subset(id_to_embedding, n_samples, seed)
    elif method == "random":
        return select_random_subset(id_to_embedding, n_samples, seed)
    else:
        raise ValueError(f"未知のサンプリング手法: {method}")

def select_greedy_diverse_subset(
    id_to_embedding: Dict[str, np.ndarray],
    n_samples: int,
    seed: int = None
) -> List[str]:
    if seed is not None:
        np.random.seed(seed)

    assert n_samples <= len(id_to_embedding)

    ids = list(id_to_embedding.keys())
    embeddings = np.stack([id_to_embedding[i] for i in ids])

    selected_indices = []
    remaining_indices = list(range(len(ids)))

    first = np.random.choice(remaining_indices)
    selected_indices.append(first)
    remaining_indices.remove(first)

    for _ in range(1, n_samples):
        selected_embeds = embeddings[selected_indices]
        remaining_embeds = embeddings[remaining_indices]

        distances = np.linalg.norm(
            remaining_embeds[:, None, :] - selected_embeds[None, :, :],
            axis=2
        ).sum(axis=1)

        next_index_in_remaining = np.argmax(distances)
        next_global_index = remaining_indices[next_index_in_remaining]

        selected_indices.append(next_global_index)
        remaining_indices.remove(next_global_index)

    return [ids[i] for i in selected_indices]


def select_kmeanspp_subset(
    id_to_embedding: Dict[str, np.ndarray],
    n_samples: int,
    seed: int = None
) -> List[str]:
    if seed is not None:
        np.random.seed(seed)

    assert n_samples <= len(id_to_embedding)

    ids = list(id_to_embedding.keys())
    embeddings = np.stack([id_to_embedding[i] for i in ids])
    n_total = len(ids)

    selected_indices = []

    first = np.random.randint(0, n_total)
    selected_indices.append(first)

    for _ in range(1, n_samples):
        selected_embeds = embeddings[selected_indices]
        dist_sq = np.min(
            np.linalg.norm(embeddings[:, None, :] - selected_embeds[None, :, :], axis=2) ** 2,
            axis=1
        )

        probs = dist_sq / dist_sq.sum()
        next_idx = np.random.choice(n_total, p=probs)
        while next_idx in selected_indices:
            next_idx = np.random.choice(n_total, p=probs)

        selected_indices.append(next_idx)

    return [ids[i] for i in selected_indices]


def select_maxmin_subset(
    id_to_embedding: Dict[str, np.ndarray],
    n_samples: int,
    seed: int = None
) -> List[str]:
    if seed is not None:
        np.random.seed(seed)

    assert n_samples <= len(id_to_embedding)

    ids = list(id_to_embedding.keys())
    embeddings = np.stack([id_to_embedding[i] for i in ids])
    n_total = len(ids)

    selected_indices = []

    first = np.random.randint(0, n_total)
    selected_indices.append(first)

    for _ in range(1, n_samples):
        selected_embeds = embeddings[selected_indices]

        dists = np.linalg.norm(embeddings[:, None, :] - selected_embeds[None, :, :], axis=2)
        min_dists = np.min(dists, axis=1)

        for i in selected_indices:
            min_dists[i] = -np.inf

        next_idx = np.argmax(min_dists)
        selected_indices.append(next_idx)

    return [ids[i] for i in selected_indices]


def select_random_subset(
    id_to_embedding: Dict[str, np.ndarray],
    n_samples: int,
    seed: int = None
) -> List[str]:
    """
    ランダムにn個のIDを選択する。

    Parameters:
        id_to_embedding (Dict[str, np.ndarray]): id→embeddingの辞書
        n_samples (int): 選択する数
        seed (int): 乱数シード（オプション）

    Returns:
        List[str]: 選ばれたIDのリスト
    """
    if seed is not None:
        np.random.seed(seed)

    ids = list(id_to_embedding.keys())
    assert n_samples <= len(ids)

    return list(np.random.choice(ids, size=n_samples, replace=False))