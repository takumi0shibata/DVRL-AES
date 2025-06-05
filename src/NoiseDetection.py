# %%
import polars as pl
import numpy as np
import pickle

from data_valuation.core.loo_valuator import LOOValuator
from data_valuation.core.shapley_valuator import ShapleyValuator
from data_valuation.core.dvrl_valuator import DVRLValuator

from dvrl.sampling import select_diverse_subset
from dvrl.dataset import EssayDataset

# %%
PROMPT_ID = 1
RANDOM_SEED = 12

# %%
# Load embedding
with open('./outputs/embedding/deberta-v3-large.pkl', 'rb') as f:
    embedding_dict = pickle.load(f)

# Load essay data
print('Loading essay data...')
dataset = EssayDataset('./data/training_set_rel3.xlsx', './data/hand_crafted_v3.csv', './data/readability_features.csv')
source_data, target_data = dataset.cross_prompt_split(
    target_prompt_set=PROMPT_ID,
    add_pos=False,
)
print(f'    Number of source samples: {len(source_data["essay_id"])}')
print(f'    Number of target samples: {len(target_data["essay_id"])}')

# Select dev data
selected_dev_ids = select_diverse_subset(
    {essay_id: embedding_dict[essay_id] for essay_id in target_data['essay_id'] if essay_id in embedding_dict},
    n_samples=30,
    method='random',
    seed=RANDOM_SEED,
)
dev_mask = np.isin(target_data['essay_id'], selected_dev_ids)

x_dev = target_data['essay'][dev_mask]
y_dev = target_data['scaled_score'][dev_mask]

# %%
def add_noise(data: pl.DataFrame, noise_ratio: float, noise_range: tuple, seed: int) -> pl.DataFrame:
    np.random.seed(seed)
    noise_indices = np.random.choice(data.shape[0], int(data.shape[0] * noise_ratio), replace=False)
    noise_signs = np.random.choice([-1, 1], len(noise_indices))
    noise = np.random.uniform(noise_range[0], noise_range[1], len(noise_indices))

    noisy_scores = data['scaled_score'].to_numpy().copy()
    noisy_scores[noise_indices] += noise_signs * noise
    noisy_scores = np.clip(noisy_scores, 0, 1)  # Clipping

    data = data.with_columns(pl.Series(noisy_scores).alias('noisy_score'))

    is_noisy = np.zeros(data.shape[0], dtype=bool)
    is_noisy[noise_indices] = True
    data = data.with_columns(pl.Series(is_noisy).alias('is_noisy'))
    return data

train_data = add_noise(pl.DataFrame(source_data), noise_ratio=0.2, noise_range=(0.4, 0.6), seed=RANDOM_SEED)
train_data = train_data.drop(['ridley_feature', 'feature', 'readability'])
train_data.write_csv(f'./outputs/noise_detection/train_data_loo_{PROMPT_ID}.csv')
train_data

# %%
# Leave-One-Out Valuation
valuator = LOOValuator(
    prompt_id=PROMPT_ID,
    device='cuda',
    seed=RANDOM_SEED,
)

# # Data Shapley Valuation
# valuator = ShapleyValuator(
#     prompt_id=PROMPT_ID,
#     device='cuda',
#     seed=RANDOM_SEED,
# )

# # Data Valuation with Reinforcement Learning (DVRL)
# valuator = DVRLValuator(
#     prompt_id=PROMPT_ID,
#     device='cuda',
#     seed=RANDOM_SEED,
# )

# %%
estimated_values = valuator.estimate_values(
    x_train=train_data['essay'].to_list(),
    y_train=train_data['noisy_score'].to_numpy(),
    x_val=x_dev.tolist(),
    y_val=y_dev,
    sample_ids=train_data['essay_id'].to_numpy(),
)

# Save the estimated values
valuator.save_values(estimated_values, f'./outputs/noise_detection/estimated_values_loo_{PROMPT_ID}.csv')

# %%
# Load the estimated values as DataFrame
value_df = pl.read_csv(f'./outputs/noise_detection/estimated_values_loo_{PROMPT_ID}.csv').rename({'sample_id': 'essay_id'})
train_data = train_data.join(value_df, on='essay_id', how='left')
train_data

# %%
import matplotlib.pyplot as plt

def discover_corrupted_sample(dve_out, noise_mask, noise_rate, output_path=None, plot=True):
  """Reports True Positive Rate (TPR) of corrupted label discovery.

  Args:
    dve_out: data values (numpy array)
    noise_mask: 各サンプルがノイズか否かを示す True/False のブール配列
    noise_rate: ノイズサンプルの割合
    output_path: プロット画像の出力先パス
    plot: プロットを表示するかどうか

  Returns:
    output_perf: パーセンタイル毎（例: 5%刻み）に算出したノイズ検出率（TPR）
  """

  # サンプルをデータ値でソート
  num_bins = 20  # 100%を20分割
  sort_idx = np.argsort(dve_out)

  # 出力用初期化
  output_perf = np.zeros([num_bins,])
  total_noise = np.sum(noise_mask)  # 全ノイズサンプル数

  # 各パーセンタイルごとに計算
  for itt in range(num_bins):
    num_samples = int((itt+1) * len(dve_out) / num_bins)
    # 現在の範囲内でのノイズサンプル数をカウント
    discovered_noise = np.sum(noise_mask[sort_idx[:num_samples]])
    output_perf[itt] = discovered_noise / total_noise

  # ノイズ検出率のグラフをプロット
  if plot:
    num_x = int(num_bins / 2 + 1)
    x = [a * (1.0 / num_bins) for a in range(num_x)]
    y_dvrl = np.concatenate((np.zeros(1), output_perf[:(num_x - 1)]))
    y_opt = [min(a * ((1.0 / num_bins) / noise_rate), 1) for a in range(num_x)]
    y_random = x

    plt.figure(figsize=(5, 6))
    plt.plot(x, y_dvrl, 'o-')
    plt.plot(x, y_opt, '--')
    plt.plot(x, y_random, ':')
    plt.xlabel('Fraction of data Inspected', size=16)
    plt.ylabel('Fraction of discovered corrupted samples', size=16)
    plt.legend(['DVRL', 'Optimal', 'Random'], prop={'size': 16})
    plt.title('Corrupted Sample Discovery', size=16)
    if output_path:
      plt.savefig(output_path + 'corrupted_sample_discovery.pdf')

  # ノイズ検出率 (TPR) を返す
  return output_perf

# Assuming 'value' column contains the data values
dve_out = train_data['value'].to_numpy()
# Assuming 'is_noisy' column indicates corrupted samples (True for corrupted)
noise_mask = train_data['is_noisy'].to_numpy()
noise_rate = train_data['is_noisy'].mean()  # Calculate the noise rate

# Call the function and plot
# discover_corrupted_sample(dve_out, noise_mask, noise_rate, output_path='../outputs/noise_detection/')
discover_corrupted_sample(dve_out, noise_mask, noise_rate)

# %%



