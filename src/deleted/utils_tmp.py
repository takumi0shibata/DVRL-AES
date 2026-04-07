import os
import re
import polars as pl
from sklearn.model_selection import train_test_split

def get_score_range(dataset_name, prompt_id):
    """ASAPデータセットのスコア範囲を取得."""
    score_ranges = {
        "ASAP": {
            1: (2, 12),
            2: (1, 6),
            3: (0, 3),
            4: (0, 3),
            5: (0, 4),
            6: (0, 4),
            7: (0, 30),
            8: (0, 60),
        },
        "TOEFL11": {
            1: (0, 2),
            2: (0, 2),
            3: (0, 2),
            4: (0, 2),
            5: (0, 2),
            6: (0, 2),
            7: (0, 2),
            8: (0, 2),
        }
    }
    return score_ranges[dataset_name][prompt_id]

def load_asap_dataset(dataset_dir: str, stratify: bool = False, essay_set: int = None) -> pl.DataFrame:
    data_path = os.path.join(dataset_dir, 'training_set_rel3.xlsx')
    df = pl.read_excel(data_path, infer_schema_length=100000)
    df = df.rename({'domain1_score': 'score'})
    df = df[['essay_set', 'essay_id', 'essay', 'score']]
    df = df.drop_nulls('score')

    if essay_set:
        df = df.filter(pl.col("essay_set") == essay_set)

    if not stratify:
        return df
    else:
        score_counts = df.group_by('score').len()
        # Add classes with only one sample directly to test set
        test_df = df.filter(pl.col('score').is_in(score_counts.filter(pl.col('len') == 1)['score']))
        df_remaining = df.filter(~pl.col('score').is_in(score_counts.filter(pl.col('len') == 1)['score']))
        
        # Perform stratified sampling on remaining data
        train_df, tmp_test_df = train_test_split(df_remaining, test_size=0.1, stratify=df_remaining['score'], random_state=123)
        test_df = pl.concat([test_df, tmp_test_df], how='vertical')
        
        return test_df