import re
import polars as pl
from sklearn.model_selection import train_test_split
import numpy as np
import nltk

class EssayDataset:
    url_replacer = '<url>'
    num_regex = re.compile(r'^[+-]?[0-9]+\.?[0-9]*$')
    ref_scores_dtype = 'int32'
    MAX_SENTLEN = 50
    MAX_SENTNUM = 100

    def __init__(self, main_file, feature_file, readability_file):
        self.main_data = pl.read_excel(main_file, infer_schema_length=20000)
        self.feature_data = pl.read_csv(feature_file)
        self.readability_data = pl.read_csv(readability_file)

        # 得点のスケーリング用の範囲
        self.score_ranges = {
            1: {'min': 2, 'max': 12},
            2: {'min': 1, 'max': 6},
            3: {'min': 0, 'max': 3},
            4: {'min': 0, 'max': 3},
            5: {'min': 0, 'max': 4},
            6: {'min': 0, 'max': 4},
            7: {'min': 0, 'max': 30},
            8: {'min': 0, 'max': 60},
        }

        self._preprocess_dataframe()

    def replace_url(self, text):
        replaced_text = re.sub(r'(http[s]?://)?((www)\.)?([a-zA-Z0-9]+)\.{1}((com)(\.(cn))?|(org))', self.url_replacer, text)
        return replaced_text
    
    def tokenize(self, string):
        tokens = nltk.word_tokenize(string)
        for index, token in enumerate(tokens):
            if token == '@' and (index+1) < len(tokens):
                tokens[index+1] = '@' + re.sub('[0-9]+.*', '', tokens[index+1])
                tokens.pop(index)
        return tokens
    
    def tokenize(self, string):
        tokens = nltk.word_tokenize(string)
        for index, token in enumerate(tokens):
            if token == '@' and (index+1) < len(tokens):
                tokens[index+1] = '@' + re.sub('[0-9]+.*', '', tokens[index+1])
                tokens.pop(index)
        return tokens
    
    def shorten_sentence(self, sent, max_sentlen):
        new_tokens = []
        sent = sent.strip()
        tokens = nltk.word_tokenize(sent)
        if len(tokens) > max_sentlen:
            split_keywords = ['because', 'but', 'so', 'You', 'He', 'She', 'We', 'It', 'They', 'Your', 'His', 'Her']
            k_indexes = [i for i, key in enumerate(tokens) if key in split_keywords]
            processed_tokens = []
            if not k_indexes:
                num = len(tokens) / max_sentlen
                num = int(round(num))
                k_indexes = [(i+1)*max_sentlen for i in range(num)]

            processed_tokens.append(tokens[0:k_indexes[0]])
            len_k = len(k_indexes)
            for j in range(len_k-1):
                processed_tokens.append(tokens[k_indexes[j]:k_indexes[j+1]])
            processed_tokens.append(tokens[k_indexes[-1]:])

            for token in processed_tokens:
                if len(token) > max_sentlen:
                    num = len(token) / max_sentlen
                    num = int(np.ceil(num))
                    s_indexes = [(i+1)*max_sentlen for i in range(num)]

                    len_s = len(s_indexes)
                    new_tokens.append(token[0:s_indexes[0]])
                    for j in range(len_s-1):
                        new_tokens.append(token[s_indexes[j]:s_indexes[j+1]])
                    new_tokens.append(token[s_indexes[-1]:])

                else:
                    new_tokens.append(token)
        else:
            return [tokens]

        return new_tokens
    
    def tokenize_to_sentences(self, text, max_sentlength, create_vocab_flag=False):
        sents = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\!|\?)\s', text)
        processed_sents = []
        for sent in sents:
            if re.search(r'(?<=\.{1}|\!|\?|\,)(@?[A-Z]+[a-zA-Z]*[0-9]*)', sent):
                s = re.split(r'(?=.{2,})(?<=\.{1}|\!|\?|\,)(@?[A-Z]+[a-zA-Z]*[0-9]*)', sent)
                ss = " ".join(s)
                ssL = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\!|\?)\s', ss)

                processed_sents.extend(ssL)
            else:
                processed_sents.append(sent)

        if create_vocab_flag:
            sent_tokens = [self.tokenize(sent) for sent in processed_sents]
            tokens = [w for sent in sent_tokens for w in sent]
            return tokens

        sent_tokens = []
        for sent in processed_sents:
            shorten_sents_tokens = self.shorten_sentence(sent, max_sentlength)
            sent_tokens.extend(shorten_sents_tokens)
        return sent_tokens
    
    def text_tokenizer(self, text, replace_url_flag=True, tokenize_sent_flag=True, create_vocab_flag=False):
        text = self.replace_url(text)
        text = text.replace(u'"', u'')
        if "..." in text:
            text = re.sub(r'\.{3,}(\s+\.{3,})*', '...', text)
        if "??" in text:
            text = re.sub(r'\?{2,}(\s+\?{2,})*', '?', text)
        if "!!" in text:
            text = re.sub(r'\!{2,}(\s+\!{2,})*', '!', text)

        tokens = self.tokenize(text)
        if tokenize_sent_flag:
            text = " ".join(tokens)
            sent_tokens = self.tokenize_to_sentences(text, self.MAX_SENTLEN, create_vocab_flag)
            return sent_tokens
        else:
            raise NotImplementedError
        
    def is_number(self, token):
        return bool(self.num_regex.match(token))
    
    def read_pos_vocab(self, train_essays_list: list[str]):
        pos_tags_count = {}
        for essay in train_essays_list[:16]:
            content = self.text_tokenizer(essay, True, True, True)
            content = [w.lower() for w in content]
            tags = nltk.pos_tag(content)
            for tag in tags:
                tag = tag[1]
                try:
                    pos_tags_count[tag] += 1
                except KeyError:
                    pos_tags_count[tag] = 1

        pos_tags = {'<pad>': 0, '<unk>': 1}
        pos_len = len(pos_tags)
        pos_index = pos_len
        for pos in pos_tags_count.keys():
            pos_tags[pos] = pos_index
            pos_index += 1
        return pos_tags
    
    def read_essay_sets(self, essay_list, pos_tags):
        out_data = {
            'pos_x': [],
            'max_sentnum': -1,
            'max_sentlen': -1
        }
        for content in essay_list:
            sent_tokens = self.text_tokenizer(content, replace_url_flag=True, tokenize_sent_flag=True)
            sent_tokens = [[w.lower() for w in s] for s in sent_tokens]

            sent_tag_indices = []
            tag_indices = []
            for sent in sent_tokens:
                length = len(sent)
                if length > 0:
                    if out_data['max_sentlen'] < length:
                        out_data['max_sentlen'] = length
                    tags = nltk.pos_tag(sent)
                    for tag in tags:
                        if tag[1] in pos_tags:
                            tag_indices.append(pos_tags[tag[1]])
                        else:
                            tag_indices.append(pos_tags['<unk>'])
                    sent_tag_indices.append(tag_indices)
                    tag_indices = []

            out_data['pos_x'].append(sent_tag_indices)
            if out_data['max_sentnum'] < len(sent_tag_indices):
                out_data['max_sentnum'] = len(sent_tag_indices)

        return out_data
    
    def pad_hierarchical_text_sequences(self, index_sequences, max_sentnum, max_sentlen):
        X = np.empty([len(index_sequences), max_sentnum, max_sentlen], dtype=np.int32)

        for i in range(len(index_sequences)):
            sequence_ids = index_sequences[i]
            num = len(sequence_ids)

            for j in range(num):
                word_ids = sequence_ids[j]
                length = len(word_ids)
                for k in range(length):
                    wid = word_ids[k]
                    X[i, j, k] = wid
                X[i, j, length:] = 0

            X[i, num:, :] = 0
        return X

    def _scale_score(self, score, essay_set):
        min_score = self.score_ranges[essay_set]['min']
        max_score = self.score_ranges[essay_set]['max']
        return (score - min_score) / (max_score - min_score)
    
    def _preprocess_dataframe(self):
        ##############################
        # メインデータの前処理
        ##############################
        self.main_data = self.main_data.drop_nulls('domain1_score') # プロンプト4に得点がないデータが存在するので削除
        self.main_data = self.main_data.rename({'domain1_score': 'original_score'})
        
        # domain1_scoreのスケーリング
        self.main_data = self.main_data.with_columns(
            pl.struct(['original_score', 'essay_set'])
            .map_elements(lambda x: self._scale_score(x['original_score'], x['essay_set']), return_dtype=pl.Float64)
            .alias('scaled_score')
        )
        self.main_data = self.main_data.select(['essay_id', 'essay_set', 'essay', 'original_score', 'scaled_score'])
        
        ##############################
        # 特徴量データの前処理
        ##############################
        self.feature_data = self.feature_data.rename({'item_id': 'essay_id', 'prompt_id': 'essay_set'})
        self.feature_data = self.feature_data.drop('score')

        # essay_set単位でmin-maxスケーリングを適用し、最後に結合
        scaled_feature_data_list = []
        for essay_set in self.feature_data['essay_set'].unique():
            # 各essay_setごとにフィルタリング
            set_data = self.feature_data.filter(pl.col('essay_set') == essay_set)
            # 3列目以降に対してmin-maxスケーリングを適用
            for col in set_data.columns[2:]:  # essay_idとessay_set以外の列
                min_val = set_data[col].min()
                max_val = set_data[col].max()
                # スケーリングを適用
                set_data = set_data.with_columns(
                    ((pl.col(col) - min_val) / (max_val - min_val)).alias(col)
                )
            # スケーリング済みのデータをリストに追加
            scaled_feature_data_list.append(set_data)
        # スケーリング済みデータを結合
        self.feature_data = pl.concat(scaled_feature_data_list)
        
        ##############################
        # 読みやすさデータの前処理
        ##############################
        self.readability_data = self.readability_data.rename({'item_id': 'essay_id'})

    def prompt_specific_split(self, test_essay_set, test_size=0.2, dev_size=0.1, random_state=42):
        # essay_idを使用してデータを分割
        essay_ids = self.main_data.filter(pl.col('essay_set') == test_essay_set)['essay_id'].to_list()
        train_ids, test_ids = train_test_split(
            essay_ids, test_size=test_size, random_state=random_state, shuffle=True
        )
        train_ids, dev_ids = train_test_split(
            train_ids, test_size=dev_size / (1 - test_size), random_state=random_state, shuffle=True
        )

        # データ抽出のためのヘルパー関数を定義
        def extract_data(ids):
            main = self.main_data.filter(pl.col('essay_id').is_in(ids))
            feature = self.feature_data.filter(pl.col('essay_id').is_in(ids))
            readability = self.readability_data.filter(pl.col('essay_id').is_in(ids))
            return {
                'essay_id': main['essay_id'].to_list(),
                'essay_set': main['essay_set'].to_list(),
                'essay': main['essay'].to_list(),
                'original_score': main['original_score'].to_list(),
                'scaled_score': main['scaled_score'].to_list(),
                'feature': feature.to_numpy(),
                'readability': readability.to_numpy(),
            }

        # 各データセットを作成
        train_data = extract_data(train_ids)
        dev_data = extract_data(dev_ids)
        test_data = extract_data(test_ids)

        return train_data, dev_data, test_data
    
    def cross_prompt_split(
        self,
        target_prompt_set: int,
        add_pos: bool = False,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        
        # Filter data into source and target datasets
        source_data = self.main_data.filter(pl.col('essay_set') != target_prompt_set)
        target_data = self.main_data.filter(pl.col('essay_set') == target_prompt_set)
        
        # Function to extract data and include embeddings
        def extract_data(df: pl.DataFrame):
            
            # Sort feature and readability DataFrames by essay_id
            df = df.sort('essay_id')
            feature = self.feature_data.filter(pl.col('essay_id').is_in(df['essay_id']))
            readability = self.readability_data.filter(pl.col('essay_id').is_in(df['essay_id']))
            
            # Assertions to ensure data integrity
            assert len(df) == len(feature) == len(readability), "Data lengths must match."
            assert df['essay_id'].to_list() == feature['essay_id'].to_list(), "Essay IDs must match."
            assert df['essay_id'].to_list() == readability['essay_id'].to_list(), "Essay IDs must match."

            # Create Ridley's feature
            feature = feature.drop(['essay_id', 'essay_set']).to_numpy()
            readability = readability.drop(['essay_id']).to_numpy()
            ridley_feature = np.concatenate([feature, readability], axis=1)

            return {
                'essay_id': df['essay_id'].to_numpy(),
                'essay_set': df['essay_set'].to_numpy(),
                'essay': df['essay'].to_numpy(),
                'original_score': df['original_score'].to_numpy(),
                'scaled_score': df['scaled_score'].to_numpy(),
                'ridley_feature': ridley_feature,
                'feature': feature,
                'readability': readability,
            }

        # Extract data and include embeddings
        source_data = extract_data(source_data)
        target_data = extract_data(target_data)

        if add_pos:
            # create pos_x by Ridley style
            pos_tags = self.read_pos_vocab(source_data['essay'])
            source_pos_data = self.read_essay_sets(source_data['essay'], pos_tags)
            
            # Handle empty dev data case
            target_pos_data = self.read_essay_sets(target_data['essay'], pos_tags)
            max_sentnum = max(source_pos_data['max_sentnum'], target_pos_data['max_sentnum'])
            max_sentlen = max(source_pos_data['max_sentlen'], target_pos_data['max_sentlen'])
            
            # Pad the sequences with shape [batch, max_sentence_num, max_sentence_length]
            X_source_pos = self.pad_hierarchical_text_sequences(source_pos_data['pos_x'], max_sentnum, max_sentlen)
            source_data['pos_x'] = X_source_pos.reshape((X_source_pos.shape[0], X_source_pos.shape[1] * X_source_pos.shape[2]))
            source_data['pos_vocab'] = pos_tags
            source_data['max_sentnum'] = max_sentnum
            source_data['max_sentlen'] = max_sentlen
            
            X_target_pos = self.pad_hierarchical_text_sequences(target_pos_data['pos_x'], max_sentnum, max_sentlen)
            target_data['pos_x'] = X_target_pos.reshape((X_target_pos.shape[0], X_target_pos.shape[1] * X_target_pos.shape[2]))
            
        return source_data, target_data


# TEST
if __name__ == '__main__':
    dataset = EssayDataset('data/training_set_rel3.xlsx', 'data/hand_crafted_v3.csv', 'data/readability_features.csv')
    source_data, target_data = dataset.cross_prompt_split(
        target_prompt_set=1,
        add_pos=True
    )
    print(source_data['essay_id'].shape)
    print(target_data['essay_id'].shape)