import pandas as pd
from typing import Dict, List
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import csv


class Trie(object):
    def __init__(self, sequences: List[List[int]] = []):
        self.trie_dict = {}
        self.len = 0
        if sequences:
            for sequence in sequences:
                Trie._add_to_trie(sequence, self.trie_dict)
                self.len += 1

        self.append_trie = None
        self.bos_token_id = None

    def append(self, trie, bos_token_id):
        self.append_trie = trie
        self.bos_token_id = bos_token_id

    def add(self, sequence: List[int]):
        Trie._add_to_trie(sequence, self.trie_dict)
        self.len += 1

    def get(self, prefix_sequence: List[int]):
        return Trie._get_from_trie(
            prefix_sequence, self.trie_dict, self.append_trie, self.bos_token_id
        )

    @staticmethod
    def load_from_dict(trie_dict):
        trie = Trie()
        trie.trie_dict = trie_dict
        trie.len = sum(1 for _ in trie)
        return trie

    @staticmethod
    def _add_to_trie(sequence: List[int], trie_dict: Dict):
        if sequence:
            if sequence[0] not in trie_dict:
                trie_dict[sequence[0]] = {}
            Trie._add_to_trie(sequence[1:], trie_dict[sequence[0]])

    @staticmethod
    def _get_from_trie(
            prefix_sequence: List[int],
            trie_dict: Dict,
            append_trie=None,
            bos_token_id: int = None,
    ):
        if len(prefix_sequence) == 0:
            output = list(trie_dict.keys())
            if append_trie and bos_token_id in output:
                output.remove(bos_token_id)
                output += list(append_trie.trie_dict.keys())
            return output
        elif prefix_sequence[0] in trie_dict:
            return Trie._get_from_trie(
                prefix_sequence[1:],
                trie_dict[prefix_sequence[0]],
                append_trie,
                bos_token_id,
            )
        else:
            if append_trie:
                return append_trie.get(prefix_sequence)
            else:
                return []

    def __iter__(self):
        def _traverse(prefix_sequence, trie_dict):
            if trie_dict:
                for next_token in trie_dict:
                    yield from _traverse(
                        prefix_sequence + [next_token], trie_dict[next_token]
                    )
            else:
                yield prefix_sequence

        return _traverse([], self.trie_dict)

    def __len__(self):
        return self.len

    def __getitem__(self, value):
        return self.get(value)


def prefix_allowed_tokens_fn(candidate_trie):
    def prefix_allowed_tokens(batch_id, sentence):
        sentence = sentence.tolist()
        trie_out = candidate_trie.get(sentence)
        return trie_out if trie_out else [tokenizer.pad_token_id]

    return prefix_allowed_tokens


def read_candidates_from_csv(file_path: str) -> List[str]:
    candidates = []
    with open(file_path, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            candidates.append(row[0])
    return candidates


# 全局初始化tokenizer和model
tokenizer = T5Tokenizer.from_pretrained("/root/autodl-tmp/PulyLanguage/results_citeseer_lk/checkpoint-21000")
model = T5ForConditionalGeneration.from_pretrained("/root/autodl-tmp/PulyLanguage/results_citeseer_lk/checkpoint-21000")

# 全局初始化候选序列和前缀树
candidates = read_candidates_from_csv('candidate2.csv')
candidate_trie = Trie([[0] + tokenizer.encode("{}".format(e)) for e in candidates])


def generate_output(input_text: str) -> List[str]:
    input_ids = tokenizer.batch_encode_plus(
        [input_text], padding="longest", return_tensors="pt"
    )["input_ids"]

    prefix_allowed_tokens = prefix_allowed_tokens_fn(candidate_trie)
    output_ids = model.generate(
        input_ids,
        max_length=150,
        prefix_allowed_tokens_fn=prefix_allowed_tokens,
        num_beams=100,
        num_return_sequences=50,
    )

    decoded_outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    return decoded_outputs


if __name__ == "__main__":
    test_df = pd.read_csv('citeseer_link_right_split_test.csv')
    test_df = test_df.sample(50)

    for index, row in test_df.iterrows():
        input_text = row['input_text']
        expected_output = row['output_text']

        output_text = generate_output(input_text)
        print('output_text:')
        print(output_text)
        print('expected_output:')
        print(expected_output)
        print("--------------")