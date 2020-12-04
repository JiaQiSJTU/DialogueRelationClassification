import os, json, spacy
import numpy as np
from nltk.tokenize import word_tokenize
from sklearn.metrics import f1_score, accuracy_score
from collections import defaultdict
from typing import List, Dict, Callable, Tuple


class ClassicalTokenizer(object):
    def __init__(self):
        self.glove_dir = "vocab_dir"

        if not os.path.exists(self.glove_dir):
            os.mkdir(self.glove_dir)

            self.build_vocab("ddrel")

        with open(self.glove_dir + "/token2id.json", "r") as f:
            self.token2id = json.loads(f.read())

        with open(self.glove_dir + "/id2token.json", "r") as f:
            self.id2token = json.loads(f.read())

    def build_vocab(self, data_dir: str):
        """ Build vocabulary over the given train, dev and test file.
        """
        self.token2id = {"[PAD]": 0, "[SOS]": 1, "[EOS]": 2, "[UNK]": 3}
        self.id2token = {0: "[PAD]", 1: "[SOS]", 2: "[EOS]", 3: "[UNK]"}

        token_cnt = 4
        for file_name in ["train.txt", "dev.txt", "test.txt"]:
            with open("{}/{}".format(data_dir, file_name), "r") as f:
                for sample in f.readlines():
                    sample = json.loads(sample)

                    context = sample["context"]
                    for sent in context:
                        for token in word_tokenize(sent):
                            if not token in self.token2id.keys():
                                self.token2id[token] = token_cnt
                                self.id2token[token_cnt] = token

                                token_cnt += 1

        with open("{}/token2id.json".format(self.glove_dir), "w") as f:
            json.dump(self.token2id, f)

        with open("{}/id2token.json".format(self.glove_dir), "w") as f:
            json.dump(self.id2token, f)
        print("# of token in the whole dataset: {}".format(token_cnt))

        # extract glove embeddings for the dataset
        token2embedding = {}
        with open("glove.840B.300d.txt", "r") as f:
            for line in f.readlines():
                line = line.strip().split()

                try:
                    token = line[0]
                    embedding = [float(v) for v in line[1:]]

                    token2embedding[token] = embedding
                except:
                    pass
        needed_embedding = {
            0: list(np.zeros(300)),
            1: list(np.random.normal(size=300)),
            2: list(np.random.normal(size=300)),
            3: list(np.random.normal(size=300)),
        }
        unk = 0
        for token, id in self.token2id.items():
            if token in token2embedding.keys():
                needed_embedding[id] = token2embedding[token]
            else:
                unk += 1

        with open(self.glove_dir + "/glove.conrel.txt", "w") as f:
            for token_id in range(0, token_cnt):
                if token_id in needed_embedding.keys():
                    f.write(" ".join([str(token_id)] + [str(v) for v in needed_embedding[token_id]]) + "\n")
                else:
                    f.write(" ".join([str(token_id)] + ["0." for _ in range(300)]) + "\n")

        print("# of unknown word: {}".format(unk))

    def encode(self, sent: str) -> List[int]:
        """
        Convert all the tokens in this sentence to their id
        """
        tokens = word_tokenize(sent)
        return [0] + [self.token2id[token] if token in self.token2id.keys() \
                          else 2 for token in tokens] + [1]

    def decode(self, tokens: List[int]) -> str:
        return " ".join([self.id2token[str(id)] for id in tokens])


def compute_per_session_score(output_filename: str, pairid: int = None, average_mode: str = None) -> Tuple:
    y, y_ = [], []

    with open(output_filename, "r") as f:
        for sample in f.readlines():
            sample = json.loads(sample)

            y.append(sample["label"])
            y_.append(sample["y_"])

    print(output_filename)
    print("=" * 30, " per session score ", "=" * 30)
    print("label acc: {:.3f}\t f1-macro: {:.3f}".format(
        accuracy_score(y, y_),
        f1_score(y, y_, average="macro")))


def compute_per_pair_score_mrr(output_filename="experiment_scripts/BERT_baseline.json"):
    """ MRR"""

    def mmr(logits: List[List[int]]) -> int:
        label_cnt = len(logits[0])
        session_cnt = len(logits)

        label2score = defaultdict(int)
        for logit in logits:
            logit = sorted([(val, idx) for idx, val in enumerate(logit)])

            for rank, (val, idx) in enumerate(logit):
                label2score[idx] += 1. / (label_cnt - rank)

        for idx in range(label_cnt):
            label2score[idx] /= session_cnt

        return max(label2score.items(), key=lambda x: x[1])[0]

    pair2label = {}
    pair2out = defaultdict(list)

    with open(output_filename, "r") as f:
        for sample in f.readlines():
            sample = json.loads(sample)

            pair2out[sample["pair-id"]].append(sample["logits"])
            pair2label[sample["pair-id"]] = sample["label"]

    for pairid in pair2label.keys():
        pair2out[pairid] = mmr(pair2out[pairid])

    y_ = [val for key, val in pair2out.items()]
    y = [val for key, val in pair2label.items()]

    assert len(y) == len(y_)

    print(output_filename)
    print("=" * 30, " per pair score by MRR", "=" * 30)
    print("label acc: {:.3f}\tf1-macro: {:.3f}".format(
        accuracy_score(y, y_),
        f1_score(y, y_, average="macro")
    ))


def per_pair_score():
    compute_per_pair_score_mrr("experiment_scripts/BERT_4_baseline-42.json")
    compute_per_pair_score_mrr("experiment_scripts/BERT_4_baseline-52.json")
    compute_per_pair_score_mrr("experiment_scripts/BERT_4_baseline-62.json")
    print("\n")

    compute_per_pair_score_mrr("experiment_scripts/BERT_6_baseline-42.json")
    compute_per_pair_score_mrr("experiment_scripts/BERT_6_baseline-52.json")
    compute_per_pair_score_mrr("experiment_scripts/BERT_6_baseline-62.json")
    print("\n")

    compute_per_pair_score_mrr("experiment_scripts/BERT_13_baseline-42.json")
    compute_per_pair_score_mrr("experiment_scripts/BERT_13_baseline-52.json")
    compute_per_pair_score_mrr("experiment_scripts/BERT_13_baseline-62.json")
    print("\n")

    # compute_per_pair_score_mrr("experiment_scripts/TextCNN_4-42.json")
    # compute_per_pair_score_mrr("experiment_scripts/TextCNN_4-52.json")
    # compute_per_pair_score_mrr("experiment_scripts/TextCNN_4-62.json")
    # print("\n")

    # compute_per_pair_score_mrr("experiment_scripts/TextCNN_6-42.json")
    # compute_per_pair_score_mrr("experiment_scripts/TextCNN_6-52.json")
    # compute_per_pair_score_mrr("experiment_scripts/TextCNN_6-62.json")
    # print("\n")

    # compute_per_pair_score_mrr("experiment_scripts/TextCNN_13-42.json")
    # compute_per_pair_score_mrr("experiment_scripts/TextCNN_13-52.json")
    # compute_per_pair_score_mrr("experiment_scripts/TextCNN_13-62.json")
    # print("\n")

    # compute_per_pair_score_mrr("experiment_scripts/LSTM_4-42.json")
    # compute_per_pair_score_mrr("experiment_scripts/LSTM_4-52.json")
    # compute_per_pair_score_mrr("experiment_scripts/LSTM_4-62.json")
    # print("\n")

    # compute_per_pair_score_mrr("experiment_scripts/LSTM_6-42.json")
    # compute_per_pair_score_mrr("experiment_scripts/LSTM_6-52.json")
    # compute_per_pair_score_mrr("experiment_scripts/LSTM_6-62.json")
    # print("\n")

    # compute_per_pair_score_mrr("experiment_scripts/LSTM_13-42.json")
    # compute_per_pair_score_mrr("experiment_scripts/LSTM_13-52.json")
    # compute_per_pair_score_mrr("experiment_scripts/LSTM_13-62.json")
    # print("\n")


def per_session_score():
    compute_per_session_score("experiment_scripts/BERT_4_baseline-42.json")
    compute_per_session_score("experiment_scripts/BERT_4_baseline-52.json")
    compute_per_session_score("experiment_scripts/BERT_4_baseline-62.json")
    print("\n")

    compute_per_session_score("experiment_scripts/BERT_6_baseline-42.json")
    compute_per_session_score("experiment_scripts/BERT_6_baseline-52.json")
    compute_per_session_score("experiment_scripts/BERT_6_baseline-62.json")
    print("\n")

    compute_per_session_score("experiment_scripts/BERT_13_baseline-42.json")
    compute_per_session_score("experiment_scripts/BERT_13_baseline-52.json")
    compute_per_session_score("experiment_scripts/BERT_13_baseline-62.json")
    print("\n")

    # compute_per_session_score("experiment_scripts/TextCNN_4-42.json")
    # compute_per_session_score("experiment_scripts/TextCNN_4-52.json")
    # compute_per_session_score("experiment_scripts/TextCNN_4-62.json")
    # print("\n")

    # compute_per_session_score("experiment_scripts/TextCNN_6-42.json")
    # compute_per_session_score("experiment_scripts/TextCNN_6-52.json")
    # compute_per_session_score("experiment_scripts/TextCNN_6-62.json")
    # print("\n")

    # compute_per_session_score("experiment_scripts/TextCNN_13-42.json")
    # compute_per_session_score("experiment_scripts/TextCNN_13-52.json")
    # compute_per_session_score("experiment_scripts/TextCNN_13-62.json")
    # print("\n")

    # compute_per_session_score("experiment_scripts/LSTM_4-42.json")
    # compute_per_session_score("experiment_scripts/LSTM_4-52.json")
    # compute_per_session_score("experiment_scripts/LSTM_4-62.json")
    # print("\n")

    # compute_per_session_score("experiment_scripts/LSTM_6-42.json")
    # compute_per_session_score("experiment_scripts/LSTM_6-52.json")
    # compute_per_session_score("experiment_scripts/LSTM_6-62.json")
    # print("\n")

    # compute_per_session_score("experiment_scripts/LSTM_13-42.json")
    # compute_per_session_score("experiment_scripts/LSTM_13-52.json")
    # compute_per_session_score("experiment_scripts/LSTM_13-62.json")
    # print("\n")


def test_tokenizer():
    t = ClassicalTokenizer()
    return t


def show(pairidx):
    pair2label = {}
    pair2out = defaultdict(list)

    with open("ddrel/test.txt", "r") as f:
        for sample in f.readlines():
            sample = json.loads(sample)

            if sample["pair-id"] == str(pairidx):  # and sample["y6"] == sample["6-label"]:
                for s in sample["context"]:
                    print(s)
                print("\n")
