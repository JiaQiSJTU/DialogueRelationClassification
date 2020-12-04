import torch, json, math
import torch.nn as nn
import pytorch_lightning as pl
from torch.nn import functional as F
from transformers import BertModel
from collections import OrderedDict
from argparse import ArgumentParser
from typing import List, Union, Dict
from torch.optim import Adam, SGD, Adadelta
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.metrics.functional import accuracy, f1_score


class ConversationRelationModel(LightningModule):
    def __init__(self, random_seed: int):
        super().__init__()

        self.random_seed = random_seed

    def _val_and_test_step(self, batch, stage=None):
        output = self(**batch)

        loss = output["loss"]
        logits = output["logits"]

        device_idx = loss.device.index

        y_ = torch.argmax(logits, dim=-1)
        y = batch["label"]

        acc = accuracy(y_, y, num_classes=batch["num_class"][0])

        if self.on_gpu:
            acc = acc.cuda(device_idx)

        result = OrderedDict({"loss": loss, "acc": acc})

        if stage == "test":
            result["y_"] = y_.tolist()
            result["label"] = batch["label"].tolist()
            result["logits"] = torch.exp(logits).tolist()

            result["pair-id"] = batch["pair-id"].tolist()
            result["session-id"] = batch["session-id"].tolist()
        return result

    def _compute_epoch_end_metrics(self, outputs: List) -> Dict:
        loss_mean, acc = 0., 0.

        for output in outputs:
            loss_mean += output["loss"]
            acc += output["acc"]

        loss_mean /= len(outputs)
        acc /= len(outputs)

        return {"loss": loss_mean, "acc": acc}

    def _write_out(self, outputs: List, outfile_name: str):
        write_out = {
            "pair-id": [],
            "session-id": [],
            "label": [],
            "y_": [],
            "logits": [],
        }
        for output in outputs:
            for key in write_out.keys():
                write_out[key].extend(output[key])

        with open("experiment_scripts/" + outfile_name, "w") as f:
            for i, _ in enumerate(write_out["pair-id"]):
                json.dump({
                    "pair-id": write_out["pair-id"][i],
                    "session-id": write_out["session-id"][i],
                    "label": write_out["label"][i],
                    "y_": write_out["y_"][i],
                    "logits": write_out["logits"][i],
                }, f)
                f.write("\n")

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def configure_optimizers(self):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError

    def test_step(self, batch, batch_idx):
        raise NotImplementedError

    def validation_epoch_end(self, outputs: List):
        raise NotImplementedError

    def test_epoch_end(self, outputs: List):
        raise NotImplementedError


class BertBaselineModel(ConversationRelationModel):
    def __init__(self, num_class: int, random_seed: int, learning_rate=1e-6, **args):
        super().__init__(random_seed=random_seed)

        # Save hyperparameters for checkpoint
        self.save_hyperparameters()

        # Hyper Parameters
        self.lr = learning_rate
        self.num_class = num_class

        # Model Architecture
        self.bert = BertModel.from_pretrained("bert-base-cased")

        self.hidden_size = self.bert.config.hidden_size

        self.classification_linear = nn.Sequential(
            nn.Linear(self.hidden_size, self.num_class),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, *args, **kwargs):
        """
        Use [CLS] output for prediction.
        """
        encoded_context = kwargs["encoded_context"]
        turn_type_ids = kwargs["turn_type_ids"]
        attention_mask = kwargs["attention_mask"]

        bsz, _ = encoded_context.shape

        _, pooler_output = self.bert(
            input_ids=encoded_context,
            attention_mask=attention_mask,
            token_type_ids=turn_type_ids,
        )

        y_ = self.classification_linear(pooler_output)

        self.nllloss = nn.NLLLoss()

        y = kwargs["label"]

        loss = self.nllloss(y_, y)

        return {"logits": y_, "loss": loss}

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        output = self(**batch)
        loss = output["loss"]
        tqdm_dict = {"train_loss": loss}
        result = OrderedDict({
            "loss": loss, "progress_bar": tqdm_dict, "log": tqdm_dict
        })
        return result

    def validation_step(self, batch, batch_idx):
        return self._val_and_test_step(batch)

    def test_step(self, batch, batch_idx):
        return self._val_and_test_step(batch, stage="test")

    def validation_epoch_end(self, outputs: List) -> Dict:
        metrics = self._compute_epoch_end_metrics(outputs)

        tqdm_dict = {"val_{}".format(key): value for key, value in metrics.items()}
        result = {
            "progress_bar": tqdm_dict,
            "log": tqdm_dict,
        }
        return result

    def test_epoch_end(self, outputs: List) -> Dict:
        metrics = self._compute_epoch_end_metrics(outputs)

        tqdm_dict = {"test_{}".format(key): value for key, value in metrics.items()}
        result = {
            "progress_bar": tqdm_dict,
            "log": tqdm_dict,
        }
        self._write_out(outputs, "BERT_{}_baseline-{}.json".format(self.num_class, self.random_seed))
        return result

    @staticmethod
    def add_model_specific_arguments(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument("--learning_rate", type=float, default=1e-5)
        parser.add_argument("--output_answer_path", type=str, default=None)
        return parser


class CNNBaselineModel(ConversationRelationModel):
    """
    Re-implement CNN proposed by `Kim et.al.(2014)` (TextCNN)
    """

    def __init__(self, num_class: int, random_seed: int, learning_rate=1e-2, freeze_embedding=False):
        super().__init__(random_seed=random_seed)
        self.save_hyperparameters()

        self.lr = learning_rate
        self.num_class = num_class

        # We use GLoVe for word embedding
        embedding_weights = []
        with open("vocab_dir/glove.conrel.txt", "r") as f:
            for line in f.readlines():
                weights = [float(v) for v in line.strip().split()[1:]]
                embedding_weights.append(weights)
        embedding_weights = torch.tensor(embedding_weights)

        self.embedding = nn.Embedding.from_pretrained(
            embedding_weights,
            padding_idx=0,
            freeze=freeze_embedding)
        # self.embedding = nn.Embedding(num_embeddings=22306, embedding_dim=300, padding_idx=0)

        self.cnn3 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=100, kernel_size=(3, 300)),
            nn.Tanh(),
            nn.Dropout(0.5),
        )
        self.cnn4 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=100, kernel_size=(4, 300)),
            nn.Tanh(),
            nn.Dropout(0.5),
        )
        self.cnn5 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=100, kernel_size=(5, 300)),
            nn.Tanh(),
            nn.Dropout(0.5),
        )

        self.classification_linear = nn.Sequential(
            nn.Linear(in_features=300, out_features=num_class),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, *args, **kwargs):
        # after embedding: bsz x sent_len x word_dim
        x = self.embedding(kwargs["encoded_context"])
        bsz, sent_len, word_dim = x.shape

        x = x.reshape(bsz, 1, sent_len, word_dim)

        # max over time pooling

        x1, _ = torch.max(self.cnn3(x).reshape(bsz, 100, -1), dim=-1)
        x2, _ = torch.max(self.cnn4(x).reshape(bsz, 100, -1), dim=-1)
        x3, _ = torch.max(self.cnn5(x).reshape(bsz, 100, -1), dim=-1)

        x = torch.cat((x1, x2, x3), dim=-1)

        y_ = self.classification_linear(x)
        y = kwargs["label"]

        self.nllloss = nn.NLLLoss()

        loss = self.nllloss(y_, y)

        return {"logits": y_, "loss": loss}

    def configure_optimizers(self):
        return SGD(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        output = self(**batch)

        loss = output["loss"]
        tqdm_dict = {"train_loss": loss}
        result = OrderedDict({
            "loss": loss, "progress_bar": tqdm_dict, "log": tqdm_dict
        })
        return result

    def validation_step(self, batch, batch_idx):
        return self._val_and_test_step(batch)

    def validation_epoch_end(self, outputs: List):
        metrics = self._compute_epoch_end_metrics(outputs)

        tqdm_dict = {"val_{}".format(key): value for key, value in metrics.items()}
        return {
            "progress_bar": tqdm_dict,
            "log": tqdm_dict
        }

    def test_step(self, batch, batch_idx):
        return self._val_and_test_step(batch, stage="test")

    def test_epoch_end(self, outputs: List):
        metrics = self._compute_epoch_end_metrics(outputs)

        tqdm_dict = {"test_{}".format(key): value for key, value in metrics.items()}

        self._write_out(outputs, "TextCNN_{}-{}.json".format(self.num_class, self.random_seed))

        return {"progress_bar": tqdm_dict, "log": tqdm_dict}


class BiLSTMBaselineModel(ConversationRelationModel):
    """
    Re-implement BiLSTM proposed by `Cai et.al.(2016)`

    TODO: add entity type embedding like PER, LOC, ORG as described in `DocRED(2019)`.
    """

    def __init__(self, num_class: int, random_seed: int, learning_rate=1e-2, hidden_size=300, freeze_embedding=False):
        super().__init__(random_seed=random_seed)
        self.save_hyperparameters()

        self.lr = learning_rate
        self.num_class = num_class
        self.hidden_size = hidden_size

        embedding_weights = []
        with open("vocab_dir/glove.conrel.txt", "r") as f:
            for line in f.readlines():
                weights = [float(v) for v in line.strip().split()[1:]]
                embedding_weights.append(weights)
        embedding_weights = torch.tensor(embedding_weights)

        self.embedding = nn.Sequential(
            nn.Embedding.from_pretrained(
                embedding_weights,
                padding_idx=0,
                freeze=freeze_embedding),
            nn.Dropout(0.3),
        )

        self.lstm = nn.LSTM(
            input_size=300,
            hidden_size=self.hidden_size,
            num_layers=1,
            bidirectional=True)

        self.lstm_dropout = nn.Dropout(0.3)

        self.attention_weight = nn.Parameter(torch.randn(1, 2 * hidden_size))

        self.classification_linear = nn.Sequential(
            nn.Linear(in_features=self.hidden_size * 2, out_features=num_class),
            nn.Dropout(0.5),
            nn.LogSoftmax(dim=-1),
        )

    def forward(self, *args, **kwargs):
        x = self.embedding(kwargs["encoded_context"])
        bsz, sent_len, word_dim = x.shape

        x = x.reshape(sent_len, bsz, word_dim)

        # output: (seq_len, bsz, num_directions x hidden_size)
        output, (hn, cn) = self.lstm(x)
        _, _, hidden = output.shape

        # Attention accordingt to Zhou et.al (2016)
        output = self.lstm_dropout(output)
        mask = torch.tensor(kwargs["attention_mask"]).unsqueeze(-1).repeat([1, 1, hidden]).reshape(bsz, -1, sent_len)
        H = output.reshape(bsz, -1, sent_len) * mask
        M = torch.tanh(H)

        # alpha: (bsz, 1, sent_len)
        alpha = torch.softmax(torch.bmm(self.attention_weight.repeat(bsz, 1).reshape(bsz, 1, -1), M), dim=-1)
        # hidden_out: (bsz, 2 * hidden)
        hidden_out = torch.tanh(torch.bmm(H, alpha.reshape(bsz, sent_len, -1))).squeeze(dim=-1)

        y_ = self.classification_linear(hidden_out)
        y = kwargs["label"]

        self.nllloss = nn.NLLLoss()

        loss = self.nllloss(y_, y)

        return {"logits": y_, "loss": loss}

    def configure_optimizers(self):
        # return SGD(self.parameters(), lr=self.lr)
        # following Zhou et.al. (2016) L2 regularization strength of 1e-5
        return Adadelta(self.parameters(), weight_decay=1e-5)

    def training_step(self, batch, batch_idx):
        output = self(**batch)
        loss = output["loss"]

        tqdm_dict = {"train_loss": loss}
        return {
            "loss": loss, "progress_bar": tqdm_dict, "log": tqdm_dict,
        }

    def validation_step(self, batch, batch_idx):
        return self._val_and_test_step(batch)

    def validation_epoch_end(self, outputs: List):
        metrics = self._compute_epoch_end_metrics(outputs)
        tqdm_dict = {"val_{}".format(key): value for key, value in metrics.items()}
        return {
            "progress_bar": tqdm_dict,
            "log": tqdm_dict,
        }

    def test_step(self, batch, batch_idx):
        return self._val_and_test_step(batch, stage="test")

    def test_epoch_end(self, outputs: List):
        metrics = self._compute_epoch_end_metrics(outputs)

        tqdm_dict = {"test_{}".format(key): value for key, value in metrics.items()}

        self._write_out(outputs, "LSTM_{}-{}.json".format(self.num_class, self.random_seed))

        return {
            "progress_bar": tqdm_dict,
            "log": tqdm_dict,
        }


def test_forward_pass():
    model = BertBaselineModel()
    x = torch.LongTensor(4, 128).random_(0, 100)
    y = torch.LongTensor([1, 2, 3, 1])
    input = {"encoded_context": x, "turn_type_ids": x, "attention_mask": x,
             "4-label": y, "6-label": y, "13-label": y
             }
    y = model(**input)
    print(y.keys())
    print(y["4-label-logits"].shape)
    print(y["6-label-logits"].shape)
    print(y["13-label-logits"].shape)

    return model, y
