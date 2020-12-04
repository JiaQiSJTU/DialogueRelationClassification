import data
import pytorch_lightning as pl

from data import collator
from argparse import ArgumentParser
from transformers import AutoTokenizer
from utils import ClassicalTokenizer
from model import BertBaselineModel, CNNBaselineModel, BiLSTMBaselineModel


def train_bert_baseline(args):
    # help when training
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    if args.data_augmentation:
        tmp_preprocessor = data.ConversationRelPreprocessor.bert_fixed_sliding_window_process
    else:
        tmp_preprocessor = data.ConversationRelPreprocessor.bert_preprocess

    bert_baseline = BertBaselineModel(
        random_seed=args.random_seed,
        num_class=args.num_class,
        learning_rate=args.learning_rate)
    dataset = data.ConversationRelDataModule(
        num_class=args.num_class,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        preprocessor=tmp_preprocessor,
        collator=collator
    )
    dataset.setup(stage="fit", data_augmentation=args.data_augmentation)

    trainer = pl.Trainer(gpus=[args.device],
                         max_epochs=args.max_epochs,
                         accumulate_grad_batches=args.accumulate_grad_batches,
                         early_stop_callback=False,  # default patience = 3
                         default_root_dir=args.default_root_dir,
                         )
    trainer.fit(bert_baseline,
                train_dataloader=dataset.train_dataloader(),
                val_dataloaders=dataset.val_dataloader(),
                )

    dataset.setup(stage="test", data_augmentation=args.data_augmentation)
    trainer.test(test_dataloaders=dataset.test_dataloader())


def test_bert_baseline(args):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    if args.data_augmentation:
        tmp_preprocessor = data.ConversationRelPreprocessor.bert_fixed_sliding_window_process
    else:
        tmp_preprocessor = data.ConversationRelPreprocessor.bert_preprocess

    bert_baseline = BertBaselineModel.load_from_checkpoint(args.checkpoint)
    dataset = data.ConversationRelDataModule(
        num_class=args.num_class,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        preprocessor=tmp_preprocessor,
        collator=collator
    )
    dataset.setup(stage="test")
    # dataset.prepare_data()

    trainer = pl.Trainer(gpus=[1])
    trainer.test(bert_baseline, test_dataloaders=dataset.test_dataloader())

    # compute_f1_macro("experiment_scripts/BERT_baseline.json", average_mode="macro")


def train_cnn_baseline(args):
    tokenizer = ClassicalTokenizer()

    cnn_baseline = CNNBaselineModel(
        random_seed=args.random_seed,
        num_class=args.num_class,
        learning_rate=args.learning_rate)
    dataset = data.ConversationRelDataModule(
        num_class=args.num_class,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        preprocessor=data.ConversationRelPreprocessor.cnn_preprocess,
        collator=collator
    )
    dataset.setup(stage="fit")

    trainer = pl.Trainer(gpus=[args.device],
                         max_epochs=args.max_epochs,
                         accumulate_grad_batches=args.accumulate_grad_batches,
                         early_stop_callback=True,  # default patience = 3
                         default_root_dir=args.default_root_dir,
                         )
    trainer.fit(cnn_baseline,
                train_dataloader=dataset.train_dataloader(),
                val_dataloaders=dataset.val_dataloader(),
                )

    dataset.setup(stage="test")
    trainer.test(test_dataloaders=dataset.test_dataloader())


def test_cnn_baseline(args):
    tokenizer = ClassicalTokenizer()

    cnn_baseline = CNNBaselineModel.load_from_checkpoint(args.checkpoint)
    dataset = data.ConversationRelDataModule(
        num_class=args.num_class,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        preprocessor=data.ConversationRelPreprocessor.cnn_preprocess,
        collator=collator
    )
    dataset.setup(stage="test")
    # dataset.prepare_data()

    trainer = pl.Trainer(gpus=[1])
    trainer.test(cnn_baseline, test_dataloaders=dataset.test_dataloader())


def train_lstm_baseline(args):
    tokenizer = ClassicalTokenizer()

    lstm_baseline = BiLSTMBaselineModel(
        random_seed=args.random_seed,
        num_class=args.num_class,
        learning_rate=args.learning_rate,
        hidden_size=args.hidden_size)
    dataset = data.ConversationRelDataModule(
        num_class=args.num_class,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        preprocessor=data.ConversationRelPreprocessor.rnn_preprocess,
        collator=collator
    )
    dataset.setup(stage="fit")

    trainer = pl.Trainer(gpus=[args.device],
                         max_epochs=args.max_epochs,
                         accumulate_grad_batches=args.accumulate_grad_batches,
                         early_stop_callback=True,  # default patience = 3
                         gradient_clip_val=0.5,
                         default_root_dir=args.default_root_dir,
                         )
    trainer.fit(lstm_baseline,
                train_dataloader=dataset.train_dataloader(),
                val_dataloaders=dataset.val_dataloader(),
                )

    dataset.setup(stage="test")
    trainer.test(test_dataloaders=dataset.test_dataloader())


def test_lstm_baseline(args):
    tokenizer = ClassicalTokenizer()

    lstm_baseline = BiLSTMBaselineModel.load_from_checkpoint(args.checkpoint)
    dataset = data.ConversationRelDataModule(
        num_class=args.num_class,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        preprocessor=data.ConversationRelPreprocessor.rnn_preprocess,
        collator=collator
    )
    dataset.setup(stage="test")
    # dataset.prepare_data()

    trainer = pl.Trainer(gpus=[1])
    trainer.test(lstm_baseline, test_dataloaders=dataset.test_dataloader())


if __name__ == "__main__":
    # experiment()
    parser = ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--stage", type=str, default="train", help="train or test")
    parser.add_argument("--device", type=int, default=0, help="GPU device")
    parser.add_argument("--checkpoint", type=str, default=None, help="checkpoint path")
    parser.add_argument("--model", type=str, default=None, help="cnn_baseline, lstm_baseline, or bert_baseline")
    parser.add_argument("--hidden_size", type=int, default=300, help="hidden size")
    parser.add_argument("--num_class", type=int, default=13, help="the number of classes")
    parser.add_argument("--random_seed", type=int, default=42, help="random seed")
    parser.add_argument("--data_augmentation", type=bool, default=False, help="data augmentation mentioned in the paper")

    parser = BertBaselineModel.add_model_specific_arguments(parser)
    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()
    pl.seed_everything(args.random_seed)

    if args.stage == "train":
        if args.model == "bert_baseline":
            train_bert_baseline(args)
        elif args.model == "cnn_baseline":
            train_cnn_baseline(args)
        elif args.model == "lstm_baseline":
            train_lstm_baseline(args)

    if args.stage == "test":
        if args.model == "bert_baseline":
            test_bert_baseline(args)
        elif args.model == "cnn_baseline":
            test_cnn_baseline(args)
        elif args.model == "lstm_baseline":
            test_lstm_baseline(args)
