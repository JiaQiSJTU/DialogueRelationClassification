import json, torch
from model import BertBaselineModel
from data import ConversationRelPreprocessor
from pytorch_lightning import LightningModule
from transformers import AutoTokenizer
from typing import List, Dict, Callable

class InteractModel(object):
    def __init__(self, model: LightningModule, 
                       tokenizer: AutoTokenizer,
                       data_preprocessor: Callable):
        self.model = model
        self.tokenizer = tokenizer
        self.data_preprocessor = data_preprocessor

    def predict(self, sample: Dict):
        # read data input
        four_class = { 
            1: 0, 2: 0, 3: 0, 4: 0, 
            5: 1, 6: 1,
            7: 2, 8: 2, 9: 2,
            10: 3, 11: 3, 12: 3, 13: 3}
        six_class = {
            1: 0, 2: 0,
            3: 1, 4: 1, 
            5: 2, 6: 2, 
            7: 3, 8: 3, 9: 3,
            10: 4,
            11: 5, 12: 5, 13: 5
        }

        print("Input data pair-id: {}\tsession-id: {}".format(sample["pair-id"], sample["session-id"]))
        context = sample["context"]
        encoded_context = self.data_preprocessor(context, self.tokenizer)

        sample = {
            "raw_context": context,
            "encoded_context": torch.LongTensor(encoded_context["input_ids"]).reshape(1,-1),
            "turn_type_ids": torch.LongTensor(encoded_context["turn_type_ids"]).reshape(1,-1),
            "attention_mask": torch.LongTensor(encoded_context["attention_mask"]).reshape(1,-1),
            "4-label": three_class[int(sample["label"])],
            "6-label": five_class[int(sample["label"])],
            "13-label": int(sample["label"])-1,
        }

        print("=" * 20)
        for sent in sample["raw_context"]:
            print(sent)
        print("=" * 20)

        output = self.model(**sample)

        y4_ = torch.argmax(output["4-label-logits"], dim=-1).item()
        y6_ = torch.argmax(output["6-label-logits"], dim=-1).item()
        y13_ = torch.argmax(output["13-label-logits"], dim=-1).item()

        y4 = sample["4-label"]
        y6 = sample["6-label"]
        y13 = sample["13-label"]

        print("predicted 4-level label: {}\tgold label: {}".format(y4_, y4))
        print("predicted 6-level label: {}\tgold label: {}".format(y6_, y6))
        print("predicted 13-level label: {}\tgold label: {}\n".format(y13_, y13))

        return (y4_, y4), (y6_, y6), (y13_, y13)
    
    def load_pair_samples(self, pair_id: int) -> List[Dict]:
        samples = []
        with open("ddrel/test.txt", "r") as f:
            for sample in f.readlines():
                sample = json.loads(sample)

                if int(sample["pair-id"]) == pair_id:
                    samples.append(sample)
        return samples

def interact_with_model():
    bert_baseline = BertBaselineModel.load_from_checkpoint(
        "lightning_logs/version_3/checkpoints/epoch=27.ckpt"
    )
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    model = InteractModel(
        model=bert_baseline,
        tokenizer=tokenizer,
        data_preprocessor=ConversationRelPreprocessor.bert_preprocess
    )

    samples = model.load_pair_samples(234)
    for sample in samples:
        model.predict(sample)

interact_with_model()
        