import json
import logging
import torch, os

from utils import ClassicalTokenizer
from typing import List, Callable, Dict
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer, BertTokenizer, AutoTokenizer

class ConversationPairDataset(Dataset):
    """
    Each sample is the sessions between each pair of speakers.
    Correspondence between label numbers and relationship categories:

    * 1	--	Child-Parent
    * 2	--	Child-Other Family Elder
    * 3	--	Siblings
    * 4	--	Spouse
    * 5	--	Lovers
    * 6	--	Courtship
    * 7	--	Friends
    * 8	--	Neighbors
    * 9	--	Roommates
    * 10	--	Workplace Superior - Subordinate
    * 11	--	Colleague/Partners
    * 12	--	Opponents
    * 13	--	Professional Contact

    Statistics:
        - max # of token: 2063 (after encoded)
        - # of too long instance: 83
        - avg # of token: 137.9632
    """

    def __init__(self, dataset: str,
                 num_class: int,
                 preprocess_func: Callable,
                 tokenizer: PreTrainedTokenizer):

        self.dataset = []
        self.tokenizer = tokenizer

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

        assert num_class in [4, 6, 13], "only support 4, 6, 13 classes!"

        # TODO: cluster the samples based on pair-id, save samples into self.dataset
        prev_pair_id = None

        with open(dataset, "r") as f:

            pair_id,label = None, None
            encoded_context, turn_type_ids, attention_mask = [],[],[]

            dataset_tmp=[]

            for sample in f.readlines():
                sample = json.loads(sample)

                pair_id = sample['pair-id']

                if prev_pair_id!=None and pair_id!=prev_pair_id:
                    dataset_tmp.append({
                        "pair-id": int(prev_pair_id),
                        "num_class": num_class,
                        "encoded_context": encoded_context,
                        "turn_type_ids": turn_type_ids,
                        "attention_mask": attention_mask,
                        "session-id": len(encoded_context),  # session_num
                        "label": label
                    })
                    encoded_context, turn_type_ids, attention_mask, label = [], [], [], []

                prev_pair_id = pair_id
                context = sample["context"]
                encoded_context_tmp = preprocess_func(context, self.tokenizer)
                encoded_context.append(encoded_context_tmp['input_ids'])
                turn_type_ids.append(encoded_context_tmp['turn_type_ids'])
                attention_mask.append(encoded_context_tmp['attention_mask'])

                if num_class == 4:
                    label=four_class[int(sample["label"])]
                elif num_class == 6:
                    label=six_class[int(sample["label"])]
                else:
                    label=int(sample["label"]) - 1

            dataset_tmp.append({
                "pair-id": int(prev_pair_id),
                "num_class": num_class,
                "encoded_context": encoded_context, # list of sessions
                "turn_type_ids": turn_type_ids, # list of sessions
                "attention_mask": attention_mask, # list of sessions
                "session-id": len(encoded_context), # session_num
                "label": label
            })
        self.dataset = self.data_process(dataset_tmp)
        # print(self.dataset[0])
        print("- finished loading {} examples".format(len(self.dataset)))
        # exit(0)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        return self.dataset[idx]

    def sliding_window_augment(self, sample: Dict, ratio: int=3) -> List[Dict]:

        data_to_return = []

        for i, session in enumerate(sample["encoded_context"]):

            encoded_context = [101]
            turn_type_ids = [1]
            attention_mask = [1]

            for i, (turn, ttid, attn) in enumerate(zip(
                session,
                sample["turn_type_ids"][i],
                sample["attention_mask"][i]
            )):
                encoded_context += turn[1:]
                turn_type_ids += ttid[1:]
                attention_mask += attn[1:]

            assert len(encoded_context) == len(turn_type_ids) == len(attention_mask), \
                "{} - {} - {}".format(len(encoded_context), len(turn_type_ids), len(attention_mask))
            
            encoded_context = encoded_context[:min(512,len(encoded_context))]
            turn_type_ids = turn_type_ids[: min(512,len(turn_type_ids))]
            attention_mask = attention_mask[: min(512,len(attention_mask))]
            data_to_return.append({
                "pair-id": sample["pair-id"],
                "num_class": sample["num_class"],
                "encoded_context": encoded_context,
                "turn_type_ids": turn_type_ids,
                "attention_mask": attention_mask,
                "session-id": 1,
                "label": sample["label"],
            })
        
        for i in range(len(sample["encoded_context"]) - 1):
            l1 = len(sample["encoded_context"][i]) // ratio
            l2 = len(sample["encoded_context"][i+1]) // ratio

            for j in range(ratio-1):
                encoded_context = sample["encoded_context"][i][l1 * (j+1):]+sample["encoded_context"][i+1][:l2 * (j+1)]
                turn_type_ids = sample["turn_type_ids"][i][l1*(j+1):]+sample["turn_type_ids"][i+1][:l2*(j+1)]
                attention_mask = sample["attention_mask"][i][l1*(j+1):]+sample["attention_mask"][i+1][:l2*(j+1)]

                pro_encoded_context = [101]
                pro_turn_type_ids = [1]
                pro_attention_mask = [1]
                
                for j,(turn, ttid,attn) in enumerate(zip(
                    encoded_context,
                    turn_type_ids,
                    attention_mask
                )):
                    pro_encoded_context += turn[1:]
                    pro_turn_type_ids += ttid[1:]
                    pro_attention_mask += attn[1:]

                assert len(pro_encoded_context) == len(pro_turn_type_ids) == len(pro_attention_mask), \
                    "{} - {} - {}".format(len(pro_encoded_context), len(pro_turn_type_ids), len(pro_attention_mask))
                
                pro_encoded_context = pro_encoded_context[:min(512,len(pro_encoded_context))]
                pro_turn_type_ids = pro_turn_type_ids[:min(512,len(pro_turn_type_ids))]
                pro_attention_mask = pro_attention_mask[:min(512, len(pro_attention_mask))]

                data_to_return.append({
                    "pair-id": sample["pair-id"],
                    "num_class": sample["num_class"],
                    "encoded_context":pro_encoded_context,
                    "turn_type_ids": pro_turn_type_ids,
                    "attention_mask":pro_attention_mask,
                    "session-id": 1,
                    "label": sample["label"],
                })
        return data_to_return


    def data_process(self, samples):
        augmented_samples = []
        for sample in samples:
            augmented_samples.extend(self.sliding_window_augment(sample))
        return augmented_samples

class ConversationRelDataset(Dataset):
    """
    Correspondence between label numbers and relationship categories:

    * 1	--	Child-Parent
    * 2	--	Child-Other Family Elder
    * 3	--	Siblings
    * 4	--	Spouse
    * 5	--	Lovers
    * 6	--	Courtship
    * 7	--	Friends
    * 8	--	Neighbors
    * 9	--	Roommates
    * 10	--	Workplace Superior - Subordinate
    * 11	--	Colleague/Partners
    * 12	--	Opponents
    * 13	--	Professional Contact

    Statistics: 
        - max # of token: 2063 (after encoded)
        - # of too long instance: 83
        - avg # of token: 137.9632
    """
    def __init__(self, dataset: str, 
                       num_class: int,
                       preprocess_func: Callable,
                       tokenizer: PreTrainedTokenizer):

        self.dataset = []
        self.tokenizer = tokenizer

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

        assert num_class in [4, 6, 13], "only support 4, 6, 13 classes!"

        with open(dataset, "r") as f:
            for sample in f.readlines():
                sample = json.loads(sample)
                context = sample["context"]

                encoded_context = preprocess_func(context, self.tokenizer)
                self.dataset.append({
                    # "raw_context": context, # no need when training, just for debug
                    "pair-id": int(sample["pair-id"]),
                    "session-id": int(sample["session-id"]),
                    "num_class": num_class,
                    "encoded_context": encoded_context["input_ids"],
                    "turn_type_ids": encoded_context["turn_type_ids"],
                    "attention_mask": encoded_context["attention_mask"],
                })

                if num_class == 4:
                    self.dataset[-1]["label"] = four_class[int(sample["label"])]
                elif num_class == 6:
                    self.dataset[-1]["label"] = six_class[int(sample["label"])]
                else:
                    self.dataset[-1]["label"] = int(sample["label"]) - 1
        print("- finished loading {} examples".format(len(self.dataset)))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        return self.dataset[idx]

class ConversationRelPreprocessor(object):
    def __init__(self):
        pass

    @staticmethod
    def bert_preprocess(context: List[str], tokenizer: PreTrainedTokenizer) -> Dict[str, List[int]]:
        """
        Convert a string in a sequence of ids, using the tokenizer and vocabulary.
        """
        context = [tokenizer.encode(c) for c in context]
        
        turn_type_ids = [1]

        # TODO: use token_type_id of bert model may be not so correct.
        # ignore A: and B: in the dialogue.

        # to reproduce, add A: and B: now.
        for c in context:
            # turn_id = 0 if c[1] == 138 else 1   # A := 138 B:=139
            turn_id = 1
            # turn_type_ids.extend([turn_id for _ in range(len(c)-3)])
            turn_type_ids.extend([turn_id for _ in range(len(c)-1)])

        # input_ids = [101] + [id for c in context for id in c[3:]]
        input_ids = [101] + [id for c in context for id in c[1:]]
        attn_mask = [1 for _ in range(len(input_ids))]
        
        original_len = len(input_ids)
        input_ids = input_ids[: min(512, original_len)]
        turn_type_ids = turn_type_ids[: min(512, original_len)]
        attn_mask = attn_mask[: min(512, original_len)]
        return {
            "input_ids": input_ids,
            "turn_type_ids": turn_type_ids,
            "attention_mask": attn_mask
        }
    
    @staticmethod
    def bert_fixed_sliding_window_process(context: List[str], tokenizer: PreTrainedTokenizer) -> Dict[str, List[int]]:
        context = [tokenizer.encode(c) for c in context]

        turn_type_ids = []
        attention_mask = []

        for c in context:
            turn_type_ids.append([1 for _ in range(len(c))])
            attention_mask.append([1 for _ in range(len(c))])

        input_ids = context
        # input_ids: [[cls, A: ...], [cls, B: ...],...]
        # attention_mask: [[1, 1, ...], [1,1, ...], ]
        # token_type_ids: [[1, 1, ...], [1,1, ...], ]
        return {
            "input_ids": input_ids,
            "turn_type_ids": turn_type_ids,
            "attention_mask": attention_mask,
        }

    @staticmethod
    def cnn_preprocess(context: List[str], tokenizer: ClassicalTokenizer) -> List[int]:
        """
        Convert a string in a sequence of ids, using the space tokenizer and glove embedding
        """

        context = [tokenizer.encode(c) for c in context]
        
        input_ids = [id for c in context for id in c]

        return {
            "input_ids": input_ids,
            # dummy input for convenience
            "turn_type_ids": [0 for _ in range(len(input_ids))],
            "attention_mask": [0 for _ in range(len(input_ids))]
        }

    def rnn_preprocess(context: List[str], tokenizer: ClassicalTokenizer) -> List[int]:
        """
        Convert a string in a sequence of ids, using the space tokenizer and glove embedding
        """

        context = [tokenizer.encode(c) for c in context]
        
        input_ids = [id for c in context for id in c]
        original_len = len(input_ids)

        return {
            "input_ids": input_ids[: min(512, original_len)],
            "turn_type_ids": [0 for _ in range(min(512, original_len))],
            # "last_hidden_idx": min(511, original_len-1),
            # attention mask is necessary in LSTM
            "attention_mask": [1 for _ in range(min(512, original_len))]
        }

def collator(minibatch_data: List) -> Dict[str, torch.Tensor]:
    padding_value = 0
    batch_size = len(minibatch_data)

    data_to_return = {key: [] for key in minibatch_data[0].keys()}

    max_len = max([len(minibatch_data[i]["encoded_context"]) for i in range(batch_size)])

    for i in range(batch_size):
        for key in ["encoded_context", "turn_type_ids", "attention_mask"]:
            cur_len = len(minibatch_data[i][key])
            minibatch_data[i][key] += [padding_value for _ in range(max_len - cur_len)]
    
            data_to_return[key].append(minibatch_data[i][key])
    
        data_to_return["num_class"].append(minibatch_data[i]["num_class"])
        data_to_return["label"].append(minibatch_data[i]["label"])
        data_to_return["pair-id"].append(minibatch_data[i]["pair-id"])
        data_to_return["session-id"].append(minibatch_data[i]["session-id"])

    for key, value in data_to_return.items():
        data_to_return[key] = torch.Tensor(value).to(torch.long)
    return data_to_return

class ConversationRelDataModule(LightningDataModule):
    """
    Data preparation in PyTorch follows 5 steps:

    1. Download / tokenize / process.
    2. Clean and (maybe) save to disk.
    3. Load inside Dataset.
    4. Apply transforms (rotate, tokenize, etc…).
    5. Wrap inside a DataLoader.

    A DataModule is simply a collection of a train_dataloader, val_dataloader(s), 
    test_dataloader(s) along with the matching transforms and data processing/downloads 
    steps required.
    """
    def __init__(self, 
                 num_class: int = 13,
                 data_dir: str = "ddrel/",
                 tokenizer: PreTrainedTokenizer = None,
                 batch_size: int = 4,
                 preprocessor: Callable = None,
                 collator: Callable = None,
    ):
        super().__init__()
        self.num_class = num_class
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.preprocessor = preprocessor
        self.collator = collator

        assert self.tokenizer is not None, "Must specify data tokenizer"
        assert self.collator is not None, "Must specify batch data collator"
        assert self.preprocessor is not None, "Must specify data pre-processor"
    
    def prepare_data(self):
        """ Read train, val and test file.
        """
        pass
    
    def setup(self, stage=None, data_augmentation=False):
        if data_augmentation==True:
            # Assign train/val datasets for use in dataloaders
            if stage == "fit" or stage is None:
                self.train_data = ConversationPairDataset(
                    self.data_dir + "train.txt",
                    self.num_class,
                    self.preprocessor,
                    self.tokenizer
                )
                self.val_data = ConversationPairDataset(
                    self.data_dir + "dev.txt",
                    self.num_class,
                    self.preprocessor,
                    self.tokenizer
                )

            # Assign test datasets for use in dataloader
            if stage == "test" or stage is None:
                self.test_data = ConversationPairDataset(
                    self.data_dir + "test.txt",
                    self.num_class,
                    self.preprocessor,
                    self.tokenizer
                )

        else:
            # Assign train/val datasets for use in dataloaders
            if stage == "fit" or stage is None:
                self.train_data = ConversationRelDataset(
                    self.data_dir + "train.txt",
                    self.num_class,
                    self.preprocessor,
                    self.tokenizer
                )
                self.val_data = ConversationRelDataset(
                    self.data_dir + "dev.txt",
                    self.num_class,
                    self.preprocessor,
                    self.tokenizer
                )

            # Assign test datasets for use in dataloader
            if stage == "test" or stage is None:
                self.test_data = ConversationRelDataset(
                    self.data_dir + "test.txt",
                    self.num_class,
                    self.preprocessor,
                    self.tokenizer
                )

    def train_dataloader(self):
        return DataLoader(
            self.train_data, 
            shuffle=True,
            num_workers=10,
            batch_size=self.batch_size, 
            collate_fn=self.collator)

    def val_dataloader(self):
        return DataLoader(
            self.val_data, 
            num_workers=10,
            batch_size=self.batch_size, 
            collate_fn=self.collator)

    def test_dataloader(self):
        return DataLoader(
            self.test_data, 
            num_workers=10,
            batch_size=self.batch_size, 
            collate_fn=self.collator)

def test_load_dataset_for_bert():
    data = ConversationRelDataset("ddrel/dev.txt",
        num_class=4,
        preprocess_func = ConversationRelPreprocessor.bert_preprocess, 
        tokenizer = BertTokenizer.from_pretrained("bert-base-cased"))

    data_loader = DataLoader(data, 
        batch_size=4, 
        shuffle = True,
        collate_fn = collator,
    )

    # max_num_token = 0
    # avg_num_token = 0.
    # long_num_token = 0
    # for datum in data:
    #     max_num_token = max(max_num_token, len(datum["encoded_context"]))
    #     avg_num_token += len(datum["encoded_context"])
    #     long_num_token += len(datum["encoded_context"]) > 512
    #     assert len(datum["encoded_context"]) == \
    #         len(datum["turn_type_ids"]) == len(datum["attention_mask"])

    # print("max # of token: ", max_num_token)
    # print("# of too long instance: ", long_num_token)
    # print("avg # of token: ", avg_num_token / len(data))

    e = next(iter(data_loader))
    return data, data_loader, e