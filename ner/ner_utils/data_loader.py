"""
data_loader.py: 将 token & label 对应的数据转换成可喂入模型的数据. 
包含: 批数据生成器，以句子为轴的数据生成器
by: qliu
update date: 2021-12-16
"""
import torch, os, re
from torch.utils import data

class BertInputData(object):
    """
    bert 模型的一组输入数据（自用）
    Args:
        guid: 各组数据唯一 id
        tokens: 给定 sent 经 bert tokenize 后的 tokens
        ner_labels: 各 token 对应的 ner label
    """
    def __init__(self, guid, tokens, ner_labels):
        self.guid = guid
        self.tokens = tokens
        self.ner_labels = ner_labels

class BertInputFeatures(object):
    """
    BertInputData 转换后的特征值（自用）
    """
    def __init__(self, input_ids, input_mask, segment_ids, predict_mask, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.predict_mask = predict_mask
        self.label_ids = label_ids

class NerDataLoader(object):
    """
    数据生成器（外部调用）
    """
    def __init__(self, config):
        """
        Args:
            config: config.py 中的 CommonConfig
        """
        self.config = config

    def _read_tokens_txt(self, file_path):
        """
        将 以"领 O B_V B_V"为行的数据转换成可喂入模型的数据格式 
        """
        with open(file_path, "r+") as fp:
            dataset = []
            entries = fp.read().strip().split("\n\n")
            for entry in entries:
                tokens = []
                ner_labels = []
                for line in entry.splitlines():
                    segs = line.strip().split()
                    if len(segs) != 0:
                        tokens.append(segs[0])
                        ner_labels.append(segs[1])
                dataset.append([tokens, ner_labels])
        return dataset

    def _create_input_data(self, dataset):
        """
        生成模型喂入数据
        Args:
            dataset: list, [(tokens, labels), (tokens, labels), ...]
        """
        input_datas = []
        for i, data in enumerate(dataset):
            tokens = data[0]
            ner_labels = data[1]
            input_datas.append(BertInputData(guid=i, tokens=tokens, ner_labels=ner_labels))
        return input_datas

    def get_train_data(self):
        if not os.path.exists(os.path.join(self.config.dataset_dir, "train.txt")):
            raise Exception("none train txt in given dir ...")
        return self._create_input_data(self._read_tokens_txt(os.path.join(self.config.dataset_dir, "train.txt")))

    def get_test_data(self):
        if not os.path.exists(os.path.join(self.config.dataset_dir, "test.txt")):
            raise Exception("none test txt in given dir ...")
        return self._create_input_data(self._read_tokens_txt(os.path.join(self.config.dataset_dir, "test.txt")))

    def get_dev_data(self):
        if not os.path.exists(os.path.join(self.config.dataset_dir, "dev.txt")):
            raise Exception("none dev txt in given dir ...")
        return self._create_input_data(self._read_tokens_txt(os.path.join(self.config.dataset_dir, "dev.txt")))

def data2feature(input_data, labels_to_idx, max_seq_length, tokenizer):
    """
    将输入数据转换成可喂入 bert 的特征向量（自用）
    """
    extra_label = "X"  # 不需要进行预测＆效果评估的标签
    tokens = ["START"]
    predict_mask = [0]
    label_ids = [labels_to_idx["START"]]
    for i, w in enumerate(input_data.tokens):
        tokens.append(w)
        if re.search("##", w):
            # ##me 等 token 不需要进行预测 ,不计入效果统计
            # 之后可直接还原
            predict_mask.append(0)
            label_ids.append(labels_to_idx[extra_label])
        else:
            predict_mask.append(1)
            label_ids.append(labels_to_idx[input_data.ner_labels[i]])

    if len(tokens) > max_seq_length - 1:
        tokens = tokens[0: max_seq_length-1]
        predict_mask = predict_mask[0: max_seq_length-1]
        label_ids = label_ids[0: max_seq_length-1]
    tokens.append("END")
    predict_mask.append(0)
    label_ids.append(labels_to_idx['END'])

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    # 0 表示 sentence_a, 1 表示 sentence_b
    segment_ids = [0] * len(input_ids)
    # 1 表示有效的 token，0表示 padded value
    input_mask = [1] * len(input_ids)

    feature = BertInputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        predict_mask=predict_mask,
        label_ids=label_ids
    )

    return feature

class NerBatchDataset(data.Dataset):
    def __init__(self, input_data_list, config):
        """
        批数据生成器（外部调用）
        Args:
            input_data_list: list, [BertInputData, BertInputData, ...]
            tokenizer: 分词器，能将 sent 转换为 tokens
        """
        self.input_data_list = input_data_list
        self.config = config

    def __len__(self):
        return len(self.input_data_list)

    def __getitem__(self, idx):
        """
        data.Dataset 的继承函数
        依次获取样本数据
        """
        feature = data2feature(
            input_data=self.input_data_list[idx],
            labels_to_idx=self.config.labels_idx_dict,
            max_seq_length=self.config.max_seq_length,
            tokenizer=self.config.tokenizer
        )
        return feature.input_ids, feature.input_mask, feature.segment_ids, feature.predict_mask, feature.label_ids

    @classmethod
    def padding(cls, batch_dataset):
        """
        同批次数据中，以最长句为标准，对其他句子添加 padding，使同批次样本数据格式一致
        batch_dataset: list, [BertInputData, BertInputData, ...]
        """
        data_length_list = [len(data[0]) for data in batch_dataset]
        max_length = max(data_length_list)

        f = lambda x, seq_length: [data[x] + [0] * (seq_length - len(data[x])) for data in batch_dataset]
        input_ids_list = torch.LongTensor(f(0, max_length))
        input_mask_list = torch.LongTensor(f(1, max_length))
        segment_ids_list = torch.LongTensor(f(2, max_length))
        predict_mask_list = torch.ByteTensor(f(3, max_length))
        label_ids_list = torch.LongTensor(f(4, max_length))
        
        return input_ids_list, input_mask_list, segment_ids_list, predict_mask_list, label_ids_list

class SentNerData(object):
    """
    将句子转换为可喂入模型的输入数据（仅用于标签预测，外部调用）
    """
    def __init__(self, config):
        """
        Args:
            config: config.py 中的 CommonConfig
        """
        self.config = config

    def _get_sent_input_data(self, sent):
        tokens = self.config.tokenizer.tokenize(sent)
        # 所有 token 标签都自动赋值 'O'，预测时，输入的标签无意义
        labels = ["O"] * len(tokens)
        sent_input_data = BertInputData(guid=1, tokens=tokens, ner_labels=labels)
        sent_feature = data2feature(input_data=sent_input_data, labels_to_idx=self.config.labels_idx_dict, max_seq_length=self.config.max_seq_length, tokenizer=self.config.tokenizer)
        input_ids, input_mask, segment_ids, predict_mask, label_ids = sent_feature.input_ids, sent_feature.input_mask, sent_feature.segment_ids, sent_feature.predict_mask, sent_feature.label_ids
        input_ids = torch.LongTensor([input_ids])
        input_ids.to(self.config.device)
        input_mask = torch.LongTensor([input_mask])
        input_mask.to(self.config.device)
        segment_ids = torch.LongTensor([segment_ids])
        segment_ids.to(self.config.device)
        predict_mask = torch.LongTensor([predict_mask]) 
        predict_mask.to(self.config.device)
        label_ids = torch.LongTensor([label_ids]) 
        label_ids.to(self.config.device)
        return input_ids, input_mask, segment_ids, predict_mask, label_ids

class NerBatchDataloader(data.Dataset):
    def __init__(self, config):
        """
        Args:
            config: config.py 中的 CommonConfig
        """
        self.config = config
        self.ner_dataloader = NerDataLoader(self.config)

    def get_train_dataloader(self):
        train_dataset = self.ner_dataloader.get_train_data()
        train_dataset = NerBatchDataset(train_dataset, self.config)
        train_dataloader = data.DataLoader(
            dataset=train_dataset,
            batch_size=self.config.batch_size,
            shuffle=self.config.shuffle_train_data,
            num_workers=self.config.num_workers,
            collate_fn=NerBatchDataset.padding,
            pin_memory=self.config.pin_memory
        )
        return train_dataloader

    def get_test_dataloader(self):
        test_dataset = self.ner_dataloader.get_test_data()
        test_dataset = NerBatchDataset(test_dataset, self.config)
        test_dataloader = data.DataLoader(
            dataset=test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=NerBatchDataset.padding,
            pin_memory=self.config.pin_memory
        )
        return test_dataloader

    def get_dev_dataloader(self):
        dev_dataset = self.ner_dataloader.get_dev_data()
        dev_dataset = NerBatchDataset(dev_dataset, self.config)
        dev_dataloader = data.DataLoader(
            dataset=dev_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=NerBatchDataset.padding,
            pin_memory=self.config.pin_memory
        )
        return dev_dataloader