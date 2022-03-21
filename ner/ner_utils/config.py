"""
config.py: 参数的配置文件,不同训练任务,仅需要对这部分做出更改即可.
by: qliu
update date: 2021-12-17
"""
from multiprocessing import cpu_count
import torch
from itertools import chain
from transformers import BertTokenizer
from .common_utils import read_json

class CommonConfig(object):
    """
    不管使用哪个模型，都通用的配置参数
    """
    def __init__(self):
        # 各类文件路径
        self.entities_file = 'data/original_data/entities.json'   # 以实体中英文分别作为 key & value 的 json 文件
        self.bert_model_dir = "hfl/chinese-bert-wwm-ext"
        self.dataset_dir = "data/model_data"
        self.checkpoint_dir = "data/result"
        # 存放 daccano 生成的所有结果（格式需要为 xxx.jsonl）
        self.jsonl_files_dir = "data/original_data/jsonl" 
        self.server_ports_file = "data/original_data/ports.json"

        # 数据大小 & 形式
        # (train, test, dev)的分割比例
        self.train_data_split = (0.6, 0.2, 0.2)
        self.max_seq_length = 256
        self.batch_size = 16
        self.do_lower_case = True
        self.shuffle_train_data = True
        self.pin_memory = False

        # 模型训练环境
        self.use_gpu = False
        self.gpu = 0
        self.num_workers = cpu_count()
        self.device = self._check_gpu()

        # 自定义实体字典
        self.entities_abbr_dict = self._get_entities_dict()
        self.abbr_entities_dict = {v: k for k, v in self.entities_abbr_dict.items()}
        self.labels_types = self._get_labels_types()
        self.labels_idx_dict = self._get_labels_idx()
        self.idx_labels_dict = self._get_idx_labels()
        self.labels_num = len(self.labels_types)

        # 分字器
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_model_dir, do_lower_case=self.do_lower_case)

    def update(self, **kwargs):
        """
        参数更新（通过输入，指定要更新的参数）
        """
        for k, v in kwargs.items():
            setattr(self, k, v)

    def _check_gpu(self):
        if self.use_gpu and torch.cuda.is_available():
            return "cuda:{}".format(self.gpu)
        return "cpu"

    def _get_entities_dict(self):
        return read_json(self.entities_file)

    def _get_labels_types(self):
        entities_abbr_list = list(self.entities_abbr_dict.values())
        labels_types = ["X", "START", "END", "O"] + list(chain(*[["B_{}".format(abbr), "I_{}".format(abbr)] for abbr in entities_abbr_list]))
        return labels_types

    def _get_labels_idx(self):
        return {l: i for i, l in enumerate(self.labels_types)}

    def _get_idx_labels(self):
        return {i: l for i, l in enumerate(self.labels_types)}

    def __str__(self):
        return '\n'.join(['%s:%s' % item for item in self.__dict__.items()])

class ModelConfig(object):
    """
    模型的训练＆优化参数，不同模型，其配置可能不一样
    """
    def __init__(self):
        # 模型结果文件名
        self.bert_model_name = "bert.pt"
        self.bert_crf_model_name = "bert_crf.pt"
        self.bert_bilstm_crf_model_name = "bert_bilstm_crf.pt"

        # 模型结构参数
        self.dropout = 0.2
        self.bert_hidden_size = 768
        self.lstm_hidden_size = 64
        self.lstm_layers = 1

        # 模型训练次数 & 是否加载先前训练结果
        self.load_checkpoint = False
        self.total_train_epochs = 20

        # 优化器参数
        self.lr = 5e-5
        self.crf_fc_lr = 8e-5
        self.weight_decay_finetune = 1e-5
        self.crf_fc_weight_decay = 5e-6
        self.gradient_accumulation_steps = 1
        self.warmup_proportion = 0.1

    def update(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __str__(self):

        return '\n'.join(['%s:%s' % item for item in self.__dict__.items()])