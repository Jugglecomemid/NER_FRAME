"""
model.py: bert ner 模型
by: qliu
update date: 2021-12-17
"""
import os, torch
import torch.nn as nn
from transformers import BertForTokenClassification

class BertNer(nn.Module):
    def __init__(self, common_config, model_config):
        self.common_config = common_config
        self.model_config = model_config
        self.start_epoch = 0
        self.prev_acc_score = 0
        self.prev_f1_score = 0

    def get_model(self):
        """
        通过读取配置参数,加载 bert ner 模型
        """
        checkpoint_path = os.path.join(self.common_config.checkpoint_dir, self.model_config.bert_model_name)
        if self.model_config.load_checkpoint and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            self.start_epoch = checkpoint["epoch"] + 1
            self.prev_acc_score = checkpoint["acc_score"]
            self.prev_f1_score = checkpoint["f1_score"]
            model = BertForTokenClassification.from_pretrained(self.common_config.bert_model_dir,state_dict=checkpoint["model_state"], num_labels=self.common_config.labels_num)
            print("Loaded pretrain bert ner model, previous acc: {}, previous f1_score : {}".format(self.prev_acc_score, self.prev_f1_score))
        else:
            model = BertForTokenClassification.from_pretrained(self.common_config.bert_model_dir, num_labels=self.common_config.labels_num)
        return model