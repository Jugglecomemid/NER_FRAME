"""
model.py: BERT BILSTM CRF NER 模型模块
by: qliu
update date: 2021-12-17
"""
import torch, os
import torch.nn as nn
from torch.autograd import Variable
from transformers import BertModel
from ..ner_utils.calculate import log_sum_exp_batch

class BertBiLstmCrf(nn.Module):
    def __init__(self, bert_model, labels_idx_dict, batch_size, dropout, hidden_size, lstm_hidden_size, lstm_layers, device):
        """
        构建基础 bert 模型,输出为 bert 中的 hidden states,其上可加各类任务模型.
        """
        super(BertBiLstmCrf, self).__init__()
        self.hidden_size = hidden_size
        self.labels_idx_dict = labels_idx_dict
        self.labels_num = len(self.labels_idx_dict)
        self.batch_size = batch_size
        self.device = device
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_layers = lstm_layers

        self.bert_model = bert_model
        # 增加 lstm 层
        self.lstm = nn.LSTM(hidden_size, self.lstm_hidden_size,self.lstm_layers, batch_first=True, bidirectional=True)

        self.dropout = torch.nn.Dropout(dropout)
        self.full_connected_layer = nn.Linear(self.lstm_hidden_size*2, self.labels_num)

        # 定义状态转移矩阵
        # entry i, j 表示 从 j 转移至 i 的概率
        self.transitions = nn.Parameter(torch.randn(self.labels_num, self.labels_num))
        self.transitions.data[self.labels_idx_dict["START"], :] = -1000
        self.transitions.data[:, self.labels_idx_dict["END"]] = -1000

        nn.init.xavier_uniform_(self.full_connected_layer.weight)
        nn.init.constant_(self.full_connected_layer.bias, 0.0)

    def _get_lstm_features(self, input_ids, segment_ids, input_mask):
        """
        获取 sentence 经由 bert -> lstm 得到的隐藏层特征值
        Args:
            sentence: str, 要获取特征值的句子
        """
        outputs = self.bert_model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask)
        last_hidden_state = outputs.last_hidden_state
        batch_size = last_hidden_state.shape[0]
        seq_length = last_hidden_state.shape[1]
        a, b = torch.randn(2 * self.lstm_layers, batch_size, self.lstm_hidden_size).to(self.device), torch.randn(2 * self.lstm_layers,batch_size, self.lstm_hidden_size).to(self.device)
        hidden = (a, b)
        lstm_out, hidden = self.lstm(last_hidden_state, hidden)
        lstm_out = lstm_out.contiguous().view(-1, self.lstm_hidden_size*2)
        lstm_out = self.dropout(lstm_out)
        # last_hidden_states = last_hidden_states.cpu().detach()
        lstm_out = self.full_connected_layer(lstm_out)
        out = lstm_out.contiguous().view(batch_size, seq_length, -1)
        return out

    def _forward_alg(self, features):
        """
        crf 前向计算，计算分区函数 Z(x)
        features: torch.long, 喂入 crf 模型的特征向量，shape (batch_size, sentence_words_num, tags_num)
        P(y|x) = exp{F(y,x)}/Z(x) -> log(P(y|x)) = log(exp{F(y,x)}) - log(Z(x)) -> P(y|x) 最大，即 Z(x) 最小 (整体类似贝叶斯函数)
        """
        Num = features.shape[1]
        batch_size = features.shape[0]

        init_alphas = torch.full(size=(batch_size, 1, self.labels_num), fill_value=-1000).to(self.device)
        # start 不能转出至 start
        init_alphas[:, 0, self.labels_idx_dict['START']] = 0

        forward_variable = init_alphas
        for t in range(1, Num):
            forward_variable = (log_sum_exp_batch(self.transitions + forward_variable, axis=-1) + features[:, t]).unsqueeze(1)

        # 加上句子结束概率
        forward_variable = forward_variable + self.transitions[self.labels_idx_dict["END"]]
        alpha = log_sum_exp_batch(forward_variable)
        return alpha

    def _get_sentence_tags_score(self, features, label_ids):
        """
        给定句子，计算得到给定标签序列的可能性
        """
        Num = features.shape[1]
        batch_size = features.shape[0]

        batch_transitions = self.transitions.expand(batch_size, self.labels_num, self.labels_num)
        batch_transitions = batch_transitions.flatten(1)

        score = torch.zeros((features.shape[0],1)).to(self.device)
        for t in range(1, Num):
            # 行为转入，列为转出
            score = score + batch_transitions.gather(-1, (label_ids[:, t]*self.labels_num + label_ids[:, t-1]).view(-1, 1)) + features[:, t].gather(-1, label_ids[:, t].view(-1, 1)).view(-1, 1)
        return score

    def _viterbi_decode(self, features):
        """
        维特比算法
        基本等同于前向算法 + 反向得路径
        """
        Num = features.shape[1]
        batch_size = features.shape[0]

        path_nodes_pointers = []

        init_path_variables = torch.full((batch_size, 1, self.labels_num), -1000).to(self.device)
        init_path_variables[:, 0, self.labels_idx_dict["START"]] = 0

        forward_variables = init_path_variables
        # 路径上各节点的最大值
        path_nodes_pointers = torch.zeros((batch_size, Num, self.labels_num), dtype=torch.long).to(self.device)
        for t in range(1, Num):
            forward_variables, path_nodes_pointers[:, t] = torch.max(self.transitions + forward_variables, -1)
            forward_variables = (forward_variables + features[:, t]).unsqueeze(1)

        best_path = torch.zeros((batch_size, Num), dtype=torch.long).to(self.device)
        best_path_score, best_path[:, -1] = torch.max(forward_variables.squeeze(), -1)
        for t in range(Num-2, -1, -1):
            best_path[:, t] = path_nodes_pointers[:, t+1].gather(-1, best_path[:, t+1].view(-1, 1)).squeeze()

        return best_path_score, best_path

    def negate_log_likelihood(self, input_ids, segment_ids, input_mask, label_ids):
        """
        计算损失函数
        """
        lstm_features = self._get_lstm_features(input_ids, segment_ids, input_mask)
        forward_score = self._forward_alg(lstm_features)
        sentence_tags_score = self._get_sentence_tags_score(lstm_features, label_ids)
        return torch.mean(forward_score - sentence_tags_score)

    def forward(self, input_ids, segment_ids, input_mask):
        """
        模型预测
        """
        lstm_features = self._get_lstm_features(input_ids, segment_ids, input_mask)
        score, label_seq_ids = self._viterbi_decode(lstm_features)
        return score, label_seq_ids

class BertBiLstmCrfNer(nn.Module):
    def __init__(self, common_config, model_config):
        self.common_config = common_config
        self.model_config = model_config
        self.start_epoch = 0
        self.prev_acc_score = 0
        self.prev_f1_score = 0

    def get_model(self):
        """
        通过读取配置参数,加载 bert bilstm crf ner 模型
        """
        bert_model = BertModel.from_pretrained(self.common_config.bert_model_dir)
        model = BertBiLstmCrf(
            bert_model=bert_model,
            labels_idx_dict=self.common_config.labels_idx_dict,
            batch_size=self.common_config.batch_size,
            dropout=self.model_config.dropout,
            hidden_size=self.model_config.bert_hidden_size,
            lstm_hidden_size=self.model_config.lstm_hidden_size,
            lstm_layers=self.model_config.lstm_layers,
            device=self.common_config.device
        )
        checkpoint_path = os.path.join(self.common_config.checkpoint_dir, self.model_config.bert_bilstm_crf_model_name)
        if self.model_config.load_checkpoint and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            self.start_epoch = checkpoint["epoch"] + 1
            self.prev_acc_score = checkpoint["acc_score"]
            self.prev_f1_score = checkpoint["f1_score"]
            pretrained_params_dict = checkpoint["model_state"]
            net_state_dict = model.state_dict()
            useful_pretrained_params_dict = {k: v for k, v in pretrained_params_dict.items() if k in net_state_dict}
            net_state_dict.update(useful_pretrained_params_dict)
            model.load_state_dict(net_state_dict)
            print("Loaded pretrain bert bilstm crf ner model, previous acc: {}, previous f1_score : {}".format(self.prev_acc_score, self.prev_f1_score))
        return model