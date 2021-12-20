"""
data_checker.py: 数据标记正确性检查器
by: qliu
update date: 2021-12-17
"""
from .common_utils import read_json, get_tokens_labels
import json, requests

class DatasetChecker(object):
    def __init__(self, config, model_name="bert"):
        """
        Args:
            config: config.py 中的 CommonConfig
            model_name: str, 要用的检查器模型,可输入项有 ["bert", "bert_crf", "bert_bilstm_crf"]
        """
        self.config = config
        self.model_name = model_name
        self.ports_dict = read_json(self.config.server_ports_file)

    def _get_url(self):
        if self.model_name == "bert":
            return "http://{}:{}/predict_by_bert".format(self.ports_dict["hosts"], self.ports_dict["bert"])
        if self.model_name == "bert_crf":
            return "http://{}:{}/predict_by_bert_crf".format(self.ports_dict["hosts"], self.ports_dict["bert_crf"])
        return "http://{}:{}/predict_by_bert_bilstm_crf".format(self.ports_dict["hosts"], self.ports_dict["bert_bilstm_crf"])

    def get_sent_predict(self, sent):
        url = self._get_url()
        data = json.dumps({"sent": sent})
        r = requests.post(url, data=data.encode("utf-8"))
        res = json.loads(r.text)
        if "tokens" not in res["response"]:
            tokens = []
            labels = []
        else:
            tokens = res["response"]["tokens"]
            labels = res["response"]["labels"]
        return tokens, labels

    def check_sent(self, sent_labels_tuple):
        sent, labels = sent_labels_tuple
        if len(sent.strip()) < 2:
            return 
        labels = labels.split()
        tokens, new_labels = get_tokens_labels(self.config.tokenizer, sent, labels)
        pred_labels = self.get_sent_predict(sent)[-1]
        pred_labels = pred_labels[1:-1]
        have_error = False
        for t, rl, pl in zip(tokens, new_labels, pred_labels):
            if pl in ["O", "X", "START", "END"]:
                continue
            if rl != pl:
                have_error = True
                break
        if have_error:
            print("-"*10, "error sent", "-"*10)
            print(sent)
            for t, rl, pl in zip(tokens, new_labels, pred_labels):
                if rl != pl:
                    print(t, rl, pl)
                else:
                    print(t)
        return 

    def check_dataset(self, sents_labels_tuples):
        for idx, sent_labels in enumerate(sents_labels_tuples):
            print("check {} sent ... \n".format(str(idx)))
            self.check_sent(sent_labels)
        print("done ...")