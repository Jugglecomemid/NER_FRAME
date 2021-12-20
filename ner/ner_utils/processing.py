"""
processing.py: 数据预处理模块
将在 daccano 上标记的样本数据，转换成 token & label 对应的训练，测试，开发数据，并将结果保存至 config.dataset_dir 中
by: qliu
date: 2021-12-16
"""
import jsonlines, re, os
from .common_utils import get_tokens_idxes, get_tokens_labels

class InitialDataTxtGenerator(object):
    """
    初始训练测试开发数据生成器，
    将 daccano 生成的标签数据，转换为初始数据
    """
    def __init__(self, config):
        """
        Args:
            config: config.py 中的 CommonConfig
        """
        self.config = config
        self.labels_tuples = self._get_labels_tuples()

    def _get_labels_abbr_sent(self, sent, labels_idx_tuples):
        """
        将中文标签索引转换成标签缩写映射
        Args:
            sent: str, 原句子
            labels_idx_tuples: 标签索引&标签的三元组列表

        input e.g.:
            sent:  '本轮融资由The Chernin Group和Elysian Park领投，Temasek、JAZZ Venture Partners和eisai跟投。'
            labels_idx_tuples: [[5, 22, '关联方'], [23, 35, '关联方'], [38, 45, '关联方'], [46, 67, '关联方'], [68, 73, '关联方']]
        output:
        ("本轮融资由The Chernin Group和Elysian Park领投，Temasek、JAZZ Venture Partners和eisai跟投。",
        "O O O O O B_REL I_REL I_REL I_REL I_REL I_REL I_REL I_REL I_REL I_REL I_REL I_REL I_REL I_REL I_REL I_REL I_REL O B_REL I_REL I_REL I_REL I_REL I_REL I_REL I_REL I_REL I_REL I_REL I_REL O O O B_REL I_REL I_REL I_REL I_REL I_REL I_REL O B_REL I_REL I_REL I_REL I_REL I_REL I_REL I_REL I_REL I_REL I_REL I_REL I_REL I_REL I_REL I_REL I_REL I_REL I_REL I_REL I_REL O B_REL I_REL I_REL I_REL I_REL O O O")
        """
        labels = ["O" for _ in range(len(sent))]
        for lt in labels_idx_tuples:
            label_n = self.config.entities_abbr_dict.get(lt[-1])
            if not label_n:
                continue
            labels[lt[0]] = "B_{}".format(label_n)
            for i in range(lt[0]+1, lt[1]):
                labels[i] = "I_{}".format(label_n)
        return (sent, " ".join(labels))

    def _convert_jsonl_to_labels_tuples(self, jsonl_file_path):
        """
        从 jsonl 文件中读取数据,并将其转换成句子与标签映射的元组（daccano 输出结果）
        Args:
            jsonl_file_path: jsonl 文件路径
        """
        labels_tuples = []
        with open(jsonl_file_path, "r+", encoding="utf8") as fp:
            for item in jsonlines.Reader(fp):
                labels_tuple = self._get_labels_abbr_sent(item["data"], item["label"])
                labels_tuples.append(labels_tuple)
        return labels_tuples

    def _get_labels_tuples(self):
        jsonl_files = [f for f in os.listdir(self.config.jsonl_files_dir) if re.search("\.jsonl$", f)]
        if len(jsonl_files) == 0:
            print("none jsonl files in {} dir ...".format(self.config.jsonl_files_dir))
            return []
        labels_tuples = []
        for f in jsonl_files:
            labels_tuples += self._convert_jsonl_to_labels_tuples(os.path.join(self.config.jsonl_files_dir, f))
        return labels_tuples

    def _get_tokens_labels(self, sent, labels):
        """
        获取 tokens & labels 的映射元组
        """
        return get_tokens_labels(self.config.tokenizer, sent, labels)

    def _save_label_tuples(self):
        """
        保存标签训练数据
        """
        with open(os.path.join(self.config.dataset_dir, "label_tuples.txt"), "w+") as fp:
            for lt in self.labels_tuples:
                fp.write(lt[0] + "\n" + lt[1] + "\n\n")
        print("done ...")

    def _write_features_txt(self, sent_labels_tuples, txt_path):
        """
        将 labels_tuples 转换成可 token & label 对应的 txt 文件
        """
        with open(txt_path, "w+") as fp:
            for dt in sent_labels_tuples:
                if len(dt) < 2:
                    continue
                sent, ner_labels = dt
                if len(sent.strip()) < 2:
                    continue
                ner_labels = ner_labels.split()
                tokens, new_ner_labels = self._get_tokens_labels(sent, ner_labels)
                for token, ner_label in zip(tokens, new_ner_labels):
                    fp.write(token + " " + ner_label + "\n")
                fp.write("\n")
        return 

    def _generate_train_test_txt(self):
        """
        生成训练测试开发数据
        """
        split_tuples = self.config.train_data_split
        a, b = round(len(self.labels_tuples)*split_tuples[0]), round(len(self.labels_tuples)*split_tuples[1])
        train_data, test_data, dev_data = self.labels_tuples[: a], self.labels_tuples[a: a+b], self.labels_tuples[a+b: ]
        self._write_features_txt(train_data, os.path.join(self.config.dataset_dir, "train.txt"))
        print("train data write done ...")
        self._write_features_txt(test_data, os.path.join(self.config.dataset_dir, "test.txt"))
        print("test data write done ...")
        self._write_features_txt(dev_data, os.path.join(self.config.dataset_dir, "dev.txt"))
        print("dev data write done ...")

    def save_all_useful_data(self):
        """
        保存所有有效的文件
        """
        self._save_label_tuples()
        self._generate_train_test_txt()