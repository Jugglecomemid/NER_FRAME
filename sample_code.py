"""
sample_code.py: 完整流程的示例代码
说明: 若各个文件路径与 README.md 中一致,这部分为训练测试部署优化模型所需的全部代码,所以,请一定先看 README.md :)
by: qliu
update date: 2021-12-17
"""
## 更新配置参数
# 设置指定路径为工作路径
from ner.ner_utils.common_utils import set_work_dir
workspace_path = "/home/qliu/workspace/ner_test"
set_work_dir(workspace_path)

# 根据指定的工作路径,修改各类文件的相对路径
# 注意: 各个文件夹需要提前创建好,模型文件（bert_model_dir）& 实体缩写字典（entities_file）& 服务端口字典（server_ports_file）需要提前准备好
from ner.ner_utils.config import CommonConfig, ModelConfig
ccfg = CommonConfig()
mcfg = ModelConfig()

# 根据 daccono 的标注结果,生成训练测试开发数据 & 句子与标签对应的原始文件
from ner.ner_utils.processing import InitialDataTxtGenerator
init_data_generator = InitialDataTxtGenerator(ccfg)
init_data_generator.save_all_useful_data()

## 各类模型的训练,测试及部署
# BERT 
# 训练
from ner.bert.train import train
train(common_config=ccfg, model_config=mcfg)
# 测试
from ner.bert.test import test
test(common_config=ccfg, model_config=mcfg)
# 部署
# 更改  bert.py 中 workspace_path & server_ports_file 中端口号,然后 python bert.py 即可

# BERT CRF 模型
# 训练
from ner.bert_crf.train import train
train(common_config=ccfg, model_config=mcfg)
# 测试
from ner.bert_crf.test import test
test(common_config=ccfg, model_config=mcfg)
# 部署
# 更改 bert_crf.py 中 workspace_path & server_ports_file 中端口号,然后 python bert_crf.py 即可

# BERT BILSTM CRF 模型
# 训练
from ner.bert_bilstm_crf.train import train
train(common_config=ccfg, model_config=mcfg)
# 测试
from ner.bert_bilstm_crf.test import test
test(common_config=ccfg, model_config=mcfg)
# 部署
# 更改  bert_bilstm_crf.py 中 workspace_path & server_ports_file 中端口号,然后 python bert_bilstm_crf.py 即可


## 检查标记数据的正确性
# 各类模型部署完成后,可通过比较机器预测&人工标记的标签,检查人工标记的错误
# 修改错误标签后,再次进行模型训练
from ner.ner_utils.data_checker import DatasetChecker
from ner.ner_utils.common_utils import read_sents_labels_tuples
import os
dataset_checker = DatasetChecker(ccfg, "bert_bilstm_crf")
sents_labels_tuples = read_sents_labels_tuples(os.path.join(ccfg.dataset_dir, "label_tuples.txt"))
dataset_checker.check_dataset(sents_labels_tuples)


## 利用模型进行预测
from ner.ner_utils.post_processing import get_sent_entities
sent = "2020年4月15日，嗦粉佬宣布完成1000万人民币的天使轮融资，由新加坡优贝迪基金会领投，餐饮品牌价值发现平台吃货大陆跟投。"
tokens, labels = dataset_checker.get_sent_predict(sent)
# 获取 [实体字符串,实体类别] 的列表
entities_strs = get_sent_entities(sent=sent, tokens=tokens, labels=labels,return_idx=False)
# 获取 [实体索引,实体类别] 的列表
entities_idxes = get_sent_entities(sent=sent, tokens=tokens, labels=labels,return_idx=True)