"""
test.py: BERT NER 模型的测试模块 
by: qliu
update date: 2021-12-17
"""
from .evaluate import evaluate
from ..ner_utils.data_loader import NerBatchDataloader
from .model import BertNer

def test(common_config, model_config):
    """
    测试 Bert NER 模型的效果
    """
    # 读取所有需要的参数
    device = common_config.device
    batch_size = common_config.batch_size

    # 加载训练好的模型
    model_config.update(load_checkpoint=True)
    bert_ner = BertNer(common_config=common_config, model_config=model_config)
    model = bert_ner.get_model()

    model.to(device)

    # 读取测试数据
    ner_batch_dataloader = NerBatchDataloader(common_config)
    test_dataloader = ner_batch_dataloader.get_test_dataloader()

    # 在测试数据上跑模型
    evaluate(model, test_dataloader, batch_size, -1, "test dataset", device)