"""
test.py: BERT CRF NER 模型测试模块
by: qliu
update date: 2021-12-17
"""
from .evaluate import evaluate
from ..ner_utils.data_loader import NerBatchDataloader
from .model import BertCrfNer

def test(common_config, model_config):
    """
    测试 Bert CRF NER 模型
    Args:
        common_config: 通用参数，详见 ner_utils/config: CommonConfig
        model_config: 模型参数, 详见 ner_utils/config: ModelConfig
    """
    # 加载基础模型及模型评估结果
    print("Start testing bert crf ner model ...")
    model_config.update(load_checkpoint=True)
    bert_crf = BertCrfNer(common_config=common_config, model_config=model_config)
    model = bert_crf.get_model()

    # 读取测试数据
    ner_batch_dataloader = NerBatchDataloader(common_config)
    test_dataloader = ner_batch_dataloader.get_test_dataloader()

    model.to(common_config.device)

    evaluate(model, test_dataloader, common_config.batch_size, -1, "test dataset", common_config.device)