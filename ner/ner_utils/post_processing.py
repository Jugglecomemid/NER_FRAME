"""
post_processing.py: 后处理模块，将 ner 预测结果转换成实体序列
by: qliu
date: 2021-12-16
"""

import re
from .common_utils import (
    get_entity_type,
    get_tokens_length,
    get_tokens_labels_idxes,
    read_json
)
from .config import CommonConfig
from .data_checker import DatasetChecker
import jsonlines
from tqdm import tqdm

cfg = CommonConfig()
abbr_entities_dict = cfg.abbr_entities_dict

def _get_entity_idx_and_type(token_start_idx, tokens, labels, regardless_bi=False):
    """
    获取有效实体的索引及其类型
    """
    entity_type = re.search("[^_]+$", labels[token_start_idx]).group(0)
    token_end_idx = token_start_idx
    for _, l in enumerate(labels[token_start_idx + 1: ]):
        if re.search("[^_]+$", l).group(0) != entity_type: 
            break
        if not regardless_bi and re.search("^[^_]", l).group(0) != "I":
            break
        token_end_idx += 1
    return entity_type, (token_start_idx, token_end_idx)

def get_sent_entities(sent, tokens, labels, return_idx=True):
    """
    还原实体及其索引关系
    """
    tokens_real_indexes, tokens_real_labels = get_tokens_labels_idxes(sent, tokens[1:-1], labels[1:-1])
    real_tokens = [sent[idx[0]: idx[1]] for idx in tokens_real_indexes]
    labels_list = []
    last_check_tokens_idx = -1
    for i, (_, l) in enumerate(zip(real_tokens, tokens_real_labels)):
        if i < last_check_tokens_idx:
            continue
        if not re.search("^B", l):
            continue
        if re.search("[^_]+$", l).group(0) in ["ATTR", "LABEL", "VALUE"]:
            entity_type, (token_start_idx, token_end_idx) = _get_entity_idx_and_type(i, real_tokens, tokens_real_labels, True)
        else:
            entity_type, (token_start_idx, token_end_idx) = _get_entity_idx_and_type(i, real_tokens, tokens_real_labels, False)
        last_check_tokens_idx = token_end_idx
        if entity_type == "O":
            continue
        start = len("".join(real_tokens[: token_start_idx]))
        end = start + len( "".join(real_tokens[token_start_idx: token_end_idx+1]))
        labels_list.append([start, end, abbr_entities_dict[entity_type]])
    if return_idx:
        return labels_list
    return [[sent[ll[0]: ll[1]].strip(), ll[-1]] for ll in labels_list]

def generate_and_save_new_training_data(sents, save_path, ccfg, model_name="bert_bilstm_crf"):
    """
    利用模型对数据集进行预测，返回可导入 doccano 的 jsonl 文件
    Args:
        sents: list, [str1, str2, ...]
        save_path: str, 文件保存路径
        ccfg: CommonConfig(), 通用配置
        model_name: str, ["bert", "bert_crf", "bert_bilstm_crf"]，三选一，默认 bert_bilstm_crf， 模型已部署好的前提！
    """
    dataset_checker = DatasetChecker(ccfg, model_name)
    with jsonlines.open(save_path, "w") as fp:
        with tqdm(total=len(sents)) as pbr:
            for i, s in enumerate(sents):
                if len(s) > 250:
                    continue
                tokens, labels = dataset_checker.get_sent_predict(s)
                entities_idxes = get_sent_entities(sent=s, tokens=tokens, labels=labels,return_idx=True)
                fp.write({"id": str(i), "data": s, "label": entities_idxes})
                pbr.update(1)
    print("done ...")
