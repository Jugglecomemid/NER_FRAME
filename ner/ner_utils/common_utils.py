"""
common_utils.py: 数据读写 & 工作路径设置, token & sent & labels 间的转换模块
by: qliu
update date:  2021-12-16
"""
import json, os, re

def write_json(json_file, file_dir_path, file_name):
    if not os.path.exists(file_dir_path):
        os.mkdir(file_dir_path)
    with open(os.path.join(file_dir_path, file_name), "w+", encoding="utf-8") as fp:
        json.dump(json_file, fp, ensure_ascii=False)
    print("done ...")

def read_json(file_path):
    with open(file_path, "r+") as fp:
        json_file = json.load(fp)
    return json_file

def read_sents_labels_tuples(file_path):
    """
    读取 sent & labels 元组文件
    """
    with open(file_path, "r+") as fp:
        entries = fp.read().split("\n\n")

    sents_labels = []
    for entry in entries:
        if len(entry.strip()) < 2:
            continue
        sent, labels = entry.split("\n")
        sents_labels += [(sent, labels)]
    return sents_labels

def get_complete_path(base_path):
    """
    根据文件当前路径,补全完整路径
    """
    curr_path = os.getcwd()
    return os.path.join(curr_path, base_path)

def set_work_dir(relative_path):
    """
    设定工作路径
    """
    if os.path.exists(os.path.join(os.getenv("HOME"), relative_path)):
        os.chdir(os.path.join(os.getenv("HOME"), relative_path))
    else:
        raise Exception('Set work path error!')

def get_tokens_length(token):
    if token == '[UNK]':
        return 1
    return len(re.sub("##", "", token))

def get_entity_type(label):
    entity_type = re.search("[^_]+$", label)
    if not entity_type:
        return "O"
    return entity_type.group(0)

def get_tokens_idxes(sent, tokens):
    """
    获取各 token 的实际索引
    """
    blank_idx = [i for i, w in enumerate(sent) if w in [" ", ""] or re.search("\s", w)]
    have_blank = False
    if len(blank_idx) > 0:
        have_blank = True
    start_idx = 0
    tokens_indexes= []
    for token in tokens:
        token_length = get_tokens_length(token)
        if have_blank:
            extend_idx = len(set(blank_idx) & set(range(start_idx, start_idx + token_length + 1)))
        else:
            extend_idx = 0
        end_idx = start_idx + token_length + extend_idx
        idx = (start_idx, end_idx)
        tokens_indexes.append(idx)
        start_idx = end_idx
    return tokens_indexes

def get_tokens_labels(tokenizer, sent, labels):
    """
    获取 tokens & labels 的映射元组
    """
    sent_tokens = tokenizer.tokenize(sent)
    tokens_indexes = get_tokens_idxes(sent, sent_tokens)
    new_labels = []
    for _, idx in zip(sent_tokens, tokens_indexes):
        new_l = labels[idx[0]: idx[1]][0]
        new_labels.append(new_l)
    return sent_tokens, new_labels

def build_dataset(tokens_tag_txt_path):
    """
    将以 token & tag 为行的 txt 文件转换成以句为单位的 (tokens, tags) 元组的数据集.
    Return:
        - dataset: list: [(["我", "爱", "中", "国", ...], ["O", "O", "O", "O"]), ...]
    """
    with open(tokens_tag_txt_path, "r+") as fp:
        tokens, tags, dataset = [], [], []
        for i, line in enumerate(fp):
            if i == 0:
                continue
            line = re.sub("\n", "", line)
            if len(line) > 1:
                token, t = line.split(" ")
                tokens.append(token)
                tags.append(t)
            else:
                if len(tokens) == 0:
                    continue
                tokens_tags = (tokens, tags)
                dataset.append(tokens_tags)
                tokens, tags = [], []
    return dataset

def get_tokens_labels_idxes(sent, tokens, labels):
    """
    获取各 token 的实际索引和实际 labels
    """
    new_labels = []
    blank_idx = [i for i, w in enumerate(sent) if w in [" ", ""] or re.search("\s", w)]
    have_blank = False
    if len(blank_idx) > 0:
        have_blank = True
    start_idx = 0
    tokens_indexes= []
    pre_entity_type = "O"
    for i, (token, label) in enumerate(zip(tokens, labels)):
        # 补全细粒度 token 的 label
        if re.search("##", token):
            if pre_entity_type == "O":
                new_labels.append(pre_entity_type)
            else:
                new_labels.append("I_{}".format(pre_entity_type))
        else:
            new_labels.append(label)
            pre_entity_type = get_entity_type(label)
        # 求 token 实际索引
        token_length = get_tokens_length(token)
        if have_blank:
            extend_idx = len(set(blank_idx) & set(range(start_idx, start_idx + token_length + 1)))
        else:
            extend_idx = 0
        end_idx = start_idx + token_length + extend_idx
        idx = (start_idx, end_idx)
        tokens_indexes.append(idx)
        start_idx = end_idx
    return tokens_indexes, new_labels