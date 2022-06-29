## NER
NER：最简单的 NER 全流程组件
* Author：qliu
* Updated：2021-12-20

<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->

<!-- code_chunk_output -->

- [NER](#ner)
- [特点](#特点)
- [安装说明](#安装说明)
- [文档结构](#文档结构)
- [主要功能](#主要功能)
  - [1. 所有参数配置，一步到位](#1-所有参数配置一步到位)
  - [2. 自定义实体字典，适用各种 NER 场景](#2-自定义实体字典适用各种-ner-场景)
  - [3. 标签数据一键转换](#3-标签数据一键转换)
  - [4. 模型训练 & 测试 & 部署，四行搞定](#4-模型训练-测试-部署四行搞定)
    - [4.1 基于 BERT 的 NER 算法](#41-基于-bert-的-ner-算法)
    - [4.2 基于 BERT + CRF 的 NER 算法](#42-基于-bert-crf-的-ner-算法)
    - [4.3 基于 BERT + BILSTM + CRF 的 NER 算法](#43-基于-bert-bilstm-crf-的-ner-算法)
  - [5. 自动化核查人工标签，进一步优化模型效果](#5-自动化核查人工标签进一步优化模型效果)
  - [6. 不同模型自由切换，哪个好用用哪个](#6-不同模型自由切换哪个好用用哪个)
  - [7. 训练数据自动生成，省时省力超厉害 ：）](#7-训练数据自动生成省时省力超厉害)
- [参考论文](#参考论文)

<!-- /code_chunk_output -->


## 特点
* 支持三种 NER 模型：
    * BERT
    * BERT + CRF
    * BERT + BILSTM + CRF
* 支持自定义实体类别。
* 支持 doccano 生成的标注结果到模型数据的一键转换。
* 支持不同模型一键训练，测试，部署，预测。
* 支持对人工标记数据的检查。

## 安装说明
* 代码兼容 py3
* 半自动安装：先下载 https://gitlab.mvalley.com/data-processing-infrastructure/ner ，解压后运行 `python setup.py sdist`，然后 cd dist/ ，最后 `pip install ner-0.0.1.tar.gz`
* 手动安装：将 ner 目录放置于当前目录或者 site-packages 目录
* 通过 `import ner` 来引用

## 文档结构
**标 * 为必须准备好的文件夹 / 文件，特别重要！！！**
```
project
│   README.md
│   sample_codes.py                                                                                                      完整 NER 流程的示例代码
│   bert.py                                                                                                                           部署 bert ner 服务
│   bert_crf.py                                                                                                                   部署 bert crf ner 服务
│   bert_bilstm_crf.py                                                                                                    部署 bert bilstm crf ner 服务
│   setup.py                                                                                                                        生成 ner 安装包
│
└─── data 
│       │
│       └─  model_data                                                                                               * 存放 训练、测试、开发、原始数据
│       │       |  dev.txt                                                                                                          开发数据，InitialDataTxtGenerator 生成，下同
│       │       |   label_tuples.txt                                                                                       原始句 & 标签句的映射
│       │       |   test.txt                                                                                                         测试数据
│       │       |   train.txt                                                                                                       训练数据
│       │       |   ...
│       │
│       └─  original_data                                                                                            * 
│       │       │ 
│       │       └ jsonl                                                                                                        *  doccano 结果文件的文件夹
│       │             │  v1.jsonl                                                                                           * 
│       │             │  ...
│       │             │  
│       │       │ entities.json                                                                                         * NER 实体 & 其缩写的映射字典
│       │       │ port.json                                                                                                *各类模型预测服务的端口 & hosts 配置字典
│       │       
│       └─ result                                                                                                              * 存放训练好的模型文件
│       │       |   bert.pt                                                                                                         训练生成的 bert ner 模型
│       │   ...
│   
└─── hfl                                                                                                                       * bert 模型文件
│       │
│       └─ chinese-bert-wwm-ext                                                                           * 预训练好的 chinese-bert 模型，下载地址: https://huggingface.co/bert-base-chinese
│       │       |  config                                                                                                        *  bert 模型配置参数
│       │       |  ...
│       │ 
└─── ner                                                                                                                         ner 框架代码
│       │
│       └─  bert                                                                                                                    bert 模型的训练测试评估模块
│       │       |  __init__.py 
│       │       |  evaluate.py                                                                                                 模型评估模块
│       │       |  model.py                                                                                                      bert ner 模型模块
│       │       |  test.py                                                                                                           模型测试模块
│       │       |  train.py                                                                                                         模型训练模块
│       │      
│       └─  bert_bilstm_crf                                                                                            bert bilstm crf ner 模型的训练测试评估模块
│       │       |  __init__.py 
│       │       |  model.py                                                                                                     bert bilstm crf ner 模型模块
│       │       |  test.py                                                                                                          模型测试模块
│       │       |  train.py                                                                                                        模型训练模块
│       │       
│       └─  bert_crf                                                                                                           bert crf ner 模型的训练测试评估模块
│       │       |  __init__.py 
│       │       |  evaluate.py                                                                                                模型评估模块
│       │       |  model.py                                                                                                    bert crf ner 模型模块
│       │       |  test.py                                                                                                         模型测试模块
│       │       |  train.py                                                                                                       模型训练模块
│       │    
│       └─  ner_utils                                                                                                         NER 框架的预处理、后处理、其他通用模块
│       │       |  __init__.py 
│       │       |  calculate.py                                                                                              NER 算法中所有计算模块
│       │       |  common_utils.py                                                                                   数据读写 & 工作路径设置模块
│       │       |  config.py                                                                                                    NER 框架所涉的所有参数的配置模块
│       │       |  data_checker.py                                                                                     数据标记正确性检查器
│       │       |  data_loader.py                                                                                        模型数据生成器
│       │       |  post_processing.py                                                                               数据后处理模块
│       │       |  processing.py                                                                                           数据预处理模块
│       │     
│       │__init__.py
│       
```

## 主要功能
**请先按顺序一步步来！请先按顺序一步步来！请先按顺序一步步来！**
###  1. 所有参数配置，一步到位
------------------
* 接口说明：**(copy 示例代码，确定全局路径无误即可)**
  * `set_work_dir` 指定路径作为全局的工作路径，之后所有路径都是该路径的相对路径。
  * `CommonConfig` 通用参数模块：不管使用哪个模型，都通用的配置参数；可分五类：各类文件路径、数据大小 & 形式设置参数、模型训练环境设置、自定义实体字典、分字器。
  * `ModelConfig`模型的训练 & 优化相关的参数；可分四类：模型结果文件名、模型结构参数、模型训练次数 & 是否加载先前训练结果、优化器参数。

* 代码示例：
    ```python
    from ner.ner_utils.common_utils import set_work_dir
    # 设置指定路径为工作路径
    workspace_path = "/home/xxx/workspace/ner_test"
    set_work_dir(workspace_path)

    # 根据指定的工作路径，修改各类文件的相对路径
    # 注意：各个文件夹 & 必备文件，需要根据「文件结构」提前创建好
    from ner.ner_utils.config import CommonConfig, ModelConfig
    ccfg = CommonConfig()
    mcfg = ModelConfig()
    ```

###  2. 自定义实体字典，适用各种 NER 场景
----------------
* 方法说明：**（确定实体字典无误，然后 copy 示例代码即可）**
  * 开发者可以根据需要，指定不同的实体字典，然后将其存为 `entities.json`，存放在`ccfg.entities_file`路径下即可。
  * 用法：ccfg.update(entities_file=entities_file_name) # entities_file_name 为实体字典的相对路径。
  * 实体字典格式和 `data/original_data/entities.json` 一样，以 `实体名` 为 key，`实体缩写` 为 value。诸如「O、X、START、END」不需要添加至实体字典。

* 示例：
    ```json
    {
        "关联方": "REL",
        "属性名词": "ATTR",
        "发生时间": "OTIME",
        "金额": "VALUE",
        "交易类型": "TYPE",
        "披露时间": "ETIME",
        "融资方标签": "LABEL"
    }
    ```
* 代码示例：
    ```python
    ccfg.update(
        entities_file="data/original_data/entities.json"
    )
    >>> print("实体字典: \n", ccfg.entities_abbr_dict)
    实体字典: 
    {'关联方': 'REL', '属性名词': 'ATTR', '发生时间': 'OTIME', '金额': 'VALUE', '交易类型': 'TYPE', '披露时间': 'ETIME', '融资方标签': 'LABEL'}

    >>> print("模型标签索引: \n", ccfg.labels_idx_dict)
    模型标签索引: 
    {'X': 0, 'START': 1, 'END': 2, 'O': 3, 'B_REL': 4, 'I_REL': 5, 'B_ATTR': 6, 'I_ATTR': 7, 'B_OTIME': 8, 'I_OTIME': 9, 'B_VALUE': 10, 'I_VALUE': 11, 'B_TYPE': 12, 'I_TYPE': 13, 'B_ETIME': 14, 'I_ETIME': 15, 'B_LABEL': 16, 'I_LABEL': 17}
    ```

###  3. 标签数据一键转换
-------------
* 方法说明：将 doccano 生成的标签结果，一键转换成可喂入模型的开发数据。**(copy 示例代码即可)**
* 代码示例：
    ```python
    from ner.ner_utils.processing import InitialDataTxtGenerator
    ccfg.update(
        dataset_dir="data/model_data", # train.txt 的保存路径
        jsonl_files_dir="data/original_data/jsonl",# doccano 结果的保存路径
        train_data_split=(0.7, 0.15, 0.15) # (train, test, dev) 的分割比例
    )
    init_data_generator = InitialDataTxtGenerator(ccfg)
    init_data_generator.save_all_useful_data()
    ```
* 输出:
    `data/model_data`文件夹下会多4个文件,分别是：
  -  dev.txt: 开发数据
     - 格式：一个字占一行；每一行分两个部分：字、实体标签，用空格隔开，句子与句子间用空行分隔。
     - 示例：
        ```
            9 B_ETIME
            月 I_ETIME
            26 I_ETIME
            日 I_ETIME
        ```
  - label_tuples.txt: 句子&标签映射的原始数据
    - 格式：一个句子占一行，下接句子中各字对应的标签句，句子与句子间用空行分隔。
    - 示例：
    ```
      创世纪(300083.SZ)子公司完成引入国家制造业基金股权投资并收到5亿元投资款。
      B_REL I_REL I_REL I_REL I_REL I_REL I_REL I_REL I_REL I_REL I_REL I_REL I_REL I_REL I_REL I_REL I_REL O O O O B_REL I_REL I_REL I_REL I_REL I_REL I_REL B_TYPE I_TYPE I_TYPE I_TYPE O O O B_VALUE I_VALUE I_VALUE B_ATTR I_ATTR I_ATTR O
    ```
  - test.txt: 测试数据
    - 格式：同`dev.txt`
  - train.txt: 训练数据
    - 格式：同`dev.txt`

###  4. 模型训练 & 测试 & 部署，四行搞定
-------------
#### 4.1 基于 BERT 的 NER 算法
`from ner.bert.train import train`
* 接口说明： **(可略，直接用 BERT + BILSTM + CRF)**
  * train(common_config, model_config)
    * common_config: 通用参数模块均可不做更改
    * model_config: 仅对「优化器参数」模块的参数做出微调即可（亦可保留默认值），即`lr`、`weight_decay_finetune`、`gradient_accumulation_steps`、`warmup_proportion`，若需保留不同参数下的不同模型结果，还需更改`bert_model_name`
  * test(common_config, model_config)
    * common_config: 通用参数模块均可不做更改
    * model_config: 当需要比较不同参数下的模型结果时），通过`model_config.update(bert_model_name=bert_v1.pt)`更改模型名即可
* 模型的训练 & 测试：
    ```python
    from ner.bert.train import train
    # 设置模型学习参数(可不更改)
    mcfg.update(
        lr=5e-5,
        weight_decay_finetune = 1e-5
    )

    # 训练
    train(common_config=ccfg, model_config=mcfg)

    # 测试
    from ner.bert.test import test
    test(common_config=ccfg, model_config=mcfg)
    ```
* 模型部署 **（可略，直接用 BERT + BILSTM + CRF）**
  * 更改参数：更改代码中的`workspace_path`为指定的工作路径，然后在`data/original_data/ports.json`中指定`hosts` & `bert`端口即可
  * 运行代码：https://gitlab.mvalley.com/data-processing-infrastructure/ner/-/blob/master/bert.py

* 实体识别 **（重要，copy 示例代码即可）**
  * 用法：
      ```python
      import requests, json

      with open("./data/original_data/ports.json",'r') as load_f:
          ports_dict = json.load(load_f)

      sent = "2020年4月15日，嗦粉佬宣布完成1000万人民币的天使轮融资，由新加坡优贝迪基金会领投，餐饮品牌价值发现平台吃货大陆跟投。"
      data = {"sent": sent}
      data = json.dumps({"sent": sent})
      url = "http://{}:{}/predict_by_bert".format(ports_dict["hosts"], ports_dict["bert"])  
      # bert_bilstm_crf 的 url 为:  "http://{}:{}/predict_by_bert_bilstm_crf".format(self.ports_dict["hosts"], self.ports_dict["bert_bilstm_crf"])
      r = requests.post(url, data=data.encode("utf-8"))
      res = json.loads(r.text)
      print(res)
      ```
  * 输出：
      ```json
      {
          "response": {
              "sent": "2020年4月15日，嗦粉佬宣布完成1000万人民币的天使轮融资，由新加坡优贝迪基金会领投，餐饮品牌价值发现平台吃货大陆跟投。",
              "tokens": [
                  "START", "2020", "年", "4", "月", "15", "日", "，", "嗦", "粉", "佬", "宣", "布", "完", "成", "1000", "万", "人", "民", "币", "的", "天", "使", "轮", "融", "资", "，", "由", "新", "加", "坡", "优", "贝", "迪", "基", "金", "会", "领", "投", "，", "餐", "饮", "品", "牌", "价", "值", "发", "现", "平", "台", "吃", "货", "大", "陆", "跟", "投", "。", "END"
              ],
              "labels": [
                  "START", "B_ETIME", "I_ETIME", "I_ETIME", "I_ETIME", "I_ETIME", "I_ETIME", "O", "B_REL", "I_REL", "I_REL", "O", "O", "O", "O", "B_VALUE", "I_VALUE", "I_VALUE", "I_VALUE", "I_VALUE", "O", "B_TYPE", "I_TYPE", "I_TYPE", "I_TYPE", "I_TYPE", "O", "O", "B_REL", "I_REL", "I_REL", "I_REL", "I_REL", "I_REL", "I_REL", "I_REL", "I_REL", "O", "O", "O", "B_ATTR", "I_ATTR", "I_ATTR", "I_ATTR", "I_ATTR", "I_ATTR", "I_ATTR", "I_ATTR", "I_ATTR", "I_ATTR", "B_REL", "I_REL", "I_REL", "I_REL", "O", "O", "O", "END"
              ]
          },
          "error_message": ""
      }
      ```
#### 4.2 基于 BERT + CRF 的 NER 算法
`from ner.bert_crf.train import train`
* 接口说明：同上  **(可略，直接用 BERT + BILSTM + CRF)**
* 模型的训练 & 测试
    ```python
    # 训练
    from ner.bert_crf.train import train
    train(common_config=ccfg, model_config=mcfg)

    # 测试
    from ner.bert_crf.test import test
    test(common_config=ccfg, model_config=mcfg)
    ```
* 模型部署： **(可略，直接用 BERT + BILSTM + CRF)**
  * 更改参数：更改代码中的`workspace_path`为指定的工作路径，然后在`data/original_data/port.json`中指定`hosts` & `bert`端口即可
  * 运行代码：https://gitlab.mvalley.com/data-processing-infrastructure/ner/-/blob/master/bert_crf.py
  * 访问链接：`http://{}:{}/predict_by_bert_crf".format(ports_dict["hosts"], ports_dict["bert_crf"]) `

#### 4.3 基于 BERT + BILSTM + CRF 的 NER 算法
`from ner.bert_bilstm_crf.train import train`
* 接口说明：同上  **(copy 示例代码即可)**
* 模型的训练 & 测试
  ```python
    # 训练
    from ner.bert_bilstm_crf.train import train
    train(common_config=ccfg, model_config=mcfg)

    # 测试
    from ner.bert_bilstm_crf.test import test
    test(common_config=ccfg, model_config=mcfg)
    ```
* 模型部署：**(更改 port 配置后，终端 python bert_bilstm_crf.py 即可)**
    * 更改参数：更改代码中的`workspace_path`为指定的工作路径，然后在`data/original_data/port.json`中指定`hosts` & `bert`端口即可
    * 运行代码：https://gitlab.mvalley.com/data-processing-infrastructure/ner/-/blob/master/bert_bilstm_crf.py
    * 访问链接：`"http://{}:{}/predict_by_bert_bilstm_crf".format(ports_dict["hosts"], ports_dict["bert_bilstm_crf"]) `

###  5. 自动化核查人工标签，进一步优化模型效果
-------------
`from ner.ner_utils.data_checker import DatasetChecker`
* 接口说明：**(copy 示例代码即可)**
    * DatasetChecker(config, model_name)
    * config: 通用参数模块，无需更改
    * model_name: 指定的检查模型，可选`bert` & `bert_crf` & `bert_bilstm_crf`
* 代码示例：
```python
# encoding=utf-8
from ner.ner_utils.data_checker import DatasetChecker
from ner.ner_utils.common_utils import read_sents_labels_tuples
import os

dataset_checker = DatasetChecker(ccfg, "bert_bilstm_crf")
sents_labels_tuples = read_sents_labels_tuples(os.path.join(ccfg.dataset_dir, "label_tuples.txt"))
dataset_checker.check_dataset(sents_labels_tuples)
```
* 输出：
  含义：逐个检查标记人工标签&模型预测的不同之处，并将差异句索引 & 差异句 & 差异标签显示出来。
  示例：如下方的 `check 7 sent` 中的 7 为差异句索引，`而圆心科技自2014年成立以来，也先后斩获6轮融资，上一轮融资距离此次融资不过半年时间。`为差异句，`自`为差异字，`B_OTIME`为真实标签`O`为预测标签，之后可再根据索引对人工标签进行更改，从而进一步优化模型。
    ```
    ...
    check 7 sent ... 

    ---------- error sent ----------
    而圆心科技自2014年成立以来，也先后斩获6轮融资，上一轮融资距离此次融资不过半年时间。
    而
    圆
    心
    科
    技
    自 B_OTIME O
    2014 I_OTIME B_OTIME
    年
    ...
    ```
###  6. 不同模型自由切换，哪个好用用哪个
-------------
* 代码示例：**(copy 示例代码即可)**
  * 获取 [实体字符串,实体类别] 的列表
    ```python
    from ner.ner_utils.data_checker import DatasetChecker
    from ner.ner_utils.post_processing import get_sent_entities

    dataset_checker = DatasetChecker(ccfg, "bert_bilstm_crf")
    tokens, labels = dataset_checker.get_sent_predict(sent)
    
    entities_strs = get_sent_entities(sent=sent, tokens=tokens, labels=labels,return_idx=False)
    entities_strs
    ```
  * 输出
    ```
    [['2020年4月15日', '披露时间'],
    ['嗦粉佬', '关联方'],
    ['1000万人民币', '金额'],
    ['天使轮融资', '交易类型'],
    ['新加坡优贝迪基金会', '关联方'],
    ['餐饮品牌价值发现平台', '属性名词'],
    ['吃货大陆', '关联方']]
    ```

  * 获取 [实体索引,实体类别] 的列表
    ```python
    entities_idxes = get_sent_entities(sent=sent, tokens=tokens, labels=labels,return_idx=True)
    ```
  * 输出
    ```
    [[0, 10, '披露时间'],
    [11, 14, '关联方'],
    [18, 26, '金额'],
    [27, 32, '交易类型'],
    [34, 43, '关联方'],
    [46, 56, '属性名词'],
    [56, 60, '关联方']]
    ```

###  7. 训练数据自动生成，省时省力超厉害 ：）
-------------
* 代码示例：**(copy 示例代码即可)**
  * 模型回标，自动生成训练数据
    ```python
    from ner.ner_utils.post_processing import generate_and_save_new_training_data

    sents = ["一网打尽Google、Amazon、Microsoft、Facebook在2018年KDD上的论文：神经网络、大规模计算是热点。"]
    save_path = "../training_data.jsonl"  # jsonl 文件别错了
    generate_and_save_new_training_data(sents, save_path, ccfg, model_name="bert_bilstm_crf")
    ```
  * 输出：
  "../training_data.jsonl" 路径下会多一个文件，该文件可直接导入 doccano 进行结果核查，文件形式如下：
    ```
    {"id": "0", "data": "一网打尽Google、Amazon、Microsoft、Facebook在2018年KDD上的论文：神经网络、大规模计算是热点。", "label": [[4, 10, "有效词"], [11, 17, "有效词"], [18, 27, "有效词"], [28, 36, "有效词"], [42, 44, "有效词"], [50, 54, "有效词"], [55, 60, "有效词"]]}
    ```


## 参考论文
* 《BERT: Pre-training of Deep Bidirectional Transformers for
Language Understanding》
* 《 Conditional Random Fields: Probabilistic Models for Segmenting
and Labeling Sequence Data》
* 《 Conditional Random Fields: An Introduction》
* 《 Conditional Random Fields: Probabilistic Models for Segmenting
and Labeling Sequence Data》
* 《 Transition-Based Dependency Parsing with Stack Long Short-Term Memory》
* 《 New Research on Transfer Learning Model of Named Entity Recognition》
* 《 Neural Architectures for Named Entity Recognition》
* 《 Improving Clinical Named Entity Recognition with Global Neural Attention》
* 《 End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF》
* 《 A Survey on Deep Learning for Named Entity Recognition》
