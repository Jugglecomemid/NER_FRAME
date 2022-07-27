"""
bert_crf.py: BERT CRF NER 模型预测服务部署模块
by: qliu
update date: 2021-12-17
"""
import os, torch, json
from flask import Flask, request
from ner.bert_crf.model import BertCrfNer
from ner.ner_utils.data_loader import SentNerData
from ner.ner_utils.config import CommonConfig, ModelConfig
from ner.ner_utils.common_utils import read_json, set_work_dir

# 必须设置
workspace_absolute_path = "/home/hesy/extract_words/extract_company_name/ner_business_scope"
app = Flask(__name__)
@app.route('/predict_by_bert_crf', methods=['POST'])
def predict_by_bert_crf():
    args = request.data.decode("utf-8")
    if isinstance(args, str):
        args = json.loads(args)
    
    res = {
        "response": {},
        "error_message": ""
    }

    sent = args.get("sent")

    if not isinstance(sent, str):
        error_msg = "sent's type must be str ... "
        res["error_message"] = error_msg
        return json.dumps(res)

    if len(sent) > 256:
        error_msg = "sent is longer than 400 ... "
        res["error_message"] = error_msg
        return json.dumps(res)

    input_ids, input_mask, segment_ids, _, _ = sent_dataloader._get_sent_input_data(sent)
    _, pred_label_seq_ids = model(input_ids=input_ids, segment_ids=segment_ids, input_mask=input_mask)
    pred_label_seq_ids = pred_label_seq_ids[0]
    tokens = bert_tokenizer.tokenize(sent)
    tokens = ["START"] + tokens + ["END"]
    labels = [idx_labels_dict[l.item()] for l in pred_label_seq_ids]
    res["response"] = {"sent": sent, "tokens": tokens, "labels": labels}
    return json.dumps(res)

if __name__ == '__main__':
    set_work_dir(workspace_absolute_path)
    torch.manual_seed(1)
    ccfg = CommonConfig()
    mcfg = ModelConfig()
    mcfg.update(load_checkpoint=True)
    bert_tokenizer = ccfg.tokenizer
    idx_labels_dict = ccfg.idx_labels_dict
    sent_dataloader = SentNerData(ccfg)

    print("bert crf ner server start ...")
    bert_crf = BertCrfNer(common_config=ccfg, model_config=mcfg)
    model = bert_crf.get_model()
    model.to(ccfg.device)
    model.eval()

    ports_dict = read_json(ccfg.server_ports_file)
    app.run('0.0.0.0', ports_dict["bert_crf"], True)