"""
evaluate.py: BERT CRF & BERT BILSTM CRF 模型评估模块
by: qliu
update date: 2021-12-17
"""
import time, torch
from ..ner_utils.caculate import count_f1_score

def evaluate(model, dataloader, batch_size, epoch_th, dataset_name, device):
    print("start evaluating ...")
    model.eval()
    all_true_labels = []
    all_pred_labels = []
    total = 0
    correct = 0
    start_time = time.time()
    with torch.no_grad():
        for batch in dataloader:
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, predict_mask, label_ids = batch
            _, pred_label_seq_ids = model(input_ids=input_ids, segment_ids=segment_ids, input_mask=input_mask)
            pred_labels = torch.masked_select(pred_label_seq_ids, predict_mask)
            true_labels = torch.masked_select(label_ids, predict_mask)
            all_true_labels.extend(true_labels.tolist())
            all_pred_labels.extend(pred_labels.tolist())
            total += len(true_labels)
            correct += pred_labels.eq(true_labels).sum().item()

    test_acc = correct / total
    precision, recall, f1 = count_f1_score(all_true_labels, all_pred_labels)
    print("dataset's name is {} ... evaluate epoch : {}, precision: {}, recall: {}, f1: {}, costs : {}".format(dataset_name, epoch_th, precision, recall, f1, round(time.time() - start_time, 3)))
    return test_acc, f1
