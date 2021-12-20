"""
train.py: BERT BILSTM CRF NER 模型训练模块
by: qliu
update date: 2021-12-17
"""
import os, time, torch
import torch.optim as optim
from ..ner_utils.calculate import warmup_linear
from ..bert_crf.evaluate import evaluate
from ..ner_utils.data_loader import NerBatchDataloader
from .model import BertBiLstmCrfNer

def train(common_config, model_config):
    """
    训练 Bert BISLTM CRF NER 模型
    Args:
        common_config: 通用参数，详见 ner_utils/config: CommonConfig
        model_config: 模型参数, 详见 ner_utils/config: ModelConfig
    """
    # 加载基础模型及模型评估结果
    print("Start training bert bilstm crf ner model ...")
    bert_bilstm_crf = BertBiLstmCrfNer(common_config=common_config, model_config=model_config)
    model = bert_bilstm_crf.get_model()
    start_epoch = bert_bilstm_crf.start_epoch
    prev_acc_score = bert_bilstm_crf.prev_acc_score
    prev_f1_score = bert_bilstm_crf.prev_f1_score
    print("model loading completed ...")
    print("model epoch: {}, previous acc: {}, previous f1_score : {}".format(start_epoch, prev_acc_score, prev_f1_score))

    lr = model_config.lr

    # 将模型转移到 device 上
    ts = time.time()
    model.to(common_config.device)
    print("move model to {} costs {} ...".format(common_config.device, time.time()-ts))

    # 优化器
    named_params = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    new_param = ['transitions', 'full_connected_layer.weight', 'full_connected_layer.bias']
    optimizer_grouped_parameters = [
    {"params": [p for n, p in named_params if not any(nd in n for nd in no_decay) and not any(nd in n for nd in new_param)], "weight_decay": model_config.weight_decay_finetune},
    {"params": [p for n, p in named_params if any(nd in n for nd in no_decay) and not any(nd in n for nd in new_param)], "weight_decay": 0.0},
    {"params": [p for n, p in named_params if n in ('transitions', 'full_connected_layer.weight')], "lr": model_config.crf_fc_lr, "weight_decay": model_config.crf_fc_weight_decay},
    {"params": [p for n, p in named_params if n == "full_connected_layer.bias"], "lr": model_config.crf_fc_lr, "weight_decay": 0.0},
    ]
    optimizer = optim.Adam(optimizer_grouped_parameters, lr=lr)

    # 加载训练数据和评估数据
    ner_batch_dataloader = NerBatchDataloader(common_config)
    train_dataloader = ner_batch_dataloader.get_train_dataloader()
    dev_dataloader = ner_batch_dataloader.get_dev_dataloader()
    total_batch_num = len(train_dataloader)
    print("training data reading completed ... ")

    # 开始训练模型
    total_train_steps = int(total_batch_num  / model_config.gradient_accumulation_steps * model_config.total_train_epochs)
    global_step_th = int(total_batch_num / model_config.gradient_accumulation_steps * start_epoch)
    for epoch in range(start_epoch, model_config.total_train_epochs):
        tr_loss = 0
        start_time = time.time()
        model.train()
        optimizer.zero_grad()
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(common_config.device) for t in batch)
            input_ids, input_mask, segment_ids, _, label_ids = batch

            loss = model.negate_log_likelihood(input_ids, segment_ids, input_mask, label_ids)

            if model_config.gradient_accumulation_steps > 1:
                loss = loss / model_config.gradient_accumulation_steps

            loss.backward()

            tr_loss += loss.item()

            if (step + 1) % model_config.gradient_accumulation_steps == 0:
                # 调整 lr  (详见 bert 原始论文)
                new_lr = lr * warmup_linear(global_step_th / total_train_steps, model_config.warmup_proportion)
                for params in optimizer.param_groups:
                    params["lr"] = new_lr
                optimizer.step()
                optimizer.zero_grad()
                global_step_th += 1

            print("epoch: {}, step: {}, total_batch_num: {}, loss is {} ...".format(epoch, step, total_batch_num, loss.item()))

        print("-"*50)
        print("epoch {} completed, mean loss is {}, costs: {} ...".format(epoch, round(tr_loss / total_batch_num, 4), round((time.time() - start_time) / 60, 3)))
        eval_acc, eval_f1 = evaluate(model, dev_dataloader, common_config.batch_size, epoch, "dev dataset", common_config.device)

        # 保存最佳模型
        if eval_f1 > prev_f1_score:
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "acc_score": eval_acc,
                "f1_score": eval_f1,
                "max_seq_length": common_config.max_seq_length
                }, 
                os.path.join(common_config.checkpoint_dir, model_config.bert_bilstm_crf_model_name))
            print("save new model, new f1 & accuracy {} - {} ... previous f1 & accuracy  {} - {} ... ".format(eval_f1, eval_acc, prev_f1_score, prev_acc_score))
            prev_f1_score = eval_f1
            prev_acc_score = eval_acc