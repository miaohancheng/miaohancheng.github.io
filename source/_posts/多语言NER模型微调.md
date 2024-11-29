---
title: 多语言NER模型微调
author: Miao HanCheng
date: 2024-11-29 15:00:00
tag: python
---



### 背景

工作中有遇到多语言的地址、短句等数据，需要标注出其中人名、快递公司、电话等信息，现有的开源数据集中这部分数据较少，主要问题是要自己构建对应的数据集、以及对于开源通过wiki训练的模型要尽可能保留训练的经验

### 模型选取

选择了RoBERTa基于 [**wikiann**](https://huggingface.co/datasets/unimelb-nlp/wikiann ) 预训练的模型 [**roberta-ner-multilingual**](https://huggingface.co/julian-schelb/roberta-ner-multilingual )，对于现实中的知识已经有一部分理解，但是对于地址等信息知识较少需要额外训练；

最原始的基座模型是facebook的 [**xlm-roberta-large**](https://huggingface.co/FacebookAI/xlm-roberta-large)

### 数据准备

准备了训练和测试数据集放在了 `./data`目录下，数据大致格式如下

```json
John    B-PER
lives   O
in      O
Berlin  B-LOC
.      O

他      O
住      O
在      O
北京    B-LOC
。      O
```

通过不同的标签定义不同的实体，在模型训练中可以通过单一语言的样本迁移一部分知识到其他语言，这也是多语言模型的一个优势。



### 模型加载

因为我在本机mac上训练，后续迁移到服务器上，所以写了判断设备的代码

```python
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
from transformers import DataCollatorForTokenClassification
from datasets import Dataset, DatasetDict
import numpy as np
import evaluate
from torch import nn
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 确定设备
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

# 加载分词器和模型
tokenizer = AutoTokenizer.from_pretrained("julian-schelb/roberta-ner-multilingual", add_prefix_space=True)
model = AutoModelForTokenClassification.from_pretrained("julian-schelb/roberta-ner-multilingual")
model.to(device)  # 将模型移动到设备

```



### 增加标签

原有的模型中标签类别较少，如果有新的标签可以手动添加

```python
# 如果有新标签，更新标签列表和映射
original_labels = list(model.config.id2label.values())
new_labels = ['B-FACILITY','I-FACILITY' ] # 加入你新的标签即可
label_list = original_labels + new_labels
label_to_id = {label: idx for idx, label in enumerate(label_list)}
id_to_label = {idx: label for idx, label in enumerate(label_list)}
```



更新模型配置

```python
# 更新模型配置
model.config.num_labels = len(label_list)
model.config.id2label = id_to_label
model.config.label2id = label_to_id

# 替换分类层
model.classifier = nn.Linear(model.config.hidden_size, model.config.num_labels)
```



### 教师模型

希望模型在学习新知识的前提下，不要遗忘旧知识，所以再增加一个教师模型来做监督

```python
# 加载教师模型
teacher_model = AutoModelForTokenClassification.from_pretrained("julian-schelb/roberta-ner-multilingual")
teacher_model.to(device)  # 将教师模型移动到设备
teacher_model.eval()

# 可选：冻结学生模型的部分层
for name, param in model.named_parameters():
    if name.startswith('roberta.embeddings') or name.startswith('roberta.encoder.layer.0') or name.startswith('roberta.encoder.layer.1'):
        param.requires_grad = False

```



### 加载数据

```python
# 定义读取数据的函数
def read_conll_data(filename):
    tokens = []
    ner_tags = []
    with open(filename, 'r', encoding='utf-8') as f:
        words = []
        tags = []
        for line in f:
            line = line.strip()
            if not line:
                if words:
                    tokens.append(words)
                    ner_tags.append(tags)
                    words = []
                    tags = []
            else:
                splits = line.split()
                if len(splits) >= 2:
                    words.append(splits[0])
                    tags.append(splits[1])
        if words:
            tokens.append(words)
            ner_tags.append(tags)
    return {'tokens': tokens, 'ner_tags': ner_tags}

# 加载您的数据
train_data = read_conll_data('./data/train.txt')
validation_data = read_conll_data('./data/validation.txt')

train_dataset = Dataset.from_dict(train_data)
validation_dataset = Dataset.from_dict(validation_data)

# 创建数据集对象
datasets = DatasetDict({'train': train_dataset, 'validation': validation_dataset})
```





### 数据预处理

```python
# 定义数据预处理函数
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples['tokens'],
        truncation=True,
        is_split_into_words=True,
        padding='longest',  # 启用填充
        max_length=512,     # 可根据需要调整
    )
    labels = []
    for i in range(len(tokenized_inputs['input_ids'])):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label_to_id.get(examples['ner_tags'][i][word_idx], label_to_id['O']))
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs['labels'] = labels
    return tokenized_inputs
# 预处理数据
tokenized_datasets = datasets.map(tokenize_and_align_labels, batched=True)

```



### 评估指标

```python

# 加载评估指标
metric = evaluate.load('seqeval')

# 定义评估函数
def compute_metrics(p):

    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_labels = [
        [label_list[l] for l in label if l != -100]
        for label in labels
    ]
    true_predictions = [
        [label_list[pred] for (pred, lab) in zip(prediction, label) if lab != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        'precision': results['overall_precision'],
        'recall': results['overall_recall'],
        'f1': results['overall_f1'],
        'accuracy': results['overall_accuracy']
    }

# 定义 DataCollator
data_collator = DataCollatorForTokenClassification(
    tokenizer,
    padding='longest',
    max_length=512,
    return_tensors='pt'
)

```



### 定义trainer

因为要做知识蒸馏，所以需要自定义trainer，结合教师模型的结果来做损失判定

```python

# 定义自定义 Trainer，用于知识蒸馏
class DistillationTrainer(Trainer):
    def __init__(self, teacher_model, temperature=2.0, alpha=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.temperature = temperature
        self.alpha = alpha
        self.teacher_model.eval()

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels").to(device)
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        outputs = model(**inputs)
        student_logits = outputs.logits

        # 计算学生模型的损失（交叉熵损失）
        loss_fct = nn.CrossEntropyLoss()
        active_loss = labels.view(-1) != -100
        active_logits = student_logits.view(-1, self.model.config.num_labels)[active_loss]
        active_labels = labels.view(-1)[active_loss]
        student_loss = loss_fct(active_logits, active_labels)

        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)
            teacher_logits = teacher_outputs.logits

        # 提取学生模型中对应于原始标签的 logits
        num_labels_teacher = teacher_logits.size(-1)  # 7
        student_logits_for_kd = student_logits[:, :, :num_labels_teacher]

        # 计算蒸馏损失（KL 散度）
        loss_fct = nn.KLDivLoss(reduction='batchmean')
        student_logits_temp = student_logits_for_kd / self.temperature
        teacher_logits_temp = teacher_logits / self.temperature

        distillation_loss = loss_fct(
            F.log_softmax(student_logits_temp, dim=-1),
            F.softmax(teacher_logits_temp, dim=-1)
        ) * (self.temperature ** 2)
        print("Student logits shape:", student_logits.shape)
        print("Teacher logits shape:", teacher_logits.shape)
        print("Student logits for KD shape:", student_logits_for_kd.shape)
        # 合并损失
        loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss

        return (loss, outputs) if return_outputs else loss

# 定义训练参数，使用较小的学习率
# 定义训练参数
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    logging_dir='./logs',
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    dataloader_pin_memory=False,
)

# 初始化自定义 Trainer
trainer = DistillationTrainer(
    teacher_model=teacher_model,
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
    compute_metrics=compute_metrics,
    data_collator=data_collator,
)

# 开始训练
trainer.train()
```





### 最后保存模型

``` python
# 保存模型和分词器
trainer.save_model('./my_trained_model')
tokenizer.save_pretrained('./my_trained_model')
```





### 结论

目前测试下来，如果整体数据集都是全新的数据，增加教师模型对于模型训练帮助不大，不如直接开始微调基座模型，因为标签和数据都是未曾见过的，教师模型无法给出建议，只会有干扰，但是这对于新数据和旧数据有一些重叠的情况会有所帮助。
