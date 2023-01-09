# Counseling Chatbot with Wellness Dataset

## Award
[1st prize in AI for Future Society 2022](https://github.com/payitfoward/counseling_chatbot/blob/main/%EC%83%81%EC%9E%A5.jpg)

## How to Use

### Chatbot Model

1. Run preprocess.py

2. Run build_graph.py

3. Run train_bert.py

4. Run roberta_wellness_chatbot.py

### APP
1. Build ./counselingApp (have to check your server ip)

2. After entering the message, press the send button and the chatbot answers.
<img src="https://user-images.githubusercontent.com/92314556/211301855-308256b7-df7a-4d11-a095-93e57c968e75.png"  width="30%" />

#### Skill Stack
- Firebase
- HTTP connect
- Asychronous
- Recyclerview



## Requirements

- `dgl-cu113 == 0.9.1.post1`
- `ignite == 1.1.0`
- `python == 3.6.9`
- `torch == 1.10.0+cu113`
- `scikit-lear n== 0.24.2`
- `transformers == 4.18.0`
- `numpy =< 1.19.5`
- `networkx == 2.5.1`

## Future work

- make graph for hierarchical class
- Auto build graph for unseem data ( Now, Just Replace Embedding of last node. So, prediction task is abnormal )

## Reference

### Dataset

- [웰니스 대화 스크립트 데이터셋](https://aihub.or.kr/aihubdata/data/view.do?currMenu=120&topMenu=100&aihubDataSe=extrldata&dataSetSn=267)

### Backbone Model

- [klue/roberta-base](https://huggingface.co/klue/roberta-base)

### Citation

- [BertGCN: Transductive Text Classification by Combining GCN and BERT](https://arxiv.org/abs/2105.05727)

## Workers

### [노영준](https://github.com/youngjun-99)
- Flask API
- Change build graph method
- Preprocessing
- Finetune KLUE/roberta-base

### [이가은](https://github.com/gaeun5744)
- Develop Application

### [박정원](https://github.com/jardin00)
- Finetune RoBERTaGCN 
