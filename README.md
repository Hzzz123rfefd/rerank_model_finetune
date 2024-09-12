# rerank_model_finetune
finetune rerank model like bce_reranker_bce、bge_reranker_base、bge_reranker_large
## Installation
Operating System: Linux
```bash
```
## Usage
### fintune
1、prepare finetune data,using jsonl format
"pos" refer to the kownledge relate to query
"neg" refer to the kownledge unrelate to query
```jsonl
{"query": " ", "pos": [" ",...," "], "neg": [" ",..., " "]}
{"query": " ", "pos": [" ",...," "], "neg": [" ",..., " "]}
{"query": " ", "pos": [" ",...," "], "neg": [" ",..., " "]}
```
* sh
```bash
python train.py --model_name_or_path {base reranker model path or name} \
                         --finetune_model_dir {finetune model dir} \
                         --data_path {finetune data path} \
                         --train_group_size {a query corresponds to one positive example and train_group_size-1  negative examples} \
                         --max_len {query + konwdge max tokens}\
                         --batch_size 2 \
                         --total_epoch 1000 \
                         --lr {learnning rate} \
                         --factor {learnning rate attenuation coefficient} \
                         --patience {learnning rate attenuation threshold} \
                         --device cuda
```
* example
```bash
python train.py --model_name_or_path BAAI/bge-reranker-base \
                         --finetune_model_dir ./saved_model/test \
                         --data_path data/0medical.jsonl \
                         --train_group_size 8 \
                         --max_len 512\
                         --batch_size 2 \
                         --total_epoch 1000 \
                         --lr 1e-5 \
                         --factor 0.3 \
                         --patience 8 \
                         --device cuda
```
### rerank
1、prepare rerank data,using json format
```json
{"query":"query","kowndges":["kowndge1","kowndge2".....]}
```
* sh
```bash
python example/rerank.py --model_dir {model dir} \
                                           --data_path {rerank data path} \
                                           --max_length {max padding length} \
                                           --device cuda
```
* example
```bash
python example/rerank.py --model_dir BAAI/bge-reranker-base \
                                           --data_path data/1.json \
                                           --max_length 512 \
                                           --device cuda
```

