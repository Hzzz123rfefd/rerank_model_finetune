import sys
import os
sys.path.append(os.getcwd())

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
import argparse
import torch


def main(args):
    """ device """
    device = args.device if torch.cuda.is_available() else "cpu"

    """ load model """
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    model.to(device)

    """ load data """
    with open(args.data_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    query = data.get('query')
    kowndges = data.get('kowndges')
    sentence_pairs = [[query, kowndge] for kowndge in kowndges]

    """ inference """
    inputs = tokenizer(
            sentence_pairs, 
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt"
        )
    inputs.to(device)
    with torch.no_grad():
        scores_collection = []
        scores = model(**inputs, return_dict=True).logits.view(-1,).float()
        scores = torch.sigmoid(scores)
        if device == "cuda":
            scores = scores.cpu()
        scores_collection.extend(scores.cpu().numpy().tolist())

    print(scores_collection)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir",type = str,default="BAAI/bge-reranker-base")
    parser.add_argument("--data_path",type = str,default = "data/1.json")
    parser.add_argument("--max_length",type = int,default = 512)
    parser.add_argument("--device",type = str,default = "cuda")
    args = parser.parse_args()
    main(args)
