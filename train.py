import argparse
import os
import shutil
from tqdm import tqdm
from rerankai.arguments import DataArguments,ModelArguments,TrainArguments
from rerankai.dataset import TrainDatasetForCE
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn as nn
from torch import nn,optim
from torch.utils.data import DataLoader
from rerankai.dataset import *

class ModelLoss(nn.Module):
    """model loss"""
    def __init__(self):
        super().__init__()
        self.cross = nn.CrossEntropyLoss()

    def forward(self,logists,labels):
        out = {"loss":self.cross(logists,labels)}
        return out

class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_checkpoint(state, is_best, filename):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename[:-4] + "_best" + filename[-4:])

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def train_one_epoch(
    epoch,train_dataloader, model, optimizer,criterion,model_arg,data_arg,train_arg,clip_max_norm
):
    model.train()
    device = next(model.parameters()).device
    pbar = tqdm(train_dataloader,desc="Processing epoch "+str(epoch), unit="batch")
    total_loss = AverageMeter()
    average_hit_rate = AverageMeter()
    """ get data """
    for batch_id, batch_data in enumerate(pbar):
        batch_data_ = create_batch_data(batch_data)
        batch_size = train_arg.train_batch_size
        group_size = data_arg.train_group_size
        """ grad zeroing """
        optimizer.zero_grad()

        """ forward """
        used_memory = torch.cuda.memory_allocated(torch.cuda.current_device()) / (1024 ** 3)  
        output = model(batch_data_['input_ids'].to(device),batch_data_['attention_mask'].to(device),return_dict = True)

        """ calculate loss """
        scores = output["logits"].contiguous().view(batch_size,group_size)
        out_criterion = criterion(scores,batch_data_['labels'].to(device))
        after_used_memory = torch.cuda.memory_allocated(torch.cuda.current_device()) / (1024 ** 3) 
        out_criterion["loss"].backward()
        total_loss.update(out_criterion["loss"])
        average_hit_rate.update(math.exp(-total_loss.avg))

        """ grad clip """
        if clip_max_norm > 0:
            clip_gradient(optimizer,clip_max_norm)

        """ modify parameters """
        optimizer.step()
        postfix_str = "total_loss: {:.4f},average_hit_rate:{:.4f},use_memory: {:.1f}G".format(
            total_loss.avg, 
            average_hit_rate.avg,
            after_used_memory - used_memory
        )
        pbar.set_postfix_str(postfix_str)
        pbar.update()
    with open(train_arg.log_path, "a") as file:
        file.write(postfix_str+"\n")

def test_epoch(epoch, test_dataloader, model, criterion, model_arg, data_arg, train_arg):
    total_loss = AverageMeter()
    average_hit_rate = AverageMeter()
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        for batch_id, batch_data in enumerate(test_dataloader):
            batch_data_ = create_batch_data(batch_data)
            batch_size = train_arg.train_batch_size
            group_size = data_arg.train_group_size

            """ forward """
            output = model(batch_data_['input_ids'].to(device),batch_data_['attention_mask'].to(device),return_dict = True)

            """ calculate loss """
            scores = output["logits"].contiguous().view(batch_size,group_size)
            out_criterion = criterion(scores,batch_data_['labels'].to(device))
            total_loss.update(out_criterion["loss"])
            average_hit_rate.update(math.exp(-total_loss.avg))
    str = (        
        f"Test epoch {epoch}:"
        f"average_hit_rate:{average_hit_rate.avg:.4f} "
        f"total_loss: {total_loss.avg:.4f}\n"
        )
    print(str)
    with open(train_arg.log_path, "a") as file:
        file.write(str+"\n")
    return total_loss.avg


def main(args):
    """get config arg"""
    model_arg = ModelArguments(model_name_or_path = args.model_name_or_path,
                                                        finetune_model_dir = args.finetune_model_dir)
    
    data_arg = DataArguments(train_data = args.data_path,
                                                 train_group_size = args.train_group_size,
                                                 max_len = args.max_len)

    train_arg = TrainArguments(learning_rate = args.lr,
                                                  log_path = model_arg.finetune_model_dir + '/train.log',
                                                  train_batch_size = args.batch_size,
                                                  train_total_epoch = args.total_epoch)
    
    """get train device"""
    device = args.device if torch.cuda.is_available() else "cpu"

    """get net struction"""
    if os.path.isdir(model_arg.finetune_model_dir):
        tokenizer = AutoTokenizer.from_pretrained(model_arg.finetune_model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_arg.finetune_model_dir)
    else:
        os.makedirs(model_arg.finetune_model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_arg.model_name_or_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_arg.model_name_or_path,local_files_only=True)
    model.to(device)

    """get data loader"""
    train_datasets = TrainDatasetForCE(data_arg,tokenizer,train=True)
    test_dataset = TrainDatasetForCE(data_arg, tokenizer, train=False)
    train_dataloader = DataLoader(train_datasets,
                              batch_size = train_arg.train_batch_size, 
                              shuffle=True,
                              collate_fn = collate_fn)
    
    test_dataloader = DataLoader(test_dataset, 
                              batch_size = train_arg.train_batch_size, 
                              shuffle = True,
                              collate_fn = collate_fn)

    """get model loss criterion"""
    criterion = ModelLoss()

    """get optimizer"""
    optimizer = optim.Adam(model.parameters(),train_arg.learning_rate)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=args.factor, patience=args.patience)
    checkpoint_path = model_arg.finetune_model_dir + '/checkpoint.pth'
    train_arg.log_path = model_arg.finetune_model_dir + '/train.log'

    if not os.path.exists(checkpoint_path):   
        save_checkpoint(
            {
                "epoch": -1,
                "loss": float("inf"),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
            },
            True,
            checkpoint_path,
        )

    if  train_arg.log_path:
        if not os.path.exists( train_arg.log_path):   
            with open( train_arg.log_path, "w") as file:
                pass

    checkpoint = torch.load(checkpoint_path, map_location=device)
    last_epoch = checkpoint["epoch"] + 1
    optimizer.load_state_dict(checkpoint["optimizer"])
    lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    """ interfence """
    best_loss = float("inf")
    for epoch in range(last_epoch,train_arg.train_total_epoch):
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        train_one_epoch(epoch,
                        train_dataloader, 
                        model, 
                        optimizer,
                        criterion,
                        model_arg,
                        data_arg,
                        train_arg,
                        0.5)
        loss = test_epoch(epoch, 
                          test_dataloader, 
                          model, 
                          criterion,                        
                          model_arg,
                        data_arg,
                        train_arg)
        lr_scheduler.step(loss)
        is_best = loss < best_loss
        best_loss = min(loss, best_loss)
        save_checkpoint(
            {
                "epoch": epoch,
                "loss": loss,
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict()
            },
            is_best,
            checkpoint_path,
            )
        if is_best:
            model.save_pretrained(model_arg.finetune_model_dir)
            tokenizer.save_pretrained(model_arg.finetune_model_dir)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path",type=str,default = "BAAI/bge-reranker-base")
    parser.add_argument("--finetune_model_dir",type=str,default = "./saved_model/bge-reranker-base_finetune_nouser")
    parser.add_argument("--data_path",type=str,default = "data/0medical.jsonl")
    parser.add_argument("--train_group_size",type = int,default = 8)
    parser.add_argument("--max_len",type = int,default = 512)
    parser.add_argument("--batch_size",type = int,default = 2)
    parser.add_argument("--total_epoch",type = int,default = 1000)
    parser.add_argument("--lr",type=float,default=0.00001)
    parser.add_argument("--factor",type=float,default=0.3)
    parser.add_argument("--patience", type=int,default=10)   
    parser.add_argument("--device",type=str,default = "cuda")
    args = parser.parse_args()
    main(args)