from config import load_args
from model import GPT2
from dataloader import get_loader
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm, tqdm_notebook, trange


def train_epoch(epoch,model,criterion,optimizer,train_loader):

    losses = []
    model.train()

    with tqdm_notebook(total=len(train_loader), desc=f"Train {epoch+1}") as pbar:

        for i,value in enumerate(train_loader):
            inputs,targets = value

            optimizer.zero_grad()
            outputs = model(inputs)
            logits = outputs[0]

            labels = targets.contiguous()

            loss = criterion(logits.view(-1,logits.size(2)),labels.view(-1))
            

            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            pbar.update(1)
            pbar.set_postfix_str(f"Loss: {loss:.3f} ({np.mean(losses):.3f})")
    return np.mean(losses)



def train():
    args = load_args()
    train_loader,_ = get_loader(args)
    model = GPT2(4007,args.heads,args.embedding_dim,args.N)
    device = 'cuda' if args.cuda else 'cpu'
    model.to(device)
    optimizer = optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    losses = []
    for epoch in range(args.epochs):
        loss = train_epoch(epoch,model,criterion,optimizer,train_loader)
        losses.append(loss)
