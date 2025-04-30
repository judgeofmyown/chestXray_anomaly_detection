import torch
import torchvision
from torch import nn, optim
from torch.cuda.amp import autocast, GradScaler
import logging

logging.basicConfig(
    filename='training.log',
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    filemode='w'
)

def train(epochs, loader, model, device, optimizer):

    for param in model.backbone.parameters():
        param.requires_grad = False
    for param in model.roi_heads.parameters():
        param.requires_grad = True
    model.train()

    loss = []
    for epoch in range(epochs):
        epoch_loss=0
        for batch_idx, (images, targets) in enumerate(loader):
            images = images.to(device)
            targets = [{k : v.to(device) for k,v in t.items()} for t in targets]

            optimizer.zero_grad()

            op_loss_dict = model(images, targets)
            loss_sum = sum(v for v in op_loss_dict.values())

            loss_sum.backward()
            optimizer.step()
            
            epoch_loss += loss_sum.cpu().detach().numpy()
            print(epoch_loss)
        loss.append(epoch_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
        logging.info(f'Epoch {epoch} | Loss: {loss:.4f}')