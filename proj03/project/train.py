import os
import numpy as np
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import copy
from tensorboardX import SummaryWriter
from models import *
from dataloader import toRGB
from torch.autograd import Variable

def train(args, train_loader, val_loader):
    torch.manual_seed(1024)
    torch.cuda.manual_seed(1024)

    MODEL = UNet(22).to(args.device)

    optimizer = torch.optim.Adam(
        MODEL.parameters(), lr=1e-4)

    CEloss = torch.nn.CrossEntropyLoss(ignore_index=-1)
    CEloss.to(args.device)

    best_loss = 0.0
    best_model = MODEL
    patience = args.patience
    batch_size = args.batch_size
    writer = SummaryWriter('log/{}'.format(args.model_name))

    log_interval = int(len(train_loader) * 0.10)
    val_interval = int(len(train_loader) *0.25)
    print('train len: {}, val every {}, log every {}.'.format(len(train_loader), val_interval, log_interval))
    sys.stdout.flush()


    MODEL.train()
    for epoch in range(args.epochs):
        print('epoch:{}   patience:{}'.format(epoch, patience))
        sys.stdout.flush()
        
        for batch_idx, (sample_img, sample_tru, sample_truInt) in enumerate(train_loader):
            sample_img = sample_img.to(args.device)
            sample_tru = sample_tru.to(args.device)
            sample_truInt = sample_truInt.to(args.device)

            optimizer.zero_grad()
            output = MODEL(sample_img)
            loss = CEloss(output, sample_truInt)
            loss.backward()
            optimizer.step()

            if batch_idx % log_interval == 0:
                idx = epoch * int(len(train_loader.dataset) / batch_size) + batch_idx
                writer.add_scalar('loss', loss.item(), idx)
                writer.add_image('Label_img', sample_tru[0], idx)
                softmax=torch.argmax(output, dim=1)
                writer.add_image('Recon_img', toRGB(softmax.cpu().detach())[0], idx)
                writer.add_image('input_img', sample_img[0], idx)
                writer.flush()

            if batch_idx % val_interval == 0:
                val_loss = 0.0
                MODEL.eval()
                for batch_idx, (val_img, val_tru, val_truInt) in enumerate(val_loader):
                    val_img = val_img.to(args.device)
                    val_truInt = val_truInt.to(args.device)
                    output = MODEL(val_img)
                    softmax=torch.argmax(output, dim=1)
                    val_loss += ((softmax==val_truInt).sum()).item()
                val_loss = (val_loss*100)/(len(val_loader.dataset)*(val_img.shape[2]*val_img.shape[3]))
                MODEL.train()
                writer.add_scalar('val_loss', val_loss, idx)
                writer.flush()

                if best_loss < val_loss:
                    best_loss = val_loss
                    best_model = copy.deepcopy(MODEL)
                    torch.save(best_model.state_dict(), 'save/{}'.format(args.model_name))
                    print('Improved: current best_loss on val:{}'.format(best_loss))
                    sys.stdout.flush()
                    patience = args.patience
                    #assert False
                else:
                    if epoch > 200:
                        patience -= 1
                        if patience == 0:
                            torch.save(best_model.state_dict(), 'save/best/{}'.format(args.model_name))
                            print('Early Stopped: Best L1 loss on val:{}'.format(best_loss))
                            sys.stdout.flush()
                            writer.close()
                            return
                    print('patience', patience)
                    
            sys.stdout.flush()

    print('Finished: Best L1 loss on val:{}'.format(best_loss))
    sys.stdout.flush()
    writer.close()
