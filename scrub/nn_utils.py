import torch
import numpy as np
import random
from tqdm import tqdm

from grad_utils import getGradObjs, gradNorm


def do_epoch(model, dataloader, criterion, epoch, nepochs, optim=None, device='cpu', outString='', compute_grads=False, retrain=False):
    # saves last two epochs gradients for computing finite difference Hessian
    total_loss = 0
    total_accuracy = 0
    grad_bank = None 
    nsamps = 0
    if optim is not None:
        model.train()
    else:
        model.eval()

    if compute_grads:
        total_gradnorm = 0
    for x, y_true in tqdm(dataloader, leave=False):
    #for _, (x, y_true) in enumerate(dataloader):
        # this is to skip the last batch if it is size 1
        # Needed if model has batchnorm layers
        # comment IN if batchnorm error
        # if x.shape[0] == 1:
        #    continue
        x, y_true = x.to(device), y_true.to(device)
        y_pred = model(x)
        loss = criterion(y_pred, y_true)
        #loss = criterion(y_pred, y_true.float())

        # for training
        if optim is not None:
            optim.zero_grad()
            loss.backward()
            optim.step()

            # saving for full/old Hessian estimate for scrubbing
            if epoch >= nepochs-2 and not retrain:
                batch_gradbank, param_bank = getGradObjs(model)
                if grad_bank is None:
                    grad_bank = batch_gradbank
                else:
                    for key in grad_bank.keys():
                        grad_bank[key] += batch_gradbank[key]

        # validation and others
        else:
            if compute_grads:

                model.zero_grad()
                loss.backward()
                batch_norm = gradNorm(model)

                total_gradnorm += batch_norm


        nsamps += len(y_true)
        total_loss += loss.item()
        total_accuracy += (y_pred.max(1)[1] == y_true).float().sum().item()

    if optim is not None:
        if epoch >= nepochs-2  and not retrain:
            for key in grad_bank.keys():
                grad_bank[key] = grad_bank[key]/nsamps
            print(f'saving params at epoch {epoch}...')
            torch.save(param_bank, outString + f'_epoch_{epoch}_params.pt')
            print(f'saving gradients at epoch {epoch}...')
            torch.save(grad_bank, outString + f'_epoch_{epoch}_grads.pt')

    mean_loss = total_loss / len(dataloader)
    mean_accuracy = total_accuracy / nsamps

    if compute_grads:
        mean_gradnorm = total_gradnorm / len(dataloader)
        return mean_loss, mean_accuracy, mean_gradnorm
    else:
        return mean_loss, mean_accuracy


def retrain_model(model, train_loader, val_loader, criterion, nepochs, optim, device='cpu'):
    
    print("\t ######### RETRAINING MODEL #########")
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=200)

    for epoch in range(nepochs):
        train_loss, train_accuracy = do_epoch(model, train_loader, criterion, epoch, nepochs, optim=optim, device=device, outString='', compute_grads=False, retrain=True)

        # Not doing validation passes as we don't care much about that, just do once at the end to report final performance
        # with torch.no_grad():
        #     val_loss, val_accuracy = do_epoch(model, val_loader, criterion, epoch, nepochs, optim=None, device=device, outString='', compute_grads=False, retrain=True)

        # tqdm.write(f'{args.model} EPOCH {epoch:03d}: train_loss={train_loss:.4f}, train_accuracy={train_accuracy:.4f} '
        #            f'val_loss={val_loss:.4f}, val_accuracy={val_accuracy:.4f}')

        tqdm.write(f'\t EPOCH {epoch:03d}: train_loss={train_loss:.4f}, train_accuracy={train_accuracy:.4f} ')

        lr_scheduler.step()          # For CosineAnnealingLR

    with torch.no_grad():
        val_loss, val_accuracy = do_epoch(model, val_loader, criterion, 0, 0, optim=None, device=device, outString='', compute_grads=False, retrain=True)

    retrain_loss, retrain_accuracy, retrain_gradnorm = do_epoch(model, train_loader, criterion, 0, 0, optim=None, device=device, outString='', compute_grads=True, retrain=True)

    return val_loss, val_accuracy, retrain_loss, retrain_accuracy, retrain_gradnorm


def manual_seed(seed):
    print("Setting seeds to: ", seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
