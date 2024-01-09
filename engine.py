import torch

from tqdm import tqdm

from utils import save_model
import os
import wandb
import json

from sklearn.metrics import f1_score, recall_score

def train_step(
        model: torch.nn.Module,
        train_loader,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
):
    """
    Train model for one epoch.

    Args:
        model: PyTorch model to train.
        train_loader: PyTorch dataloader for training data.
        loss_fn: PyTorch loss function.
        optimizer: PyTorch optimizer.
        device: PyTorch device to use for training.

    Returns:
        Average loss, accuracy, macro F1 score, and macro recall for the epoch.
    """

    model.train()
    train_loss = 0.0
    train_acc = 0.0
    train_targets = []
    train_predictions = []

    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = output.max(1)

        train_targets.extend(target.cpu().numpy())
        train_predictions.extend(predicted.cpu().numpy())

        train_acc += predicted.eq(target).sum().item() / len(target)

    train_loss /= len(train_loader)
    train_acc /= len(train_loader)
    train_macro_f1 = f1_score(train_targets, train_predictions, average='macro')
    train_macro_recall = recall_score(train_targets, train_predictions, average='macro')

    return train_loss, train_acc, train_macro_f1, train_macro_recall

def val_step(
        model: torch.nn.Module,
        val_loader,
        loss_fn: torch.nn.Module,
        device: torch.device,
):
    """
    Evaluate model on val data.

    Args:
        model: PyTorch model to evaluate.
        val_loader: PyTorch dataloader for val data.
        loss_fn: PyTorch loss function.
        device: PyTorch device to use for evaluation.

    Returns:
        Average loss, accuracy, macro F1 score, and macro recall for the validation set.
    """

    model.eval()
    val_loss = 0.0
    val_acc = 0.0
    val_targets = []
    val_predictions = []

    with torch.no_grad():
        for data, target in tqdm(val_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            val_loss += loss_fn(output, target).item()

            _, predicted = output.max(1)
            val_targets.extend(target.cpu().numpy())
            val_predictions.extend(predicted.cpu().numpy())

            val_acc += predicted.eq(target).sum().item() / len(target)

    val_loss /= len(val_loader)
    val_acc /= len(val_loader)
    val_macro_f1 = f1_score(val_targets, val_predictions, average='macro')
    val_macro_recall = recall_score(val_targets, val_predictions, average='macro')

    return val_loss, val_acc, val_macro_f1, val_macro_recall

def trainer(
        model: torch.nn.Module,
        train_loader,
        val_loader,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler,
        lr_scheduler_name: str,
        device: torch.device,
        epochs: int,
        save_dir: str,
        early_stopper=None,
        linear_probing_epochs=None,
        start_epoch = 1
):
    """
    Train and evaluate model.

    Args:
        model: PyTorch model to train.
        train_loader: PyTorch dataloader for training data.
        val_loader: PyTorch dataloader for val data.
        loss_fn: PyTorch loss function.
        optimizer: PyTorch optimizer.
        lr_scheduler: PyTorch learning rate scheduler.
        device: PyTorch device to use for training.
        epochs: Number of epochs to train the model for.

    Returns:
        Average loss and accuracy for the val set.
    """

    results = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "train_f1":[],
        "val_f1":[],
        "train_recall":[],
        "val_recall":[]
    }
    best_val_loss = 1e10

    for epoch in range(start_epoch, epochs + 1):

        if linear_probing_epochs is not None:
            if epoch == linear_probing_epochs:
                for param in model.parameters():
                    param.requires_grad = True

        print(f"Epoch {epoch}:")
        train_loss, train_acc, train_macro_f1, train_macro_recall = train_step(model, train_loader, loss_fn, optimizer, device)
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train F1: {train_macro_f1:.4f}, Train recall: {train_macro_recall:.4f}")

        

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["train_f1"].append(train_macro_f1)
        results["train_recall"].append(train_macro_recall)



        val_loss, val_acc, val_f1, val_recall = val_step(model, val_loader, loss_fn, device)
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}, Val recall: {val_recall:.4f}")
        print()

        if lr_scheduler_name == "ReduceLROnPlateau":
            lr_scheduler.step(val_loss)
        elif lr_scheduler_name != "None":
            lr_scheduler.step()
        
        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_acc)
        results["val_f1"].append(val_f1)
        results["val_recall"].append(val_recall)
        
        wandb.log({"train_loss": train_loss, "val_loss": val_loss, "train_acc": train_acc, "val_acc": val_acc, "train_f1": train_macro_f1, "train_recall": train_macro_recall, "val_f1": val_f1, "val_recall": val_recall})
        

        if lr_scheduler_name=="CyclicLR":
            checkpoint = { 
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_sched': lr_scheduler.state_dict()}
        else:
            checkpoint = { 
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_sched': lr_scheduler}
            
                    
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, os.path.join(save_dir, "best_checkpoint.pth"))

        torch.save(checkpoint, os.path.join(save_dir, "last_checkpoint.pth"))

        if early_stopper is not None:
            if early_stopper.early_stop(val_loss):
                print("Early stopping")
                break

    return results



# import torch

# from tqdm import tqdm

# from utils import save_model
# import os
# import wandb



# def train_step(
#         model: torch.nn.Module,
#         train_loader,
#         loss_fn: torch.nn.Module,
#         optimizer: torch.optim.Optimizer,
#         device: torch.device,
#         n_classes = 10,
#         mixup_alpha= None,
# ):
#     """
#     Train model for one epoch.

#     Args:
#         model: PyTorch model to train.
#         train_loader: PyTorch dataloader for training data.
#         loss_fn: PyTorch loss function.
#         optimizer: PyTorch optimizer.
#         device: PyTorch device to use for training.

#     Returns:
#         Average loss for the epoch.
#     """

#     model.train()
#     train_loss = 0.0
#     train_acc = 0.0
#     for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
#         if mixup_alpha!=None:
#             print('yesssssssss')
#             data, target = mixup(data, target, mixup_alpha, n_classes)
        
#         data, target = data.to(device), target.to(device)

#         optimizer.zero_grad()
#         output = model(data)

#         loss = loss_fn(output,target)
#         loss.backward()
#         optimizer.step()

#         train_loss += loss.item()

#         if mixup_alpha!=None:
#             _, target = target.max(dim=1)


#         _, predicted = output.max(1)



#         train_acc += predicted.eq(target).sum().item() / len(target)

#     train_loss /= len(train_loader)
#     train_acc /= len(train_loader)

#     return train_loss, train_acc

# def val_step(
#         model: torch.nn.Module,
#         val_loader,
#         loss_fn: torch.nn.Module,
#         device: torch.device,
# ):
#     """
#     Evaluate model on val data.

#     Args:
#         model: PyTorch model to evaluate.
#         val_loader: PyTorch dataloader for val data.
#         loss_fn: PyTorch loss function.
#         device: PyTorch device to use for evaluation.

#     Returns:
#         Average loss and accuracy for the val set.
#     """

#     model.eval()
#     val_loss = 0.0
#     val_acc = 0.0
#     with torch.no_grad():
#         for data, target in tqdm(val_loader):
#             data, target = data.to(device), target.to(device)

#             output = model(data)
#             val_loss += loss_fn(output, target).item()

#             _, predicted = output.max(1)
#             val_acc += predicted.eq(target).sum().item() / len(target)


#     val_loss /= len(val_loader)

#     val_acc /= len(val_loader)
#     return val_loss, val_acc


# def trainer(
#         model: torch.nn.Module,
#         train_loader,
#         val_loader,
#         loss_fn: torch.nn.Module,
#         optimizer: torch.optim.Optimizer,
#         lr_scheduler: torch.optim.lr_scheduler,
#         lr_scheduler_name: str,
#         device: torch.device,
#         epochs: int,
#         save_dir: str,
#         early_stopper=None,
#         linear_probing_epochs=None,
#         start_epoch = 1
# ):
#     """
#     Train and evaluate model.

#     Args:
#         model: PyTorch model to train.
#         train_loader: PyTorch dataloader for training data.
#         val_loader: PyTorch dataloader for val data.
#         loss_fn: PyTorch loss function.
#         optimizer: PyTorch optimizer.
#         lr_scheduler: PyTorch learning rate scheduler.
#         device: PyTorch device to use for training.
#         epochs: Number of epochs to train the model for.

#     Returns:
#         Average loss and accuracy for the val set.
#     """

#     results = {
#         "train_loss": [],
#         "val_loss": [],
#         "train_acc": [],
#         "val_acc": [],
#     }
#     best_val_loss = 1e10

#     for epoch in range(start_epoch, epochs + 1):

#         if linear_probing_epochs is not None:
#             if epoch == linear_probing_epochs:
#                 for param in model.parameters():
#                     param.requires_grad = True

#         print(f"Epoch {epoch}:")
#         train_loss, train_acc = train_step(model, train_loader, loss_fn, optimizer, device)
#         print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

        

#         results["train_loss"].append(train_loss)
#         results["train_acc"].append(train_acc)

#         val_loss, val_acc = val_step(model, val_loader, loss_fn, device)
#         print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
#         print()

#         if lr_scheduler_name == "ReduceLROnPlateau":
#             lr_scheduler.step(val_loss)
#         elif lr_scheduler_name != "None":
#             lr_scheduler.step()
        
#         results["val_loss"].append(val_loss)
#         results["val_acc"].append(val_acc)

#         wandb.log({"train_loss": train_loss, "val_loss": val_loss, "train_acc": train_acc, "val_acc": val_acc})

#         checkpoint = { 
#                 'epoch': epoch,
#                 'model': model.state_dict(),
#                 'optimizer': optimizer.state_dict(),
#                 'lr_sched': lr_scheduler}
        
#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
            
#             torch.save(checkpoint, os.path.join(save_dir, "best_checkpoint.pth"))

#         torch.save(checkpoint, os.path.join(save_dir, "last_checkpoint.pth"))

#         if early_stopper is not None:
#             if early_stopper.early_stop(val_loss):
#                 print("Early stopping")
#                 break

#     return results
