import os
import numpy as np
import pandas as pd
import random
import cv2
import torch
import tqdm
import matplotlib.pyplot as plt
from Cindex import concordance_index
from copy import deepcopy
from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score, roc_auc_score, accuracy_score, balanced_accuracy_score, precision_recall_fscore_support, matthews_corrcoef, f1_score, confusion_matrix, classification_report
from sklearn import metrics
from sklearn.preprocessing import LabelBinarizer
# from mlxtend.plotting import plot_confusion_matrix

def train(model, device, loss_fxn, optimizer, data_loader, history, epoch, model_dir):
    """
    Parameters
    ----------
        model : PyTorch model
        device : PyTorch device
        loss_fxn : PyTorch loss function
        ls : int
            Ratio of label smoothing to apply during loss computation
        optimizer : PyTorch optimizer
        data_loader : PyTorch data loader
        history : pandas DataFrame
            Data frame containing history of training metrics
        epoch : int
            Current epoch number (1-K)
        model_dir : str
            Path to output directory where metrics, model weights, etc. will be stored
        classes : list[str]
            Ordered list of names of output classes
        fusion : bool
            Whether or not fusion is being performed (image + metadata inputs)
        meta_only : bool
            Whether or not to train on *only* metadata as input
    Returns
    -------
        history : pandas DataFrame
            Updated history data frame with metrics from completed training epoch
    """
    model.train()

    pbar = tqdm.tqdm(enumerate(data_loader), total=len(data_loader), desc=f'Epoch {epoch}')
    running_loss = 0.
    ye_true, yt_true, y_hat = [], [], []
    acc_nums = []
    plax_probs = []
    for i, batch in pbar:
        x, y , cli= batch['x'].to(device), batch['y'].to(device),batch['cli'].to(device)
        acc_num = batch['ID']

        # Forward pass
        out = model(x,cli)
        # out = model(x)

        # out = model(torch.squeeze(x))

        # Compute loss
        loss = loss_fxn(out, y)

        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # Keep running sum of batch losses
        running_loss += loss.item()

        ye_true.append(y[:,0].detach().cpu().numpy())
        yt_true.append(y[:,1].detach().cpu().numpy())
        # y_hat.append(out.sigmoid().detach().cpu().numpy())
        y_hat.append(out.detach().cpu().numpy())
        acc_nums.append(acc_num)

        pbar.set_postfix({'loss': running_loss / (i + 1)})
    
    video_label_df = pd.DataFrame({'ye_true': np.concatenate(ye_true).ravel(),'yt_true': np.concatenate(yt_true).ravel(), 'y_hat': np.concatenate(y_hat).ravel(), 'ID': np.concatenate(acc_nums).ravel()})

    patient_label_df = video_label_df.groupby(by=['ID']).agg({'ye_true': np.mean,'yt_true': np.mean, 'y_hat': np.mean})

    # y_true = video_label_df['y_true']
    # y_hat = video_label_df['y_hat']
    #
    # # AUROC
    # auroc = roc_auc_score(y_true, y_hat)
    # pr, re, _ = precision_recall_curve(y_true, y_hat)
    # aupr = metrics.auc(re, pr)
    #
    # # Other metrics
    # acc = accuracy_score(y_true, y_hat.round())
    # b_acc = balanced_accuracy_score(y_true, y_hat.round())
    # mcc = matthews_corrcoef(y_true, y_hat.round())
    # pr, re, f1, _ = precision_recall_fscore_support(y_true, y_hat.round(), average='binary')
    #
    # out_str = f'[VIDEO LEVEL]   AUROC: {auroc:.3f} | AUPR: {aupr:.3f} | Acc: {acc:.3f} | BAcc: {b_acc:.3f} | MCC: {mcc:.3f} | Precision: {pr:.3f} | Recall: {re:.3f} | F1: {f1:.3f}'
    # print(out_str)

    ##  FOR PATIENT LEVEL METRICS
    ye_true = patient_label_df['ye_true']
    yt_true = patient_label_df['yt_true']
    y_hat = patient_label_df['y_hat']


    # C-index
    cindex = concordance_index(yt_true,y_hat,ye_true)

    # AUROC
    # auroc = roc_auc_score(y_true, y_hat)
    # pr, re, _ = precision_recall_curve(y_true, y_hat)
    # aupr = metrics.auc(re, pr)

    # Other metrics
    # acc = accuracy_score(y_true, y_hat.round())
    # b_acc = balanced_accuracy_score(y_true, y_hat.round())
    # mcc = matthews_corrcoef(y_true, y_hat.round())
    # pr, re, f1, _ = precision_recall_fscore_support(y_true, y_hat.round(), average='binary')

    # out_str = f'[STUDY LEVEL] AUROC: {auroc:.3f} | AUPR: {aupr:.3f} | Acc: {acc:.3f} | BAcc: {b_acc:.3f} | MCC: {mcc:.3f} | Precision: {pr:.3f} | Recall: {re:.3f} | F1: {f1:.3f}'
    out_str = f'[Train] c-index: {cindex:.5f} '
    print(out_str)

    # Append metrics to history df
    # current_metrics = pd.DataFrame([[epoch, 'train', running_loss/(i+1), auroc, aupr, acc, b_acc, mcc, pr, re, f1]], columns=history.columns)
    current_metrics = pd.DataFrame([[epoch, 'train', running_loss/(i+1), cindex]], columns=history.columns)
    current_metrics.to_csv(os.path.join(model_dir, 'history.csv'), mode='a', header=False, index=False)

    # return history.append(current_metrics) #df.append was deprecated
    return pd.concat([history,current_metrics])

def validate(model, device, loss_fxn, optimizer, data_loader, history, epoch, model_dir, early_stopping_dict, best_model_wts, weight_saving=False):

    model.eval()

    pbar = tqdm.tqdm(enumerate(data_loader), total=len(data_loader), desc=f'[VAL] Epoch {epoch}')
    running_loss = 0.
    ye_true,yt_true, y_hat = [], [], []
    acc_nums = []
    plax_probs = []
    with torch.no_grad():
        for i, batch in pbar:
            x, y , cli= batch['x'].to(device), batch['y'].to(device),batch['cli'].to(device)
            acc_num = batch['ID']

            # Forward pass
            out = model(x,cli)
            # out = model(x)

            # Get softmax probabilities
            # out = out.sigmoid().mean(dim=0)

            # Compute loss
            loss = loss_fxn(out, y)

            # Keep running sum of batch losses
            running_loss += loss.item()

            ye_true.append(y[:, 0].detach().cpu().numpy())
            yt_true.append(y[:, 1].detach().cpu().numpy())
            # y_hat.append(out.sigmoid().detach().cpu().numpy())
            y_hat.append(out.detach().cpu().numpy())
            acc_nums.append(acc_num)

            pbar.set_postfix({'loss': running_loss / (i + 1)})

    video_label_df = pd.DataFrame({'ye_true': np.concatenate(ye_true).ravel(),'yt_true': np.concatenate(yt_true).ravel(), 'y_hat': np.concatenate(y_hat).ravel(), 'ID': np.concatenate(acc_nums).ravel()})

    patient_label_df = video_label_df.groupby(by=['ID']).agg({'ye_true': np.mean,'yt_true': np.mean, 'y_hat': np.mean})


    ##  FOR PATIENT LEVEL METRICS
    ye_true = patient_label_df['ye_true']
    yt_true = patient_label_df['yt_true']
    y_hat = patient_label_df['y_hat']


    # C-index
    cindex = concordance_index(yt_true,y_hat,ye_true)

    out_str = f'[Validation] c-index: {cindex:.5f} '
    print(out_str)

    # Append metrics to history df
    val_loss = running_loss / (i + 1)
    # current_metrics = pd.DataFrame([[epoch, 'train', running_loss/(i+1), auroc, aupr, acc, b_acc, mcc, pr, re, f1]], columns=history.columns)
    current_metrics = pd.DataFrame([[epoch, 'val', running_loss/(i+1), cindex]], columns=history.columns)
    current_metrics.to_csv(os.path.join(model_dir, 'history.csv'), mode='a', header=False, index=False)

## Early Stopping for loss
    # if weight_saving == True:
    #     # Early stopping: save model weights only when val loss has improved
    #     if val_loss < early_stopping_dict['best_loss']:
    #         print(f'EARLY STOPPING: Loss has improved from {round(early_stopping_dict["best_loss"], 3)} to {round(val_loss, 3)}! Saving weights.')
    #         early_stopping_dict['epochs_no_improve'] = 0
    #         early_stopping_dict['best_loss'] = val_loss
    #         best_model_wts = deepcopy(model.state_dict()) if not isinstance(model, torch.nn.DataParallel) else deepcopy(model.module.state_dict())
    #         torch.save({'weights': best_model_wts, 'optimizer': optimizer.state_dict()}, os.path.join(model_dir, f'chkpt_epoch-{epoch}.pt'))
    #     else:
    #         print(f'EARLY STOPPING: Loss has not improved from {round(early_stopping_dict["best_loss"], 3)}')
    #         early_stopping_dict['epochs_no_improve'] += 1

## Early Stopping for C-index
    if weight_saving == True:
        # Early stopping: save model weights only when val loss has improved
        if cindex > early_stopping_dict['best_cindex']:
            print(f'EARLY STOPPING: C-index has improved from {round(early_stopping_dict["best_cindex"], 3)} to {round(cindex, 3)}! Saving weights.')
            early_stopping_dict['epochs_no_improve'] = 0
            early_stopping_dict['best_cindex'] = cindex
            early_stopping_dict['best_epoch'] = epoch
            best_model_wts = deepcopy(model.state_dict()) if not isinstance(model, torch.nn.DataParallel) else deepcopy(model.module.state_dict())
            torch.save({'weights': best_model_wts, 'optimizer': optimizer.state_dict()}, os.path.join(model_dir, f'chkpt_epoch-{epoch}.pt'))
        else:
            print(f'EARLY STOPPING: C-index has not improved from {round(early_stopping_dict["best_cindex"], 3)},@ {early_stopping_dict["best_epoch"]}')

            early_stopping_dict['epochs_no_improve'] += 1
    else:
        if cindex > early_stopping_dict['best_cindex']:
            print(f'EARLY STOPPING: C-index has improved from {round(early_stopping_dict["best_cindex"], 3)} to {round(cindex, 3)}! Saving weights.')
            early_stopping_dict['epochs_no_improve'] = 0
            early_stopping_dict['best_cindex'] = cindex
            early_stopping_dict['best_epoch'] = epoch
        else:
            print(f'EARLY STOPPING: C-index has not improved from {round(early_stopping_dict["best_cindex"], 3)},@ {early_stopping_dict["best_epoch"]}')
            early_stopping_dict['epochs_no_improve'] += 1
    # return history.append(current_metrics), early_stopping_dict, best_model_wts
    return pd.concat([history,current_metrics]), early_stopping_dict, best_model_wts,cindex

def evaluate(model, device, loss_fxn, data_loader, split, classes, history, model_dir, weights):
    if isinstance(model, torch.nn.DataParallel):
        model.module.load_state_dict(weights)
    else:
        model.load_state_dict(weights)
    model.eval()

    pbar = tqdm.tqdm(enumerate(data_loader), total=len(data_loader), desc=f'[{split.upper()}] EVALUATION')
    running_loss = 0.
    ye_true, yt_true, y_hat = [], [], []
    acc_nums = []
    plax_probs = []
    with torch.no_grad():
        for i, batch in pbar:
            x, y , cli= batch['x'].to(device), batch['y'].to(device),batch['cli'].to(device)
            acc_num = batch['ID']

            # Forward pass
            # out = torch.stack([model(x[:, :, clip, :, :, :]) for clip in range(x.shape[2])], dim=0)
            out = model(x,cli)
            # out = model(x)

            # Get softmax probabilities and video-wise average over clip dimension
            # out = out.sigmoid().mean(dim=0)
            # out = out[0].mean(dim=0)

            # Compute loss
            loss = loss_fxn(out, y)

            # Keep running sum of batch losses
            running_loss += loss.item()

            ye_true.append(y[:, 0].detach().cpu().numpy())
            yt_true.append(y[:, 1].detach().cpu().numpy())
            # y_hat.append(out.sigmoid().detach().cpu().numpy())
            y_hat.append(out.detach().cpu().numpy())
            acc_nums.append(acc_num)

            pbar.set_postfix({'loss': running_loss / (i + 1)})

        video_label_df = pd.DataFrame(
            {'ye_true': np.concatenate(ye_true).ravel(), 'yt_true': np.concatenate(yt_true).ravel(),
             'y_hat': np.concatenate(y_hat).ravel(), 'ID': np.concatenate(acc_nums).ravel()})

        patient_label_df = video_label_df.groupby(by=['ID']).agg(
            {'ye_true': np.mean, 'yt_true': np.mean, 'y_hat': np.mean})

    # y_true = video_label_df['y_true']
    # y_hat = video_label_df['y_hat']
    #
    # # AUROC
    # auroc = roc_auc_score(y_true, y_hat)
    # pr, re, _ = precision_recall_curve(y_true, y_hat)
    # aupr = metrics.auc(re, pr)
    #
    # # Other metrics
    # acc = accuracy_score(y_true, y_hat.round())
    # b_acc = balanced_accuracy_score(y_true, y_hat.round())
    # mcc = matthews_corrcoef(y_true, y_hat.round())
    # pr, re, f1, _ = precision_recall_fscore_support(y_true, y_hat.round(), average='binary')
    # cls_report = classification_report(y_true, y_hat.round(), target_names=classes, digits=3)
    #
    # video_out_str = f'[VIDEO LEVEL]   AUROC: {auroc:.3f} | AUPR: {aupr:.3f} | Acc: {acc:.3f} | BAcc: {b_acc:.3f} | MCC: {mcc:.3f} | Precision: {pr:.3f} | Recall: {re:.3f} | F1: {f1:.3f}'
    # video_out_str += f'\n{cls_report}'
    # print(video_out_str)

    ## REPEAT FOR PATIENT LEVEL METRICS
    ye_true = patient_label_df['ye_true']
    yt_true = patient_label_df['yt_true']
    y_hat = patient_label_df['y_hat']

    # C-index
    cindex = concordance_index(yt_true,y_hat,ye_true)
    # # AUROC
    # fpr, tpr, _ = roc_curve(y_true, y_hat)
    # auroc = roc_auc_score(y_true, y_hat)
    # prs, res, thrs = precision_recall_curve(y_true, y_hat)
    # aupr = metrics.auc(res, prs)

    # # Other metrics
    # acc = accuracy_score(y_true, y_hat.round())
    # b_acc = balanced_accuracy_score(y_true, y_hat.round())
    # mcc = matthews_corrcoef(y_true, y_hat.round())
    # pr, re, f1, _ = precision_recall_fscore_support(y_true, y_hat.round(), average='binary')
    # conf_mat = confusion_matrix(y_true, y_hat.round())
    # cls_report = classification_report(y_true, y_hat.round(), target_names=classes, digits=3)

    # out_str = f'[STUDY LEVEL] AUROC: {auroc:.3f} | AUPR: {aupr:.3f} | Acc: {acc:.3f} | BAcc: {b_acc:.3f} | MCC: {mcc:.3f} | Precision: {pr:.3f} | Recall: {re:.3f} | F1: {f1:.3f}'
    # out_str += f'\n{cls_report}'


    out_str = f'[External Validation] c-index: {cindex:.5f} '
    print(out_str)

    # Get threshold that maximizes F1
    # num = 2 * prs * res
    # den = prs + res
    # f1s = np.divide(num, den, out=np.zeros_like(den), where=(den != 0))
    #
    # idx = np.argmax(f1s)
    # best_thr = thrs[idx]
    # best_f1, best_pr, best_re = f1s[idx], prs[idx], res[idx]
    #
    # out_str += f'\nBest threshold: {best_thr}\n'
    # out_str += f'\tF1: {best_f1:.3f} | Precision: {best_pr:.3f} | Recall/Sensitivity: {best_re:.3f}'
    #
    # # Get threshold that maximizes F2
    # num = (1 + 2**2) * prs * res
    # den = (2**2 * prs) + res
    # f2s = np.divide(num, den, out=np.zeros_like(den), where=(den != 0))
    #
    # idx = np.argmax(f2s)
    # best_thr = thrs[idx]
    # best_f2, best_pr, best_re = f2s[idx], prs[idx], res[idx]
    #
    # out_str += f'\nBest threshold: {best_thr}\n'
    # out_str += f'\tF2: {best_f2:.3f} | Precision: {best_pr:.3f} | Recall/Sensitivity: {best_re:.3f}'
    #
    # print(out_str)

    # SAVE OUTPUT
    patient_label_df.to_csv(os.path.join(model_dir, f'{split}_preds.csv'))

    # Plot loss learing curves
    # fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    # ax.plot(history.loc[history['phase'] == 'train', 'epoch'], history.loc[history['phase'] == 'train', 'loss'], label='train')
    # ax.plot(history.loc[history['phase'] == 'val', 'epoch'], history.loc[history['phase'] == 'val', 'loss'], label='val')
    # ax.set_xlabel('Epoch')
    # ax.set_ylabel('Loss')
    # ax.legend()
    # fig.savefig(os.path.join(model_dir, 'loss_history.png'), dpi=300, bbox_inches='tight')
    #
    # # Plot AUROC learning curves
    # fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    # ax.plot(history.loc[history['phase'] == 'train', 'epoch'], history.loc[history['phase'] == 'train', 'auroc'], label='train')
    # ax.plot(history.loc[history['phase'] == 'val', 'epoch'], history.loc[history['phase'] == 'val', 'auroc'], label='val')
    # ax.set_xlabel('Epoch')
    # ax.set_ylabel('AUROC')
    # ax.legend()
    # fig.savefig(os.path.join(model_dir, 'auroc_history.png'), dpi=300, bbox_inches='tight')


    # # Plot AUPR (AP) learning curves
    # fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    # ax.plot(history.loc[history['phase'] == 'train', 'epoch'], history.loc[history['phase'] == 'train', 'aupr'], label='train')
    # ax.plot(history.loc[history['phase'] == 'val', 'epoch'], history.loc[history['phase'] == 'val', 'aupr'], label='val')
    # ax.set_xlabel('Epoch')
    # ax.set_ylabel('AUPR')
    # ax.legend()
    # fig.savefig(os.path.join(model_dir, 'aupr_history.png'), dpi=300, bbox_inches='tight')

    # Plot confusion matrix
    # fig, ax = plot_confusion_matrix(conf_mat, figsize=(6, 6), colorbar=True, show_absolute=True, show_normed=True, class_names=classes)
    # ax.set_xlabel('Predicted Label', fontsize=13)
    # ax.set_ylabel('True Label', fontsize=13)
    # fig.savefig(os.path.join(model_dir, f'{split}_cm.png'), dpi=300, bbox_inches='tight')

    # Plot ROC
    # fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    # ax.plot(fpr, tpr, lw=2, label=f'DIM (AUC: {auroc:.3f})')
    # ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    # ax.set_xlim([-0.05, 1.0])
    # ax.set_ylim([0.0, 1.05])
    # ax.set_xlabel('1 - Specificity', fontsize=13)
    # ax.set_ylabel('Sensitivity', fontsize=13)
    # ax.legend(loc="lower right", fontsize=11)
    # fig.savefig(os.path.join(model_dir, f'{split}_roc.png'), dpi=300, bbox_inches='tight')
    #
    # # Plot PR curve
    # fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    # ax.plot(res, prs, lw=2, label=f'DIM (AUC: {aupr:.3f})')
    # ax.axhline(y=y_true.sum()/y_true.size, color='navy', lw=2, linestyle='--')
    # ax.set_xlim([-0.05, 1.05])
    # ax.set_ylim([-0.05, 1.05])
    # ax.set_xlabel('Recall', fontsize=13)
    # ax.set_ylabel('Precision', fontsize=13)
    # ax.legend(loc="upper right", fontsize=11)
    # fig.savefig(os.path.join(model_dir, f'{split}_pr.png'), dpi=300, bbox_inches='tight')

    # Create summary text describing final performance
    # summary = video_out_str
    # summary += '-' * 20 + '\n'
    # summary += out_str
    # f = open(os.path.join(model_dir, f'{split}_summary.txt'), 'w')
    # f.write(summary)
    # f.close()

def printscores(model, device, loss_fxn, data_loader, split, model_dir, weights):
    """
    Parameters
    ----------
        model : PyTorch model
        device : PyTorch device
        loss_fxn : PyTorch loss function
        ls : int
            Ratio of label smoothing to apply during loss computation
        optimizer : PyTorch optimizer
        data_loader : PyTorch data loader
        history : pandas DataFrame
            Data frame containing history of training metrics
        epoch : int
            Current epoch number (1-K)
        model_dir : str
            Path to output directory where metrics, model weights, etc. will be stored
        classes : list[str]
            Ordered list of names of output classes
        fusion : bool
            Whether or not fusion is being performed (image + metadata inputs)
        meta_only : bool
            Whether or not to train on *only* metadata as input
    Returns
    -------
        history : pandas DataFrame
            Updated history data frame with metrics from completed training epoch
    """
    if isinstance(model, torch.nn.DataParallel):
        model.module.load_state_dict(weights)
    else:
        model.load_state_dict(weights)
    model.eval()

    pbar = tqdm.tqdm(enumerate(data_loader), total=len(data_loader), desc=f'[{split.upper()}] EVALUATION')
    running_loss = 0.
    ye_true, yt_true, y_hat = [], [], []
    acc_nums = []
    plax_probs = []
    with torch.no_grad():
        for i, batch in pbar:
            x, y , cli= batch['x'].to(device), batch['y'].to(device),batch['cli'].to(device)
            acc_num = batch['ID']

            # Forward pass
            out = model(x,cli)

            # Get softmax probabilities and video-wise average over clip dimension
            # out = out.sigmoid().mean(dim=0)
            # out = out[0].mean(dim=0)

            # Compute loss
            loss = loss_fxn(out, y)

            # Keep running sum of batch losses
            running_loss += loss.item()

            ye_true.append(y[:, 0].detach().cpu().numpy())
            yt_true.append(y[:, 1].detach().cpu().numpy())
            # y_hat.append(out.sigmoid().detach().cpu().numpy())
            y_hat.append(out.detach().cpu().numpy())
            acc_nums.append(acc_num)

            pbar.set_postfix({'loss': running_loss / (i + 1)})

        video_label_df = pd.DataFrame(
            {'ye_true': np.concatenate(ye_true).ravel(), 'yt_true': np.concatenate(yt_true).ravel(),
             'y_hat': np.concatenate(y_hat).ravel(), 'ID': np.concatenate(acc_nums).ravel()})

        patient_label_df = video_label_df.groupby(by=['ID']).agg(
            {'ye_true': np.mean, 'yt_true': np.mean, 'y_hat': np.mean})

    ## REPEAT FOR PATIENT LEVEL METRICS
    ye_true = patient_label_df['ye_true']
    yt_true = patient_label_df['yt_true']
    y_hat = patient_label_df['y_hat']

    # C-index
    cindex = concordance_index(yt_true,y_hat,ye_true)
    out_str = f'[External Validation] c-index: {cindex:.5f} '
    print(out_str)

    # SAVE OUTPUT
    patient_label_df.to_csv(os.path.join(model_dir, f'{split}_preds.csv'))

def load_sequences(fpath,fname, seqnum):
    if not os.path.exists(fpath):
        raise FileNotFoundError(fpath)
    v = np.zeros((seqnum, 256, 256, 3), np.float32)
    foldsum = os.listdir(fpath)
    seqsum = [seq for seq in foldsum if fname in seq]

    for i in range(seqnum):
        sequence = np.load(os.path.join(fpath,seqsum[i]))
        sequence_fliped = np.transpose(sequence, (1, 2, 0)) ## SITK stupidity needing matrix revertion
        sequence_resized = cv2.resize(sequence_fliped, dsize=(256, 256))
        v[i] = sequence_resized
    return v

## Arrange the data in [[sagittal],[coronal],[axial]], [sagittal]=[bz, channel, slice, h, w]
def load_sequences_XSA(fpath, fname, seqnum=4):
    assert seqnum in [4,6], "sequence number must be in [4,6]"
    """
    Loads and processes image sequences for XSA, ensuring correct order.

    Args:
        fpath (str): Directory path containing the sequences.
        fname (str): Identifier used to filter relevant sequences.
        seqnum (int): Expected number of sequences (default: 4 for dwipre, t1pre, t2pre, t1post).

    Returns:
        np.ndarray: Processed sequences with shape (seqnum, 1, 3, 256, 256).
    """
    # Define expected modality order
    if seqnum ==4:
        modalities = ["dwipre", "t2pre", "t1pre",  "premask"]
    if seqnum==6:
        modalities = ["dwipre", "t2pre", "t1pre", "premask","t1post","t1tumourpost"]
    # Initialize array to store sequences efficiently
    v = np.empty((seqnum, 3, 256, 256), dtype=np.float32)
    if '.0' in fname:
        fname = fname.replace('.0','')
    # Find and sort files based on the expected modality order
    files = {mod: None for mod in modalities}  # Dictionary to store file paths

    for filename in os.listdir(fpath):
        for mod in modalities:
            if filename.startswith(f"{fname}_{mod}"):
                files[mod] = filename
                break  # Stop checking once a match is found

    # Ensure all required modalities are present
    if None in files.values():
        missing = [mod for mod, file in files.items() if file is None]
        raise ValueError(f"Missing expected modalities for {fname}: {missing}")

    # Load and process files in the correct order
    for i, mod in enumerate(modalities):
        seqname = files[mod]
        sequence = np.load(os.path.join(fpath, seqname))
        ## for one sliced t1 post tumour ROI
        # if mod == "t1tumourpost":
        #     sequence = np.array([sequence,sequence,sequence])
        # if sequence.shape[0] != 3:  # Ensure the sequence has the expected channels
        #     raise ValueError(f"Unexpected shape {sequence.shape} in {seqname}, expected (3, H, W)")

        # Adjust matrix orientation for compatibility
        sequence_flipped = np.transpose(sequence, (1, 2, 0))  # (3, H, W) -> (H, W, 3)
        sequence_resized = cv2.resize(sequence_flipped, dsize=(256, 256))  # Resize to (256, 256)
        sequence_XSA = np.transpose(sequence_resized, (2, 1, 0))  # (H, W, 3) -> (3, 256, 256)

        v[i] = sequence_XSA  # Store processed sequence
    # return v[:, None, :, :, :]  # Reshape to (seqnum, 1, 3, 256, 256)
    return v.reshape(seqnum, 1, 3, 256, 256)  # Reshape to [[sagittal], [coronal], [axial]]

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True # sometimes not PERFECTLY deterministic when True, but major speedup during training
