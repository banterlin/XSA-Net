import os,sys,warnings
import shutil
import argparse
import numpy as np
import pandas as pd
import torch
import torchvision
from sklearn.utils import compute_class_weight
from utils import train, validate, evaluate, seed_worker, set_seed,printscores
from dataset import XSADataset
from coxloss import CoxLoss
from model import model as XSA



if not sys.warnoptions:
    warnings.simplefilter("ignore")

def main(args):
    # Set device
    device = torch.device('cuda:0')
    #torch.cuda.get_device_name(device_id)
    MODEL_NAME = args.model_name
    MODEL_NAME += args.exp_name
    # MODEL_NAME += f'_{args.label.upper()}'
    MODEL_NAME += f'_frac-{args.frac}' if args.frac != 1.0 else ''
    # MODEL_NAME += '_Cox' if args.use_cox_loss else ''
    # MODEL_NAME += f'_ssl-{args.ssl.split("/")[-1]}' if args.ssl != '' else ''
    # MODEL_NAME += f'_{args.ssl}' if args.ssl != '' else ''
    MODEL_NAME += '_aug' if args.augment else ''

    # MODEL_NAME += f'_clip-len-{args.clip_len}-stride-{args.sampling_rate}'
    # MODEL_NAME += f'_num-clips-{args.num_clips}'
    MODEL_NAME += '_cw' if args.use_class_weights else ''
    MODEL_NAME += f'_lr-{args.lr}'
    MODEL_NAME += f'_{args.epochs}ep'
    MODEL_NAME += f'_patience-{args.patience}' if args.patience != 1e4 else ''
    MODEL_NAME += f'_bs-{args.batch_size}'
    MODEL_NAME += f'_ls-{args.label_smoothing}' if args.label_smoothing != 0. else ''
    # MODEL_NAME += f'_drp-0.25' if args.dropout_fc else ''
    MODEL_NAME += f'_weight-avg' if args.weight_averaging else ''
    MODEL_NAME += f'_TTA-{args.n_TTA}' if args.n_TTA > 0 else ''
    MODEL_NAME += f'_seed-{args.seed}' if args.seed != 0 else ''

    print(MODEL_NAME)
    if args.eval_only == '':
        # Create output directory for model (and delete if already exists)
        if not os.path.exists(args.output_dir):
            os.mkdir(args.output_dir)

        model_dir = os.path.join(args.output_dir, MODEL_NAME)
        if os.path.isdir(model_dir):
            shutil.rmtree(model_dir)
        os.mkdir(model_dir)

    # Set all seeds for reproducibility
    set_seed(args.seed)



    # Create model
    if args.model_name == 'res':
        model = torchvision.models.video.r3d_18(pretrained=not args.rand_init)

        if args.dropout_fc:
            model.fc = torch.nn.Sequential(torch.nn.Linear(512, 1), torch.nn.Dropout(0.25))

            if args.lpft != '':
                weights = dict(torch.load(args.lpft, map_location='cpu')['weights'])

                model.fc[0].weight.data = weights['fc.0.weight']
                model.fc[0].bias.data = weights['fc.0.bias']

        else:
            model.fc = torch.nn.Linear(512, 1)
            if args.lpft != '':
                weights = dict(torch.load(args.lpft, map_location='cpu')['weights'])

                model.fc.weight.data = weights['fc.weight']
                model.fc.bias.data = weights['fc.bias']

        if args.ssl != '':
            # ssl_path = os.path.join('/data/smart/lxq/mmdl/sslmodels/',args.ssl)
            # checkpoints = [f for f in os.listdir(ssl_path) if f.endswith('.pt')]
            checkpoints = args.ssl
            weights = torch.load(checkpoints, map_location='cpu')['weights']
            # weights = torch.load(os.path.join(ssl_path, checkpoints[idx]), map_location='cpu')['weights']
            weights = {k.replace('encoder.', ''):v for k, v in weights.items() if 'encoder' in k}

            model.load_state_dict(weights, strict=False)
    elif args.model_name == 'XSA':
        Net = XSA.XSA_Net
        model = Net(backbone="ResNet3D", encoder_layer=18, pretrain=True if args.ssl else False, kargs=args)
        ## Add option for Dropout to backbone latter

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.n_gpu))).to(device)
    else:
        model = model.to(device)

    # Create datasets
    train_dataset    = XSADataset(data_dir=args.data_dir, split='train', clip_len=args.clip_len,
                                   sampling_rate=args.sampling_rate, num_clips=args.num_clips, augment=args.augment,
                                   frac=args.frac, kinetics=(args.ssl == '') and (not args.rand_init))
    # Use training scaler for all splits to avoid data leakage from test set statistics
    train_scaler = train_dataset.scaler
    val_dataset      = XSADataset(data_dir=args.data_dir, split='val', clip_len=args.clip_len,
                                   sampling_rate=args.sampling_rate, num_clips=args.num_clips,
                                   kinetics=(args.ssl == '') and (not args.rand_init), scaler=train_scaler)
    test_dataset1     = XSADataset(data_dir=args.data_dir, split='test1', clip_len=args.clip_len, sampling_rate=args.sampling_rate, num_clips=args.num_clips, n_TTA=args.n_TTA, kinetics=(args.ssl == '') and (not args.rand_init), scaler=train_scaler)
    test_dataset2     = XSADataset(data_dir=args.data_dir, split='test2', clip_len=args.clip_len, sampling_rate=args.sampling_rate, num_clips=args.num_clips, n_TTA=args.n_TTA, kinetics=(args.ssl == '') and (not args.rand_init), scaler=train_scaler)
    # Create loaders
    train_loader    = torch.utils.data.DataLoader(train_dataset, batch_size=args.n_gpu*args.batch_size, shuffle=True,
                                                  # num_workers=4, worker_init_fn=seed_worker)
                                                  num_workers=4)
    val_loader      = torch.utils.data.DataLoader(val_dataset, batch_size=args.n_gpu*args.batch_size, shuffle=False, num_workers=4)
    test_loader1     = torch.utils.data.DataLoader(test_dataset1, batch_size=args.n_gpu*args.batch_size, shuffle=False, num_workers=4)
    test_loader2     = torch.utils.data.DataLoader(test_dataset2, batch_size=args.n_gpu*args.batch_size, shuffle=False, num_workers=4)

    # Set class weights
    if args.use_class_weights:
        class_weights = compute_class_weight(class_weight='balanced',
                                             classes=np.sort(np.unique((train_dataset.label_df['Recur'] == 1).values)),
                                             y=(train_dataset.label_df['Recur'] == 0).values)
        
        print('Class weights:', class_weights)
        
        pos_weight = (class_weights / class_weights.min())[1]

        print('Normalized positive class weight:', pos_weight)

        loss_fxn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([pos_weight]).to(device))
        eval_loss_fxn = torch.nn.BCELoss(weight=torch.Tensor([pos_weight]).to(device))
    elif args.use_cox_loss:
        loss_fxn = CoxLoss()
        eval_loss_fxn = CoxLoss()
    else:
        loss_fxn = torch.nn.BCEWithLogitsLoss()
        eval_loss_fxn = torch.nn.BCELoss()

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.eval_only == '':
        # Create csv documenting training history
        # history = pd.DataFrame(columns=['epoch', 'phase', 'loss', 'auroc', 'aupr', 'acc', 'b_acc', 'mcc', 'precision', 'recall', 'f1'])
        history = pd.DataFrame(columns=['epoch', 'phase', 'loss', 'cindex'])
        history.to_csv(os.path.join(model_dir, 'history.csv'), index=False)

        # Train with early stopping (only use validation set for model selection — no test set peeking)
        epoch = 1
        early_stopping_dict = {'best_cindex': 0.65, 'epochs_no_improve': 0,'best_epoch':0}
        best_model_wts = None
        while epoch <= args.epochs and early_stopping_dict['epochs_no_improve'] <= args.patience:
            history = train(model=model, device=device, loss_fxn=loss_fxn, optimizer=optimizer, data_loader=train_loader,
                            history=history, epoch=epoch, model_dir=model_dir)
            history, early_stopping_dict, best_model_wts, cidx_val = validate(model=model, device=device, loss_fxn=eval_loss_fxn,
                                                                    optimizer=optimizer, data_loader=val_loader, history=history,
                                                                    epoch=epoch, model_dir=model_dir,
                                                                    early_stopping_dict=early_stopping_dict,
                                                                    best_model_wts=best_model_wts, weight_saving=args.save_weights, split='val')
            print(f'Epoch {epoch} | val c-index: {cidx_val:.5f} | best val c-index: {early_stopping_dict["best_cindex"]:.5f} @ epoch {early_stopping_dict["best_epoch"]}')
            epoch += 1

        if args.weight_averaging:
            checkpoints = [torch.load(os.path.join(model_dir, f), map_location='cpu')['weights'] for f in os.listdir(model_dir) if f.endswith('.pt')]
            for key in best_model_wts:
                best_model_wts[key] = torch.stack([chkpt[key] for chkpt in checkpoints], dim=0).sum(dim=0) / len(checkpoints)
    else:
        print('Inferencing')
        eval_only_path = os.path.join(args.output_dir,args.eval_only)
        # history = pd.read_csv(os.path.join(eval_only_path, 'history.csv'))
        # checkpoints = [f for f in os.listdir(eval_only_path) if f.endswith('.pt')]
        # idx = np.argmax([int(f.split('.')[0].split('-')[1]) for f in checkpoints])
        # best_model_wts = torch.load(os.path.join(eval_only_path, checkpoints[idx]), map_location='cpu')['weights']

        best_model_wts = torch.load(os.path.join(eval_only_path,args.eval_model), map_location='cpu')['weights']
        # evaluate(model=model, device=device, loss_fxn=eval_loss_fxn, data_loader=test_loader, split=args.eval_center,
        # classes=test_dataset.CLASSES, history=history, model_dir=eval_only_path, weights=best_model_wts)

        ## Change it back to eval_only_path, we currently have no space on /data
        temp_savepath = '/home/lxq/paper/' \
                        'LTP-Postoperative Image only'
        if not os.path.exists(temp_savepath):
            os.mkdir(temp_savepath)


        printscores(model=model, device=device, loss_fxn=eval_loss_fxn, data_loader=train_loader, split='train', model_dir=temp_savepath, weights=best_model_wts)
        printscores(model=model, device=device, loss_fxn=eval_loss_fxn, data_loader=val_loader, split='val', model_dir=temp_savepath, weights=best_model_wts)
        printscores(model=model, device=device, loss_fxn=eval_loss_fxn, data_loader=test_loader1, split='test1', model_dir=temp_savepath, weights=best_model_wts)
        printscores(model=model, device=device, loss_fxn=eval_loss_fxn, data_loader=test_loader2, split='test2', model_dir=temp_savepath, weights=best_model_wts)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/data/smart/lxq/mmdl/allData/')
    # parser.add_argument('--output_dir', type=str, default='/home/lxq/data/finetune_results')
    parser.add_argument('--output_dir', type=str, default='/data/smart/lxq/mmdl/Paper/')
    parser.add_argument('--clip_len', type=int, default=4)
    parser.add_argument('--sampling_rate', type=int, default=1)
    parser.add_argument('--num_clips', type=int, default=1)
    parser.add_argument('--label_smoothing', type=float, default=0.)
    parser.add_argument('--use_class_weights', action='store_true', default=False)
    parser.add_argument('--use_cox_loss', action='store_true', default=True)
    parser.add_argument('--augment', action='store_true', default=False)
    parser.add_argument('--lpft', type=str, default='')
    parser.add_argument('--save_weights', type=str, default=True)
    parser.add_argument('--frac', type=float, default=1.0)
    parser.add_argument('--label', type=str, default='DIM', choices=['DIM', 'lvh'])
    parser.add_argument('--weight_averaging', action='store_true', default=False)
    parser.add_argument('--n_TTA', type=int, default=0)
    parser.add_argument('--rand_init', action='store_true', default=True)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--weight_path', type=str, default='')
    # parser.add_argument('--prefix_name', type=str, default="ResNet3D")
    parser.add_argument('--a', type=str, default='')
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--flush_history', type=int, choices=[0, 1], default=0)
    parser.add_argument('--save_model', type=int, choices=[0, 1], default=0)
    # parser.add_argument('--patience', type=int, default=500)
    # parser.add_argument('--log_every', type=int, default=100)
    parser.add_argument('--gpu', type=str, default=1)
    parser.add_argument('--half', action='store_true', default=False)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--DiseaseList', type=list, default=['Risk_Score'])
    parser.add_argument('--ViewList', type=list, default=['Sag', 'Cor', 'Axi'])
    # parser.add_argument('--SequenceList', type=list, default=["PreT1", "PreT2", "PreDWI", "premask"])
    parser.add_argument('--SequenceList', type=list, default=["dwipre", "t2pre", "t1pre", "premask","t1post","t1tumourpost"])
    parser.add_argument('--ClassNum', type=int, default=1)

    parser.add_argument('--backbone', type=str, default="ResNet3D")
    parser.add_argument('--model_depth', type=int, default=18)
    parser.add_argument('--no_co_att', action='store_true', default=False)
    parser.add_argument('--no_cross_modal', action='store_true', default=False)
    parser.add_argument('--no_post_img', action='store_true', default=False)
    parser.add_argument('--post_img_only', action='store_true', default=True)
    parser.add_argument('--no_clic', action='store_true', default=False)
    parser.add_argument('--encoder_only', action='store_true', default=False)
    parser.add_argument('--separate_final', action='store_true', default=False)

    parser.add_argument('--emb_dim', type=int, default=512)
    parser.add_argument('--emb_num', type=int, default=14)
    # parser.add_argument('--emb_dim', type=int, default=512)
    # parser.add_argument('--emb_num', type=int, default=28)
    parser.add_argument('--alpha', type=float, default=0.1)

    # analysis args
    parser.add_argument('--write_metrix', type=bool, default=False)
    parser.add_argument('--show_patch_sample', type=bool, default=False)


    parser.add_argument('--model_name', type=str, default='XSA')
    # parser.add_argument('--model_name', type=str, default='res')
    # parser.add_argument('--exp_name', type=str, default='-recur-exp1016-pre-plain')
    parser.add_argument('--exp_name', type=str, default='-Recur-XPT Encoders-sa-Postoperative Image only')
    # parser.add_argument('--XSApretrain', action='store_true', default=True) ## For XSS-XSA
    parser.add_argument('--dropout_fc', action='store_true', default=False) ## For ResNet dropping out at the end
    parser.add_argument('--n_gpu', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--patience', type=int, default=1e4)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--batch_size', type=int, default=64)
    ## Run the following 2 lines for eval only
    parser.add_argument('--eval_only', type=str, default='/data/smart/lxq/mmdl/Paper/XSA-LTP-XPT Encoders-sa-Postoperative Image only_lr-3e-05_300ep_bs-64_seed-388/')
    parser.add_argument('--eval_model', type=str, default='chkpt_epoch-94.pt')
    # parser.add_argument('--eval_only', type=str, default='') ## '' enter path

    # parser.add_argument('--eval_center', type=str, default='test1') ## val/test1/test2
    parser.add_argument('--ssl', type=str, default='') ## For e2e resnet ssl
    # parser.add_argument('--ssl', type=str, default='/data/smart/lxq/mmdl/sslmodels/Exp_bz-64_'
    #                                                'proj-128_lr-0.01shuff-True/'
    #                                                'chkpt_epoch-300.pt') ## For e2e resnet ssl
    parser.add_argument('--seed', type=int, default=388)
    parser.add_argument('--active_branch', type=list, default=[0, 0, 1])

    args = parser.parse_args()

    print(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    main(args)