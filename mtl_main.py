import numpy as np
import torch, argparse
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from torch.nn import BCELoss
from torch_geometric.data import DataLoader

from Dataset import MolNet
from k_fold_data import load_fold_data
from multi_task import Multitask
from utils import get_logger, metrics_c, set_seed

import warnings

warnings.filterwarnings("ignore")

_use_shared_memory = True
torch.backends.cudnn.benchmark = True


def training(model, bs_train_loader, rt_train_loader, fhm_train_loader, shm_train_loader, optimizer, loss_f, metric,
             task, device, mean,
             stds):
    loss_record, record_count = 0., 0.
    model.train()
    if task == 'clas':

        # 在每个任务的 DataLoader 上进行循环
        for data_task1, data_task2, data_task3, data_task4 in zip(bs_train_loader, rt_train_loader, fhm_train_loader,
                                                                  shm_train_loader):
            # 分别获取每个任务的数据和标签
            data_bs = data_task1.to(device)
            data_rt = data_task2.to(device)
            data_fhm = data_task3.to(device)
            data_shm = data_task4.to(device)

            labels_bs = data_task1.y.to(device)
            labels_rt = data_task2.y.to(device)
            labels_fhm = data_task3.y.to(device)
            labels_shm = data_task4.y.to(device)

            # 传递数据到模型：
            output_bs, output_rt, output_fhm, output_shm = model(data_bs, data_rt, data_fhm, data_shm)

            # 计算损失和反向传播等步骤也需要分别针对每个任务执行
            loss_bs = loss_f(output_bs.squeeze(), labels_bs.squeeze())
            loss_rt = loss_f(output_rt.squeeze(), labels_rt.squeeze())
            loss_fhm = loss_f(output_fhm.squeeze(), labels_fhm.squeeze())
            loss_shm = loss_f(output_shm.squeeze(), labels_shm.squeeze())

            loss = loss_bs + loss_rt + loss_fhm + loss_shm
            loss_record += float(loss.item())
            record_count += 1
            optimizer.zero_grad()

            loss_bs.backward()
            loss_rt.backward()
            loss_fhm.backward()
            loss_shm.backward()

            nn.utils.clip_grad_value_(model.parameters(), clip_value=2)
            optimizer.step()
        epoch_loss = loss_record / record_count * 4
        return epoch_loss


def testing(model, bs_valid_loader, rt_valid_loader, fhm_valid_loader, shm_valid_loader, loss_f, metric, task, device,
            mean, stds, resu):
    model.eval()  # 切换为评估模式

    loss_record, record_count = 0., 0.

    bs_preds = torch.Tensor([])
    bs_tars = torch.Tensor([])

    rt_preds = torch.Tensor([])
    rt_tars = torch.Tensor([])

    fhm_preds = torch.Tensor([])
    fhm_tars = torch.Tensor([])

    shm_preds = torch.Tensor([])
    shm_tars = torch.Tensor([])
    with torch.no_grad():
        for data_task1, data_task2, data_task3, data_task4 in zip(bs_valid_loader, rt_valid_loader, fhm_valid_loader,
                                                                  shm_valid_loader):
            # 获取当前任务的验证 DataLoader
            data_bs = data_task1.to(device)
            data_rt = data_task2.to(device)
            data_fhm = data_task3.to(device)
            data_shm = data_task4.to(device)

            labels_bs = data_task1.y.to(device)
            labels_rt = data_task2.y.to(device)
            labels_fhm = data_task3.y.to(device)
            labels_shm = data_task4.y.to(device)

            output_bs, output_rt, output_fhm, output_shm = model(data_bs, data_rt, data_fhm, data_shm)

            loss_bs = loss_f(output_bs.squeeze(), labels_bs.squeeze())
            loss_rt = loss_f(output_rt.squeeze(), labels_rt.squeeze())
            loss_fhm = loss_f(output_fhm.squeeze(), labels_fhm.squeeze())
            loss_shm = loss_f(output_shm.squeeze(), labels_shm.squeeze())

            loss = loss_bs + loss_rt + loss_fhm + loss_shm
            loss_record += float(loss.item())
            record_count += 1

            bs_pre = output_bs.detach().cpu()
            rt_pre = output_rt.detach().cpu()
            fhm_pre = output_fhm.detach().cpu()
            shm_pre = output_shm.detach().cpu()

            bs_preds = torch.cat([bs_preds, bs_pre], 0);
            bs_tars = torch.cat([bs_tars, labels_bs.cpu()], 0)
            rt_preds = torch.cat([rt_preds, rt_pre], 0);
            rt_tars = torch.cat([rt_tars, labels_rt.cpu()], 0)
            fhm_preds = torch.cat([fhm_preds, fhm_pre], 0);
            fhm_tars = torch.cat([fhm_tars, labels_fhm.cpu()], 0)
            shm_preds = torch.cat([shm_preds, shm_pre], 0);
            shm_tars = torch.cat([shm_tars, labels_shm.cpu()], 0)
            bs_clas = bs_preds > 0.5
            rt_clas = rt_preds > 0.5
            fhm_clas = fhm_preds > 0.5
            shm_clas = shm_preds > 0.5

            bs_acc, bs_pre, bs_rec, bs_auc = metric(bs_clas.squeeze().numpy(), bs_preds.squeeze().numpy(),
                                                    bs_tars.squeeze().numpy())
            rt_acc, rt_pre, rt_rec, rt_auc = metric(rt_clas.squeeze().numpy(), rt_preds.squeeze().numpy(),
                                                    rt_tars.squeeze().numpy())
            fhm_acc, fhm_pre, fhm_rec, fhm_auc = metric(fhm_clas.squeeze().numpy(), fhm_preds.squeeze().numpy(),
                                                        fhm_tars.squeeze().numpy())
            shm_acc, shm_pre, shm_rec, shm_auc = metric(shm_clas.squeeze().numpy(), shm_preds.squeeze().numpy(),
                                                        shm_tars.squeeze().numpy())

    epoch_loss = loss_record / record_count
    return epoch_loss, bs_acc, bs_pre, bs_rec, bs_auc, rt_acc, rt_pre, rt_rec, rt_auc, fhm_acc, fhm_pre, fhm_rec, fhm_auc, shm_acc, shm_pre, shm_rec, shm_auc


def main(tasks, task, device, train_epoch, seed, fold, batch_size, rate, scaffold, logger, lr,
         attn_head, output_dim, attn_layers, dropout, mean, stds, D, met, savem,
         fp_type):
    global best_test_acc, bs_best_auc, bs_best_acc, rt_best_auc, rt_best_acc, fhm_best_auc, fhm_best_acc, shm_best_auc, shm_best_acc, bs_best_pre, bs_best_re, rt_best_pre, rt_best_re, fhm_best_pre, fhm_best_re, shm_best_pre, shm_best_re, i
    dataset = ['BS', 'RT', 'FHM', 'SHM']
    logger.info('Dataset: {}  task: {}  train_epoch: {}'.format(dataset, task, train_epoch))

    d_k, seed_ = round(output_dim / attn_head), seed

    fold_result = [[], [], [], [], [], [], [], []]
    fold_result1 = [[], [], [], [], [], [], [], []]
    if task == 'clas':
        loss_f = BCELoss().to(device)
        # loss_f = CapsuleLoss().to(device)
        metric = metrics_c(accuracy_score, precision_score, recall_score, roc_auc_score)
        for fol in range(fold):
            best_val_auc = 0.

            if seed is not None:
                # seed_ = seed + fol-1
                seed_ = seed
                set_seed(seed_)

            model = Multitask(task, tasks, attn_head, output_dim, d_k, d_k, attn_layers, D, dropout, 1.5, device,
                              fp_type).to(device)
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)

            bs_train_loader, bs_valid_loader = load_fold_data(fol, batch_size[0], 0, fold, dataset[0])
            rt_train_loader, rt_valid_loader = load_fold_data(fol, batch_size[1], 0, fold, dataset[1])
            fhm_train_loader, fhm_valid_loader = load_fold_data(fol, batch_size[2], 0, fold, dataset[2])
            shm_train_loader, shm_valid_loader = load_fold_data(fol, batch_size[3], 0, fold, dataset[3])

            logger.info('Dataset: {}  Fold: {:<4d}'.format(moldata, fol))
            logger.info('Dataset: {}'.format(moldata))

            for i in range(1, train_epoch + 1):
                train_loss = training(model, bs_train_loader, rt_train_loader, fhm_train_loader, shm_train_loader,
                                      optimizer, loss_f,
                                      metric, task, device, mean, stds)
                valid_loss, bs_acc, bs_pre, bs_rec, bs_auc, rt_acc, rt_pre, rt_rec, rt_auc, fhm_acc, fhm_pre, fhm_rec, \
                    fhm_auc, shm_acc, shm_pre, shm_rec, shm_auc = testing(model, bs_valid_loader, rt_valid_loader,
                                                                          fhm_valid_loader, shm_valid_loader, loss_f,
                                                                          metric, task, device, mean, stds, False)

                logger.info('Dataset: {}  Epoch: {:<3d}  train_loss: {:.4f}'.format(dataset, i, train_loss))
                logger.info('Dataset: {}  Epoch: {:<3d}  valid_loss: {:.4f}'.format(dataset, i, valid_loss))

                logger.info('Dataset: {}  Epoch: {:<3d}  bs_valid_auc: {:.4f} bs_valid_acc: {:.4f} '
                            'bs_valid_re: {:.4f} bs_valid_pre: {:.4f}'.format(dataset[0], i, bs_auc, bs_acc,
                                                                              bs_rec, bs_pre))

                logger.info('Dataset: {}  Epoch: {:<3d}  rt_valid_auc: {:.4f} rt_valid_acc: {:.4f} '
                            'rt_valid_re: {:.4f} rt_valid_pre: {:.4f}'.format(dataset[1], i, rt_auc, rt_acc,
                                                                              rt_rec, rt_pre))

                logger.info('Dataset: {}  Epoch: {:<3d} fhm_valid_auc: {:.4f} fhm_valid_acc: {:.4f} '
                            'fhm_valid_re: {:.4f} fhm_valid_pre: {:.4f}'.format(dataset[2], i, fhm_auc,
                                                                                fhm_acc, fhm_rec, fhm_pre))
                logger.info('Dataset: {}  Epoch: {:<3d} fhm_valid_auc: {:.4f} shm_valid_acc: {:.4f} '
                            'shm_valid_re: {:.4f} shm_valid_pre: {:.4f}'.format(dataset[3], i, shm_auc, shm_acc,
                                                                                shm_rec, shm_pre))
                logger.info('---------------------------------------------------------------------------------------')
                valid_auc = bs_auc + rt_auc + fhm_auc + shm_auc
                if valid_auc > best_val_auc:

                    best_val_auc = valid_auc
                    bs_best_acc = bs_acc;
                    bs_best_auc = bs_auc;
                    bs_best_pre = bs_pre;
                    bs_best_re = bs_rec;
                    rt_best_acc = rt_acc;
                    rt_best_auc = rt_auc;
                    rt_best_pre = rt_pre;
                    rt_best_re = rt_rec
                    fhm_best_acc = fhm_acc;
                    fhm_best_auc = fhm_auc;
                    fhm_best_pre = fhm_pre;
                    fhm_best_re = fhm_rec
                    shm_best_acc = shm_acc;
                    shm_best_auc = shm_auc;
                    shm_best_pre = shm_pre;
                    shm_best_re = shm_rec

                    torch.save(model.state_dict(), 'model_perform/fol{}.pkl'.format(fol))
            fold_result[0].append(bs_best_auc);
            fold_result[1].append(bs_best_acc)
            fold_result[2].append(rt_best_auc);
            fold_result[3].append(rt_best_acc)
            fold_result[4].append(fhm_best_auc);
            fold_result[5].append(fhm_best_acc)
            fold_result[6].append(shm_best_auc);
            fold_result[7].append(shm_best_acc)

            fold_result1[0].append(bs_best_pre)
            fold_result1[1].append(bs_best_re)
            fold_result1[2].append(rt_best_pre)
            fold_result1[3].append(rt_best_re)
            fold_result1[4].append(fhm_best_pre)
            fold_result1[5].append(fhm_best_re)
            fold_result1[6].append(shm_best_pre)
            fold_result1[7].append(shm_best_re)

            logger.info(
                'Dataset: {} best_val_auc: {:.4f} best_val_acc: {:.4f} best_val_re: {:.4f} best_val_pre: {:.4f}'.format(
                    dataset[0], bs_best_auc, bs_best_acc, bs_best_re, bs_best_pre))
            logger.info(
                'Dataset: {} best_val_auc: {:.4f} best_val_acc: {:.4f} best_val_re: {:.4f} best_val_pre: {:.4f}'.format(
                    dataset[1], rt_best_auc, rt_best_acc, rt_best_re, rt_best_pre))
            logger.info(
                'Dataset: {} best_val_auc: {:.4f} best_val_acc: {:.4f} best_val_re: {:.4f} best_val_pre: {:.4f}'.format(
                    dataset[2], fhm_best_auc, fhm_best_acc, fhm_best_re, fhm_best_pre))
            logger.info(
                'Dataset: {} best_val_auc: {:.4f} best_val_acc: {:.4f} best_val_re: {:.4f} best_val_pre: {:.4f}'.format(
                    dataset[3], shm_best_auc, shm_best_acc, shm_best_re, shm_best_pre))
        # logger.info('Dataset: {} Fold result: {}'.format(dataset, fold_result, fold_result1))
    return fold_result, fold_result1
def test_mol(tasks, task, dataset, device, train_epoch, seed, fold, batch_size, rate, scaffold, modelpath, logger, lr, attn_head, output_dim, attn_layers, dropout, mean, stds, D, met, savem,
         fp_type):
    logger.info('Dataset: {}  task: {}  testing:'.format(dataset, task))
    d_k = round(output_dim/attn_head)
    if seed is not None:
        set_seed(seed)
    model = Multitask(task, tasks, attn_head, output_dim, d_k, d_k, attn_layers, D, dropout, 1.5, device, fp_type).to(device)
    state_dict = torch.load('E:/final_code2/single_task/model_/fol4.pkl')
    model.load_state_dict(state_dict)

    data = MolNet(root='./dataset', dataset=dataset)
    loader = DataLoader(data, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0, drop_last=False)
    if task == 'clas':
        loss_f = BCELoss().to(device)
        metric = metrics_c(accuracy_score, precision_score, recall_score, roc_auc_score)
        loss, acc, pre,rec,auc = testing(model, loader, loss_f, metric, task, device, mean, stds, False)
        logger.info('Dataset: {}  test_loss: {:.4f}  test_acc: {:.4f}  test_f1: {:.4f}  test_auc: {:.4f}  test_pre: {:.4f}  test_rec: {:.4f}'.format(dataset, loss, acc, auc, pre, rec))
        results = {
            'test_loss': loss,
            'test_acc': acc,
            'test_pre': pre,
            'test_rec': rec,
            'test_auc': auc,
        }
        np.save('log/Result' + moldata + '_test.npy', results, allow_pickle=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TransFoxMol')
    parser.add_argument('--mode', default='train', type=str, choices=['train'],
                        help='train, test or hyperparameter_search')
    parser.add_argument('--moldata', default='ALL', type=str, help='Dataset name')
    parser.add_argument('--task', default='clas', type=str, choices=['clas'], help='Classification or Regression')
    parser.add_argument('--device', type=str, default='cuda:0', help='Which gpu to use if any (default: cuda:0)')
    parser.add_argument('--batch_size', type=int, default=[87, 120, 90, 33], help='Input batch size for training')
    parser.add_argument('--train_epoch', type=int, default=30, help='Number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
    parser.add_argument('--valrate', type=float, default=0.2, help='valid rate (default: 0.1)')
    parser.add_argument('--testrate', type=float, default=0, help='test rate (default: 0.1)')
    parser.add_argument('--fold', type=int, default=5, help='Number of folds for cross validation (default: 3)')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout ratio')
    parser.add_argument('--scaffold', type=str, default=True,
                        help="True: random scaffold dataset split; False: random dataset split (default: True)")
    parser.add_argument('--attn_head', type=int, default=4, help="Number of attention heads for transformer")
    parser.add_argument('--attn_layers', type=int, default=2, help='Number of GNN message passing layers')
    parser.add_argument('--output_dim', type=int, default=256, help='Hidden size of embedding layer')
    parser.add_argument('--D', type=int, default=4, help='Hidden size of readout layer')
    parser.add_argument('--seed', type=int,default=42, help="Seed for splitting the dataset")
    parser.add_argument('--metric', type=str, choices=['rmse', 'mae'],
                        help='Metric to evaluate the regression performance')
    args = parser.parse_args()

    device = torch.device(args.device)
    moldata = args.moldata
    rate = [args.valrate, args.testrate]

    if moldata in ['ALL']:
        task = 'clas'
        mean, std = None, None

        if moldata == 'mtl_fish3':
            numtasks = 4
            dataset = ['BS', 'RT', 'FHM', 'SHM']

    logf = 'log/{}_{}_{}.log'.format(moldata, args.task, args.mode)
    logger = get_logger(logf)

    # moldata += task
    fp_type = ['morgan', 'maccs', 'rdit', 'apc2d', 'ecfp']

    if args.mode == 'train':
        logger.info('Training')
        num_tasks = 4
        fold_result, fold_result1 = main(num_tasks, task, device, args.train_epoch, args.seed, args.fold,
                                         args.batch_size,
                                         rate, args.scaffold, logger, args.lr, args.attn_head,
                                         args.output_dim,
                                         args.attn_layers, args.dropout, mean, std, args.D, args.metric, True, fp_type)
        print('----------------------------------------------------')
        bs_ava_auc = sum(fold_result[0]) / len(fold_result[0])
        bs_ava_acc = sum(fold_result[1]) / len(fold_result[1])
        bs_ava_pre = sum(fold_result1[0]) / len(fold_result1[0])
        bs_ava_re = sum(fold_result1[1]) / len(fold_result1[1])

        rt_ava_auc = sum(fold_result[2]) / len(fold_result[2])
        rt_ava_acc = sum(fold_result[3]) / len(fold_result[3])
        rt_ava_pre = sum(fold_result1[2]) / len(fold_result1[2])
        rt_ava_re = sum(fold_result1[3]) / len(fold_result1[3])

        fhm_ava_auc = sum(fold_result[4]) / len(fold_result[4])
        fhm_ava_acc = sum(fold_result[5]) / len(fold_result[5])
        fhm_ava_pre = sum(fold_result1[4]) / len(fold_result1[4])
        fhm_ava_re = sum(fold_result1[5]) / len(fold_result1[5])

        shm_ava_auc = sum(fold_result[6]) / len(fold_result[6])
        shm_ava_acc = sum(fold_result[7]) / len(fold_result[7])
        shm_ava_pre = sum(fold_result1[6]) / len(fold_result1[6])
        shm_ava_re = sum(fold_result1[7]) / len(fold_result1[7])
        logger.info('-------------------------------------------------------------------------------------------------')
        logger.info('bs_ava_auc:{:.4f} bs_ava_acc:{:.4f} bs_ava_pre:{:.4f} bs_ava_re:{:.4f}'.format(bs_ava_auc, bs_ava_acc,
                                                                                              bs_ava_re, bs_ava_pre))
        logger.info('rt_ava_auc:{:.4f} rt_ava_acc:{:.4f} rt_ava_pre:{:.4f} rt_ava_re:{:.4f}'.format(rt_ava_auc, rt_ava_acc,
                                                                                              rt_ava_re, rt_ava_pre))
        logger.info('fhm_ava_auc:{:.4f} fhm_ava_acc:{:.4f} fhm_ava_pre:{:.4f} fhm_ava_re:{:.4f}'.format(fhm_ava_auc,
                                                                                                  fhm_ava_acc,
                                                                                                  fhm_ava_re,
                                                                                                  fhm_ava_pre))
        logger.info('shm_ava_auc:{:.4f} shm_ava_acc:{:.4f} shm_ava_pre:{:.4f} shm_ava_re:{:.4f}'.format(shm_ava_auc,
                                                                                                  shm_ava_acc,
                                                                                                  shm_ava_re,
                                                                                                  shm_ava_pre))
