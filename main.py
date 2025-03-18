import torch
import pandas as pd
import os
import pickle
from torch.optim import Adam
from args import get_args
from utils import set_seed
from model import GUET
from train import train_GUET, train_GUET_
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_pkl(parent_dir, filename):
    file_path = os.path.join(parent_dir, filename)

    with open(file_path, 'rb') as file:
        loaded_data = pickle.load(file)
    return loaded_data


def main():
    args = get_args()
    set_seed(args.seed)
    is_tenfold = 'true'
    sim_set = load_pkl(args.parent_dir, 'sim_set.pkl') # obtain SNF views information
    SVD_NMF = load_pkl(args.parent_dir, 'SVD_NMF.pkl') # obtain embedding from SVD and NMF
    # obtain embedding of miRNA and disease based on SVD

    print('--------------------------------Data organized and ready to build the model-------------------')
    model = GUET()
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.we_decay)
    Acc = []
    Pre = []
    Rec = []
    Spe = []
    Mcc = []
    F1 = []
    AUROC = []
    AUPR = []
    if is_tenfold == 'true':
            pair_pos_neg_fengceng = load_pkl(args.parent_dir_, f'pos_neg_pair_10_1.pkl')  # Loading positive and negative samples for ten-fold training and testing
            # pair_pos_neg_fengceng = load_pkl(args.parent_dir_, f'pos_neg_pair_stratified_balanced.pkl')
            accuracy, precision, recall, specificity, mcc, f1, auc_, aupr_ = train_GUET(args, model, sim_set, SVD_NMF, optimizer, pair_pos_neg_fengceng, device)
            Acc.append(accuracy)
            Pre.append(precision)
            Rec.append(recall)
            Spe.append(specificity)
            Mcc.append(mcc)
            F1.append(f1)
            AUROC.append(auc_)
            AUPR.append(aupr_)
    else:
        pair_pos_neg_fengceng = load_pkl(args.parent_dir_, 'case.pkl')
        accuracy, precision, recall, specificity, mcc, f1, auc_, aupr_ =  train_GUET(args, model, sim_set, SVD_NMF, optimizer,pair_pos_neg_fengceng, device)
        Acc.append(accuracy)
        Pre.append(precision)
        Rec.append(recall)
        Spe.append(specificity)
        Mcc.append(mcc)
        F1.append(f1)
        AUROC.append(auc_)
        AUPR.append(aupr_)
    print('--------------------------------- print metric ----------------------------------')
    for acc in Acc:
        print(acc)
    print(f'avg ACC:{sum(Acc) / len(Acc)}')
    for pre in Pre:
        print(pre)
    print(f'avg Pre:{sum(Pre) / len(Pre)}')
    for rec in Rec:
        print(rec)
    print(f'avg Rec:{sum(Rec) / len(Rec)}')
    for spe in Spe:
        print(spe)
    print(f'avg Spe:{sum(Spe) / len(Spe)}')
    for mcc in Mcc:
        print(mcc)
    print(f'avg Mcc:{sum(Mcc) / len(Mcc)}')
    for f1 in F1:
        print(f1)
    print(f'avg F1:{sum(F1) / len(F1)}')
    for auc in AUROC:
        print(auc)
    print(f'avg AUROC:{sum(AUROC) / len(AUROC)}')
    for aupr in AUPR:
        print(aupr)
    print(f'avg AUPR:{sum(AUPR) / len(AUPR)}')
    print('----------------------------------- ending ----------------------------------')
if __name__ == "__main__":
    main()