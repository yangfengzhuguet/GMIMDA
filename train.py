import pickle

import numpy as np
import torch.nn
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve,precision_recall_curve,auc,precision_recall_fscore_support,matthews_corrcoef,accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import average_precision_score
import torch.nn.functional as F
import pandas as pd
from matplotlib.colors import ListedColormap
import tqdm
import time
import os

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def show_auc(pre_score, label, flag):
    y_true = label.flatten().detach().cpu().numpy()
    y_score = pre_score.flatten().detach().cpu().numpy()
    fpr,tpr,rocth = roc_curve(y_true,y_score)
    auroc = auc(fpr,tpr)
    precision,recall,prth = precision_recall_curve(y_true,y_score)
    aupr = auc(recall,precision)
    # if flag == 'test':
    #     # Plot the roc curve and save it
    #     plt.figure()
    #     plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auroc:.4f})')
    #     plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    #     plt.xlim([0.0, 1.0])
    #     plt.ylim([0.0, 1.05])
    #     plt.xlabel('False Positive Rate')
    #     plt.ylabel('True Positive Rate')
    #     plt.title('Receiver Operating Characteristic')
    #     plt.legend(loc='lower right')
    #
    #     # Save the curve to the specified directory
    #     # output_dir = 'result/HMDD v3-2-wuguangdui/10-fold_'
    #     output_dir = 'result/HMDD V4/fenceng'
    #     os.makedirs(output_dir, exist_ok=True)
    #     # plt.savefig(os.path.join(output_dir, '10-fold_9_test_auc.png'))
    #     plt.savefig(os.path.join(output_dir, 'fenceng.png'))
    return auroc, aupr, fpr, tpr

def train_GUET(arges, model, sim_set, SVD_NMF, optimizer, pair_pos_neg_fengceng, device):
    model = model.to(device)
    train_miRNA_index_fc = torch.tensor(pair_pos_neg_fengceng['train'][:,0], dtype=torch.long).to(device) # miRNA index for train
    train_disease_index_fc = torch.tensor(pair_pos_neg_fengceng['train'][:,1],dtype=torch.long).to(device) # disease index for train
    train_label_fc = torch.tensor(pair_pos_neg_fengceng['train'][:,2]).to(device).float() # label for train


    test_miRNA_index_fc = torch.tensor(pair_pos_neg_fengceng['test'][:, 0],dtype=torch.long).to(device)  # miRNA index for test
    test_disease_index_fc = torch.tensor(pair_pos_neg_fengceng['test'][:,1],dtype=torch.long).to(device)  # disease index for test
    test_label_fc = torch.tensor(pair_pos_neg_fengceng['test'][:,2]).to(device).float()  # label for test

    # # case study
    # train_miRNA_index_fc = torch.tensor(pair_pos_neg_fengceng[:,0].astype(float), dtype=torch.long).to(device) # miRNA index for train
    # train_disease_index_fc = torch.tensor(pair_pos_neg_fengceng[:,1].astype(float),dtype=torch.long).to(device) # disease index for train
    # mda = pd.read_csv('data/HMDD v2.0/association_matrix_.csv', header=None, sep=',')
    # mda = mda.values
    # train_label_fc = torch.tensor(mda).to(device).float()

    loss_min = float('inf')
    best_auc = 0
    best_aupr = 0
    print('######################### start to training #############################')
    for epoch_ in tqdm.tqdm(range(arges.epoch), desc='Training Epochs'):
        time_start = time.time()
        model.train()
        train_score, nash_loss, p = model(sim_set, SVD_NMF, train_miRNA_index_fc, train_disease_index_fc)
        # p_ = p.detach()
        # long_label = train_label_fc.to(torch.long)
        # dataload = {'train_input': p_, 'train_label': long_label}
        # parent_dir = 'data'
        # filename = 'dataload.pkl'
        # file_path = os.path.join(parent_dir, filename)
        # os.makedirs(parent_dir, exist_ok=True)
        # with open(file_path, 'wb') as file:
        #     pickle.dump(dataload, file)
        # print('good')
        # train_score = model(sim_set, SVD_NMF, train_miRNA_index_fc, train_disease_index_fc)
        loss1 = torch.nn.BCELoss()
        # loss_train = F.binary_cross_entropy(train_score, train_label_fc)
        loss_train = loss1(train_score, train_label_fc)
        loss = 0.5 * loss_train + 0.5 * nash_loss
        # loss = loss_train
        auc_, aupr, f, t = show_auc(train_score, train_label_fc, 'train')
        if loss_train < loss_min:
            loss_min = loss_train
        if auc_ > best_auc:
            best_auc = auc_
        if aupr > best_aupr:
            best_aupr = aupr
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        time_end = time.time()
        time_epoch = time_end - time_start
        # if epoch_ in[0, 29, 69]:
        #    plt.rcParams.update({'font.size': 20})
        #    fig, ax = plt.subplots(figsize=(3, 2))
        #    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
        #    y = train_label_fc.flatten().detach().cpu().numpy()
        #    features = p
        #    features = features.detach().cpu().numpy()
        #    low_feature = tsne.fit_transform(features)
        #    plt.figure(figsize=(12, 10), dpi=600)
        #    cmap_custom = ListedColormap(['blue', 'red'])
        #    scatter = plt.scatter(
        #        low_feature[:, 0], low_feature[:, 1],
        #        c=y, cmap=cmap_custom, alpha=0.8,
        #        edgecolors='w', linewidths=0.5
        #    )
        #    plt.legend(*scatter.legend_elements(), title="Classes",
        #               loc='upper right', bbox_to_anchor=(1.15, 1))
        #    plt.title(f'epoch:{epoch_+1}', fontsize=20, y=-0.12)
        #    # plt.xlabel('t-SNE Dimension 1', fontsize=20)
        #    # plt.ylabel('t-SNE Dimension 2', fontsize=20)
        #
        #    plt.savefig(f"data/tsne1/hmdd_{epoch_+1}.png", dpi=600, bbox_inches='tight', pad_inches=0.1)
        #    plt.close()

        # print('-------Time when the epoch runs£º{} seconds ----------'.format(time_epoch))
        print('-------The train: epoch{}, Loss:{}, AUC:{}, AUPR{}-------------'.format(epoch_+1, loss_train, auc_, aupr))
    print('The loss_min:{}, best auc{}, best aupr{}'.format(loss_min, best_auc, best_aupr))
    print('######################### start to testing #############################')
    model.eval()
    with (torch.no_grad()):
        test_score, nash_loss, p = model(sim_set, SVD_NMF, test_miRNA_index_fc, test_disease_index_fc)
        # test_score = model(sim_set, SVD_NMF, test_miRNA_index_fc, test_disease_index_fc)
        loss2 = torch.nn.BCELoss()
        loss_test = loss2(test_score, test_label_fc)
        loss_ = 0.5 * loss_test + 0.5 * nash_loss
        # loss_ = loss_test
        auc_, aupr_, fp, tp = show_auc(test_score, test_label_fc, 'test')
        # np.save(f'data/HMDD v2.0/f_tpr/fpr_1.npy', fp)
        # np.save(f'data/HMDD v2.0/f_tpr/tpr_1.npy', tp)
        print('-------The test: Loss:{}, AUC:{}, AUPR{}-----------'.format(loss_, auc_, aupr_))
        # test_score, payoff = model(sim_set, SVD_NMF, train_miRNA_index_fc, train_disease_index_fc)
        # av_payoff = torch.mean(payoff)
        # # loss2 = torch.nn.BCELoss()
        # # loss_test = loss2(test_score, train_miRNA_index_fc)
        # loss_test = F.binary_cross_entropy(test_score, train_label_fc)
        # loss_ = 0.5 * loss_test + 0.5 * av_payoff
        # auc_, aupr_ = show_auc(test_score, train_label_fc, 'test')
        # print('-------The test: Loss:{}, AUC:{}, AUPR{}-----------'.format(loss_, auc_, aupr_))
        # data = test_score.cpu().numpy()
        # df = pd.DataFrame(data)
        # df.to_csv('case.csv', header=False, index=False)
    #
        # Setting image resolution and fonts
        plt.rcParams['figure.dpi'] = 600
        font1 = {"family": "Arial", "weight": "book", "size": 9}

        # sample data
        y_true = np.array(test_label_fc.detach().cpu())
        y_true = np.where(y_true == 1, True, False)
        y_scores = np.array(test_score.detach().cpu())
        # Calculate the ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc_ = auc(fpr, tpr)
        roc_auc_ = round(roc_auc_, 3)


        # Calculating Precision-Recall Curves
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_scores)
        aupr = auc(recall, precision)
        aupr = round(aupr, 3)
        # Calculate the performance metrics at different thresholds and find the optimal thresholds
        best_threshold = 0.0
        best_f1 = 0.0
        best_metrics = {}

        # Sensitivity and specificity at each threshold are preserved
        sensitivities = []
        specificities = []

        for threshold in thresholds:
            y_pred = (y_scores >= threshold).astype(int)
            precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
            accuracy = (y_pred == y_true).mean()
            mcc = matthews_corrcoef(y_true, y_pred)
            tn = ((y_pred == 0) & (y_true == 0)).sum()
            fp = ((y_pred == 1) & (y_true == 0)).sum()
            fn = ((y_pred == 0) & (y_true == 1)).sum()
            tp = ((y_pred == 1) & (y_true == 1)).sum()
            specificity = tn / (tn + fp)

            sensitivities.append(recall)
            specificities.append(specificity)

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
                best_metrics = {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "mcc": mcc,
                    "specificity": specificity
                }

        # Plotting the ROC curve
        plt.figure(1)
        plt.plot(fpr, tpr, label=f"AUROC={roc_auc_}")
        plt.xlabel('False Positive Rate', font1)
        plt.ylabel('True Positive Rate', font1)
        plt.title('ROC Curve', font1)
        plt.legend(prop=font1)

        # Plotting Precision-Recall Curves
        plt.figure(2)
        plt.plot(recall, precision, label=f"AUPR={aupr}", color='purple')
        plt.xlabel('Recall', font1)
        plt.ylabel('Precision', font1)
        plt.title('Precision-Recall Curve', font1)
        plt.legend(prop=font1)

        # Displays performance metrics at optimal thresholds
        best_metrics_str = (f"Best Threshold: {best_threshold:.4f}\n"
                            f"Accuracy: {best_metrics['accuracy']:.4f}\n"
                            f"Precision: {best_metrics['precision']:.4f}\n"
                            f"Recall: {best_metrics['recall']:.4f}\n"
                            f"Specificity: {best_metrics['specificity']:.4f}\n"
                            f"MCC: {best_metrics['mcc']:.4f}\n"
                            f"F1 Score: {best_metrics['f1']:.4f}")
        plt.text(0.6, 0.2, best_metrics_str, bbox=dict(facecolor='white', alpha=0.5), fontsize=9)

        # Display and save images
        # plt.savefig("./Result_causal_for_ROC_10_fold_mean.tiff", dpi=600)
        # plt.show()
        # plt.close()
        # Printing Optimal Thresholds and Performance Metrics
        print(f"Best Threshold: {best_threshold}")
        print(f"Accuracy: {best_metrics['accuracy']:.4f}")
        print(f"Precision: {best_metrics['precision']:.4f}")
        print(f"Recall: {best_metrics['recall']:.4f}")  # ÁéÃô¶È
        print(f"Specificity: {best_metrics['specificity']:.4f}")  # ÌØÒìÐÔ
        print(f"MCC: {best_metrics['mcc']:.4f}")
        print(f"F1 Score: {best_metrics['f1']:.4f}")
        print(f"AUROC: {roc_auc_:.4f}")
        print(f"AUPR: {aupr:.4f}")
        model.train()
    return best_metrics['accuracy'], best_metrics['precision'], best_metrics['recall'], best_metrics['specificity'], best_metrics['mcc'], best_metrics['f1'], auc_, aupr_
    # return 0

# for RF and DecisionTree
def train_GUET_(arges, model, sim_set, SVD_NMF, optimizer, pair_pos_neg_fengceng, device):
    model = model.to(device)
    train_miRNA_index_fc = torch.tensor(pair_pos_neg_fengceng['train'][:,0], dtype=torch.long).to(device) # miRNA index for train
    train_disease_index_fc = torch.tensor(pair_pos_neg_fengceng['train'][:,1],dtype=torch.long).to(device) # disease index for train
    train_label_fc = torch.tensor(pair_pos_neg_fengceng['train'][:,2]).to(device).float() # label for train

    test_miRNA_index_fc = torch.tensor(pair_pos_neg_fengceng['test'][:, 0],dtype=torch.long).to(device)  # miRNA index for test
    test_disease_index_fc = torch.tensor(pair_pos_neg_fengceng['test'][:,1],dtype=torch.long).to(device)  # disease index for test
    test_label_fc = torch.tensor(pair_pos_neg_fengceng['test'][:, 2]).to(device).float()  # label for test

    print('######################### start to training #############################')
    mi_emb, di_emb = model(sim_set, SVD_NMF, train_miRNA_index_fc, train_disease_index_fc) # obtain the embeddings of miRNA and disease based on GCN
    mi_train = mi_emb[train_miRNA_index_fc]
    mi_test = mi_emb[test_miRNA_index_fc]
    di_train = di_emb[train_disease_index_fc]
    di_test = di_emb[test_disease_index_fc]
    pair_train = mi_train * di_train
    pair_train = pair_train.cpu().detach().numpy()
    pair_test = mi_test * di_test
    pair_test = pair_test.cpu().detach().numpy()
    train_label_fc = train_label_fc.cpu().detach().numpy()
    test_label_fc = test_label_fc.cpu().detach().numpy()

    #model_RF = RandomForestClassifier(n_estimators=20, random_state=42, max_features=12, max_depth=10)
    # model_RF = DecisionTreeClassifier(random_state=42) # decisiontree
    #model_RF.fit(pair_train, train_label_fc)
    #y_pred = model_RF.predict(pair_test)
    #y_prob = model_RF.predict_proba(pair_test)[:,1]

    # xgb
    param_grid = {'n_estimators': [100, 200, 300],
                  'learning_rate': [0.01, 0.1, 0.2],
                  'max_depth': [3, 4, 5],
                  'subsample': [0.8, 1.0],
                  'colsample_bytree': [0.8, 1.0],
                  'gamma': [0, 0.1, 0.2]}
    model_xgb = xgb.XGBClassifier(eval_metric='logloss', random_state=42)
    grid_search = GridSearchCV(model_xgb, param_grid, cv=3, scoring='roc_auc', verbose=0)
    grid_search.fit(pair_train, train_label_fc)
    best_xgb = grid_search.best_estimator_
    y_pred = best_xgb.predict(pair_test)
    y_prob = best_xgb.predict_proba(pair_test)[:, 1]

    acc = accuracy_score(test_label_fc, y_pred)
    pre = precision_score(test_label_fc, y_pred)
    rec = recall_score(test_label_fc, y_pred)
    f1 = f1_score(test_label_fc, y_pred)
    mcc = matthews_corrcoef(test_label_fc, y_pred)
    auc = roc_auc_score(test_label_fc, y_prob)
    aupr = average_precision_score(test_label_fc, y_prob)

    # calculate spe
    tn = np.sum((y_pred == 0) & (test_label_fc == 0))
    fp = np.sum((y_pred == 1) & (test_label_fc == 0))
    spe = tn / (tn + fp) if (tn + fp) > 0 else 0


    return acc, pre,rec, spe, mcc, f1, auc, aupr