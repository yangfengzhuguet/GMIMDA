import copy
from sklearn.metrics import roc_curve,precision_recall_curve,auc,precision_recall_fscore_support,matthews_corrcoef,accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from dataload import *
from model import *
from torch.utils.data import DataLoader
from model import autoEncoder, encoderLoss
from torch.optim import Adam
from dataload import netsDataset
from utils import obtain_constraints
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import minmax_scale
import numpy as np
from utils import show_auc

class Experiment:
    def __init__(self):
        super(Experiment, self).__init__()

    def setup_seed(self, seed):
        t.manual_seed(seed)
        t.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)
        t.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def MACFE(self, args, data, entrie):
        self.setup_seed(args.seed)
        top_rate = args.CFE_rate
        networks = data[f'{entrie}']['features']
        symbols = data[f'{entrie}']['symbols']
        if entrie == 'miRNA':
            CFE_net_nums = [args.CFE_miRNA_num for i in range(args.CFE_nets)]
            CFE_hidden_dim = args.CFE_hidden_miRNA
            CFE_batch_size = args.CFE_batch_size_miRNA
        else:
            CFE_net_nums = [args.CFE_disease_num for i in range(args.CFE_nets)]
            CFE_hidden_dim = args.CFE_hidden_disease
            CFE_batch_size = args.CFE_batch_size_disease

        constraints_ml = [np.zeros((i, i)) for i in CFE_net_nums]
        constraints_cl = [np.zeros((i, i)) for i in CFE_net_nums]
        CFE_layers = args.CFE_layers
        models = [autoEncoder(CFE_net_nums[i], CFE_hidden_dim) for i in range(len(networks))]
        for idx_layer in range(CFE_layers):
            emb = [np.zeros((CFE_net_nums[i], CFE_hidden_dim[idx_layer])) for i in range(args.CFE_nets)]
            reshape_xs = []
            for idx_net in range(args.CFE_nets):
                model = models[idx_net]
                optimizer = Adam(model.parameters(), lr=args.CFE_lr[idx_layer])
                ml = torch.from_numpy(constraints_ml[idx_net]).float().cuda()
                cl = torch.from_numpy(constraints_cl[idx_net]).float().cuda()
                criterion = encoderLoss(ml, cl, CFE_batch_size, args.CFE_gamma)
                if torch.cuda.is_available():
                    model = model.cuda()
                dataset = netsDataset(networks[idx_net])
                dataloader = DataLoader(dataset, batch_size=CFE_batch_size, shuffle=True)

                for epoch in range(1, args.CFE_epoch + 1):
                    model.train()
                    losses, losses_ml, losses_cl = [], [], []
                    for step, (X, y, indx) in enumerate(dataloader):
                        X, y = X.float().cuda(), y.float().cuda()
                        indx = indx
                        y_pred, _ = model(X, idx_layer)
                        loss, loss_ml, loss_cl = criterion(y_pred, y, indx)
                        losses.append(loss.item())
                        losses_ml.append(loss_ml.item())
                        losses_cl.append(loss_cl.item())
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                with torch.no_grad():
                    model.eval()
                    data = torch.from_numpy(networks[idx_net].astype(np.float32)).cuda()
                    reshape_x, features = model(data, idx_layer, flag='reshape')
                    emb[idx_net] = features.cpu().numpy()
                    reshape_xs.append(reshape_x.cpu().numpy())
            # print("### Extracting constraints..")
            constraints_ml, constraints_cl = obtain_constraints(args.CFE_nets, reshape_xs, symbols, top_rate,
                                                                idx_layer)
            networks = emb
        if args.fusion:
            networks = np.concatenate(networks, axis=1)
            networks = cosine_similarity(networks)
            networks = minmax_scale(networks)
            return networks
        else:
            return networks

    def loss_gcl(self, gcl_model, graph_learner, features, args, anchor_adj):
        if args.maskfeat_rate_anchor:
            mask_v1, _ = datapro().get_feat_mask(features, args.maskfeat_rate_anchor)
            features_v1 = features * (1 - mask_v1)
        else:
            features_v1 = copy.deepcopy(features)
        z1, _ = gcl_model(features_v1, anchor_adj, args, branch='anchor')

        if args.maskfeat_rate_learner:
            mask, _ = datapro().get_feat_mask(features, args.maskfeat_rate_learner)
            features_v2 = features * (1 - mask)
        else:
            features_v2 = copy.deepcopy(features)

        learned_adj = graph_learner(features)
        learned_adj = datapro().symmetrize(learned_adj)
        learned_adj = datapro().normalize(learned_adj)
        z2, _ = gcl_model(features_v2, learned_adj, args, branch='learner')
        # compute loss
        gcl_loss = calc_loss(z1, z2)

        return gcl_loss, learned_adj

    def loss_cls(self, model, features, Adj, train_matrix):
        m_d_res = model(features, Adj)
        loss1 = nn.MSELoss()
        loss = loss1(m_d_res, train_matrix)
        return loss, m_d_res

    def get_metrics(self, real_score, predict_score):
        real_score = real_score.detach().numpy()
        sorted_predict_score = np.array(sorted(list(set(np.array(predict_score).flatten()))))
        sorted_predict_score_num = len(sorted_predict_score)
        thresholds = sorted_predict_score[np.int32(sorted_predict_score_num * np.arange(1, 1000) / 1000)]
        thresholds = np.mat(thresholds)
        thresholds_num = thresholds.shape[1]

        predict_score_matrix = np.tile(predict_score, (thresholds_num, 1))
        negative_index = np.where(predict_score_matrix < thresholds.T)
        positive_index = np.where(predict_score_matrix >= thresholds.T)
        predict_score_matrix[negative_index] = 0
        predict_score_matrix[positive_index] = 1

        TP = predict_score_matrix.dot(real_score.T)
        FP = predict_score_matrix.sum(axis=1) - TP
        FN = real_score.sum() - TP
        TN = len(real_score.T) - TP - FP - FN

        # 计算MCC (Matthews Correlation Coefficient)
        denominator = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
        denominator = np.where(denominator == 0, 1, denominator)  # 避免除以0
        mcc_list = (TP * TN - FP * FN) / denominator

        fpr = FP / (FP + TN)
        tpr = TP / (TP + FN)

        ROC_dot_matrix = np.mat(sorted(np.column_stack((fpr, tpr)).tolist())).T
        ROC_dot_matrix.T[0] = [0, 0]
        ROC_dot_matrix = np.c_[ROC_dot_matrix, [1, 1]]
        x_ROC = ROC_dot_matrix[0].T
        y_ROC = ROC_dot_matrix[1].T
        auc = 0.5 * (x_ROC[1:] - x_ROC[:-1]).T * (y_ROC[:-1] + y_ROC[1:])

        recall_list = tpr
        precision_list = TP / (TP + FP)
        PR_dot_matrix = np.mat(sorted(np.column_stack(
            (recall_list, precision_list)).tolist())).T
        PR_dot_matrix.T[0] = [0, 1]
        PR_dot_matrix = np.c_[PR_dot_matrix, [1, 0]]
        x_PR = PR_dot_matrix[0].T
        y_PR = PR_dot_matrix[1].T
        aupr = 0.5 * (x_PR[1:] - x_PR[:-1]).T * (y_PR[:-1] + y_PR[1:])

        f1_score_list = 2 * TP / (len(real_score.T) + TP - TN)
        accuracy_list = (TP + TN) / len(real_score.T)
        specificity_list = TN / (TN + FP)

        max_index = np.argmax(f1_score_list)
        f1_score = f1_score_list[max_index]
        accuracy = accuracy_list[max_index]
        specificity = specificity_list[max_index]
        recall = recall_list[max_index]
        precision = precision_list[max_index]
        mcc = mcc_list[max_index]  # 获取最佳阈值对应的MCC

        # 将MCC加入metric和metric_list
        metric = [aupr[0, 0], auc[0, 0], f1_score, accuracy, recall, specificity, precision, mcc]
        metric_list = [fpr, tpr, recall_list, precision_list, auc[0, 0], aupr[0, 0]]
        return metric, metric_list

    def get_metric(self, test_score, test_label_fc, i):
        test_score = torch.from_numpy(test_score)
        auc_, aupr_, fp, tp, pre, rec = show_auc(test_score, test_label_fc, 'test')
        np.save(f'../Data/circRNA-disease_datasets/f_tpr/fpr_{i}.npy', fp)
        np.save(f'../Data/circRNA-disease_datasets/f_tpr/tpr_{i}.npy', tp)
        np.save(f'../Data/circRNA-disease_datasets/p_r/p_{i}.npy', pre)
        np.save(f'../Data/circRNA-disease_datasets/p_r/r_{i}.npy', rec)
        # sample datasets
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

        # Displays performance metrics at optimal thresholds
        best_metrics_str = (f"Best Threshold: {best_threshold:.4f}\n"
                            f"Accuracy: {best_metrics['accuracy']:.4f}\n"
                            f"Precision: {best_metrics['precision']:.4f}\n"
                            f"Recall: {best_metrics['recall']:.4f}\n"
                            f"Specificity: {best_metrics['specificity']:.4f}\n"
                            f"MCC: {best_metrics['mcc']:.4f}\n"
                            f"F1 Score: {best_metrics['f1']:.4f}")
        print(f"Best Threshold: {best_threshold}")
        print(f"Accuracy: {best_metrics['accuracy']:.4f}")
        print(f"Precision: {best_metrics['precision']:.4f}")
        print(f"Recall: {best_metrics['recall']:.4f}")  # ������
        print(f"Specificity: {best_metrics['specificity']:.4f}")  # ������
        print(f"MCC: {best_metrics['mcc']:.4f}")
        print(f"F1 Score: {best_metrics['f1']:.4f}")
        print(f"AUROC: {roc_auc_:.4f}")
        print(f"AUPR: {aupr:.4f}")

        return best_metrics['accuracy'], best_metrics['mcc'], best_metrics['f1'], auc_, aupr_

    def train(self, args, data):
        evaluation_results = {}
        # self.setup_seed(args.seed)
        features, original_adj, m_d_matrix = datapro().get_data(data)
        num_feats = features.shape[1]
        index = datapro().datasplit(args, m_d_matrix)
        pre_matrix = np.zeros(m_d_matrix.shape)
        metric = np.zeros((1, 5))
        fpr_fold, tpr_fold, recall_fold, precision_fold, auc_fold, aupr_fold, MCC= [], [], [], [], [], [], []
        for fold in range(args.k_fold):
            print("------this is %d/%d cross validation ------" % (fold + 1, args.k_fold))
            train_matrix = np.matrix(m_d_matrix, copy=True)
            train_matrix[tuple(np.array(index[fold]).T)] = 0  # train set label=0
            # anchor view
            anchor_adj = t.from_numpy(original_adj)
            anchor_adj = datapro().normalize(anchor_adj)
            anchor_adj = anchor_adj.float()
            out = train_matrix.shape[1]
            # graph learner——get the learner view
            graph_learner = FGP_learner(features.cpu(), args.k, args.sim_function, 6)
            # gcl_model = GCL(num_feats, args)
            gcl_model = MoCo_MDA(num_feats, args)
            gsl_model = GCN(num_feats, out, args)

            optimizer_learner = t.optim.Adam(graph_learner.parameters(), lr=args.lr, weight_decay=args.w_decay)
            optimizer_gcl = t.optim.Adam(gcl_model.parameters(), lr=args.lr, weight_decay=args.w_decay)
            optimizer_gsl = t.optim.Adam(gsl_model.parameters(), lr=args.lr_cls, weight_decay=args.w_decay_cls)
            if t.cuda.is_available():
                graph_learner = graph_learner.cuda()
                gcl_model = gcl_model.cuda()
                gsl_model = gsl_model.cuda()
                features = features.cuda()
                train_matrix = t.FloatTensor(train_matrix).cuda()
                anchor_adj = anchor_adj.cuda()
            # start train
            for epoch in range(1, args.epochs + 1):
                graph_learner.train()
                gcl_model.train()

                gcl_loss, Adj = self.loss_gcl(gcl_model, graph_learner, features, args, anchor_adj)

                optimizer_learner.zero_grad()
                optimizer_gcl.zero_grad()
                gcl_loss.backward()
                optimizer_learner.step()
                optimizer_gcl.step()

                gsl_model.train()
                learner_adj = Adj
                learner_adj = learner_adj.detach()
                gsl_loss, m_d_res = self.loss_cls(gsl_model, features, learner_adj, train_matrix)
                optimizer_gsl.zero_grad()
                gsl_loss.backward()
                optimizer_gsl.step()

                if args.c:
                    anchor_adj = anchor_adj * args.tau + learner_adj * (1 - args.tau)

                if epoch == args.epochs:
                    graph_learner.eval()
                    gcl_model.eval()
                    gsl_model.eval()
                    predict_y_proba = m_d_res.reshape(args.d_num + args.m_num,
                                                      args.d_num + args.m_num).cpu().detach().numpy()
                    pre_matrix[tuple(np.array(index[fold]).T)] = predict_y_proba[tuple(np.array(index[fold]).T)]
                    real_score = m_d_matrix[tuple(np.array(index[fold]).T)]
                    pre_score = predict_y_proba[tuple(np.array(index[fold]).T)]
                    # metric_tmp, metric_list_tmp = self.get_metrics(pre_score, real_score)
                    ACC, MCC, f1, auc, aupr = self.get_metric(pre_score, real_score, fold+1)
                    final_results = {'Aupr': aupr, 'AUC': auc, 'F1_Score': f1,
                                     'ACC': ACC, 'MCC': MCC}
                    metric_tmp = [aupr, auc, f1, ACC, MCC]
            metric += metric_tmp
            result = {'Aupr': metric_tmp[0], 'AUC': metric_tmp[1], 'F1_Score': metric_tmp[2],
                                     'ACC': metric_tmp[3], 'MCC': metric_tmp[4]}
            evaluation_results['Fold {}'.format(fold + 1)] = result
        final_result = (metric / args.k_fold)[0]
        final_results = {'Aupr': final_result[0], 'AUC': final_result[1], 'F1_Score': final_result[2],
                         'ACC': final_result[3], 'MCC': final_result[4]}
        #             fpr_fold.append(metric_list_tmp[0])
        #             tpr_fold.append(metric_list_tmp[HMDD v2])
        #             recall_fold.append(metric_list_tmp[2])
        #             precision_fold.append(metric_list_tmp[3])
        #             auc_fold.append(metric_list_tmp[4])
        #             aupr_fold.append(metric_list_tmp[5])
        #     metric += metric_tmp
        #     result = {'Aupr': metric_tmp[0], 'AUC': metric_tmp[HMDD v2], 'F1_Score': metric_tmp[2],
        #               'ACC': metric_tmp[3], 'Recall': metric_tmp[4], 'Specificity': metric_tmp[5],
        #               'Precision': metric_tmp[6], 'MCC': metric_tmp[7]}
        #     evaluation_results['Fold {}'.format(fold + HMDD v2)] = result
        #     print('fold {}:{}'.format(fold + HMDD v2, result))
        # final_result = (metric / args.k_fold)[0]
        # final_results = {'Aupr': final_result[0], 'AUC': final_result[HMDD v2], 'F1_Score': final_result[2],
        #                  'ACC': final_result[3], 'Recall': final_result[4], 'Specificity': final_result[5],
        #                  'Precision': final_result[6], 'MCC':final_result[7]}
        # print('Average: {}'.format(final_results))
        # fpr_fold = np.array(fpr_fold)
        # tpr_fold = np.array(tpr_fold)
        # precision_fold = np.array(precision_fold)
        # recall_fold = np.array(recall_fold)
        # np.save(f'../Data/HMDD v2/f_tpr/fpr.npy', fpr_fold)
        # np.save(f'../Data/HMDD v2/f_tpr/tpr.npy', tpr_fold)
        # np.save(f'../Data/HMDD v2/p_r/p.npy', precision_fold)
        # np.save(f'../Data/HMDD v2/p_r/r.npy', recall_fold)

        # return evaluation_results, final_results
        return evaluation_results, final_results
