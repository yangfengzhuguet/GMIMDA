import time
import torch
import random
from datapro import CVEdgeDataset
from model import MHMDA, EmbeddingM, EmbeddingD
import numpy as np
from sklearn import metrics
import torch.utils.data.dataloader as DataLoader
from sklearn.model_selection import KFold
import os
import pandas as pd
from sklearn.metrics import roc_curve,precision_recall_curve,auc,precision_recall_fscore_support,matthews_corrcoef,accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def save_predictions_labels(test_score, test_label, save_path):
    # Save prediction scores and corresponding labels to a CSV file
    results = np.vstack((test_label, test_score))
    results_df = pd.DataFrame(results.T, columns=["Labels", "Predictions"])
    results_df.to_csv(save_path, index=False)


def setup_seed(seed):
    # Fix random seed for reproducibility across torch, numpy, and random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def construct_het_mat(rna_dis_mat, dis_mat, rna_mat):
    # Construct heterogeneous adjacency matrix by concatenating RNA and disease similarity matrices
    mat1 = np.hstack((rna_mat, rna_dis_mat))
    mat2 = np.hstack((rna_dis_mat.T, dis_mat))
    ret = np.vstack((mat1, mat2))
    return ret


def get_metrics(score, label):
    # Calculate evaluation metrics given prediction scores and true labels
    y_pre = score
    y_true = label
    metric = caculate_metrics(y_pre, y_true)
    return metric


def caculate_metrics(pre_score, real_score):
    # Compute AUC, AUPR, accuracy, F1-score, recall, precision based on prediction scores and ground truth
    y_true = real_score
    y_pre = pre_score
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pre, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    precision_u, recall_u, thresholds_u = metrics.precision_recall_curve(y_true, y_pre)
    aupr = metrics.auc(recall_u, precision_u)
    y_score = [0 if j < 0.5 else 1 for j in y_pre]

    acc = metrics.accuracy_score(y_true, y_score)
    f1 = metrics.f1_score(y_true, y_score)
    recall = metrics.recall_score(y_true, y_score)
    precision = metrics.precision_score(y_true, y_score)

    metric_result = [auc, aupr, acc, f1, recall, precision]
    print("One epoch metric： ")
    print_met(metric_result)
    return metric_result


def print_met(list):
    # Print computed metrics in formatted style
    print('AUC ：%.4f ' % (list[0]),
          'AUPR ：%.4f ' % (list[1]),
          'Accuracy ：%.4f ' % (list[2]),
          'f1_score ：%.4f ' % (list[3]),
          'recall ：%.4f ' % (list[4]),
          'precision ：%.4f \n' % (list[5]))


def check_input_data(data):
    # Verify all input tensor values are in [0, HMDD v2] range; raise error on violation
    for i, value in enumerate(data.flatten()):
        if not (0 <= value <= 1):
            print(f"Problematic tensor: {data}")
            print("Shape of the tensor:", data.shape)
            assert 0 <= value <= 1, f"Input datasets value out of range [0, 1] at index {i}: {value}"



def show_auc(pre_score, label, flag):
    y_true = label.flatten().detach().cpu().numpy()
    y_score = pre_score.flatten().detach().cpu().numpy()
    fpr,tpr,rocth = roc_curve(y_true,y_score)
    auroc = auc(fpr,tpr)
    precision,recall,prth = precision_recall_curve(y_true,y_score)
    aupr = auc(recall,precision)
    return auroc, aupr, fpr, tpr, precision, recall



# def train_test(simData, train_data, param):
#     """
#     Unified training and testing function that trains on all training datasets and evaluates on test datasets
#     without batching or model saving.
#
#     Args:
#         simData: Heterogeneous adjacency matrix or similarity datasets
#         train_data: Dictionary containing train and test edges and labels
#         param: Hyperparameters and configurations object
#
#     Returns:
#         Tuple of (trained_model, train_losses, test_metrics)
#     """
#     # Extract datasets
#     train_edges = train_data['train_Edges']
#     train_labels = train_data['train_Labels']
#     test_edges = train_data['test_Edges']
#     test_labels = train_data['test_Labels']
#
#     torch.manual_seed(42)
#
#     # Initialize model and optimizer
#     model = MHMDA(param, EmbeddingM(param), EmbeddingD(param))
#     model.cuda()
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
#
#     # Prepare datasets (no DataLoader, using all datasets directly)
#     train_data_tensor = torch.from_numpy(train_edges).cuda()
#     train_labels_tensor = torch.from_numpy(train_labels).cuda()
#     test_data_tensor = torch.from_numpy(test_edges).cuda()
#     test_labels_tensor = torch.from_numpy(test_labels).cuda()
#
#     # Training phase
#     print("----- Training -----")
#     train_losses = []
#
#     for epoch in range(param.epoch):
#         model.train()
#         start_time = time.time()
#
#         optimizer.zero_grad()
#         outputs = model(simData, train_data_tensor)
#         auc_, aupr, f, t, p1, r = show_auc(outputs, train_labels_tensor, 'train')
#         loss = torch.nn.BCELoss()(outputs, train_labels_tensor)
#
#         loss.backward()
#         optimizer.step()
#
#         train_losses.append(loss.item())
#         print(f"Epoch {epoch + 1}/{param.epoch}, Loss: {loss.item():.4f}, Time: {time.time() - start_time:.2f}s, AUC: {auc_}, AUPR: {aupr}")
#
#     # Testing phase
#     print("\n----- Testing -----")
#     model.eval()
#     start_time = time.time()
#
#     with torch.no_grad():
#         test_scores = model(simData, test_data_tensor)
#         auc_, aupr_, fp, tp, pre, rec = show_auc(test_scores, test_labels_tensor, 'test')
#         np.save(f'datasets/circRNA-disease_datasets/f_tpr/fpr_0.npy', fp)
#         np.save(f'datasets/circRNA-disease_datasets/f_tpr/tpr_0.npy', tp)
#         np.save(f'datasets/circRNA-disease_datasets/p_r/p_0.npy', pre)
#         np.save(f'datasets/circRNA-disease_datasets/p_r/r_0.npy', rec)
#         print('-------The test: AUC:{}, AUPR{}-----------'.format(auc_, aupr_))
#         font1 = {"family": "Arial", "weight": "book", "size": 9}
#
#         # sample datasets
#         y_true = np.array(test_labels_tensor.detach().cpu())
#         y_true = np.where(y_true == 1, True, False)
#         y_scores = np.array(test_scores.detach().cpu())
#         # Calculate the ROC curve
#         fpr, tpr, thresholds = roc_curve(y_true, y_scores)
#         roc_auc_ = auc(fpr, tpr)
#         roc_auc_ = round(roc_auc_, 3)
#
#         # Calculating Precision-Recall Curves
#         precision, recall, pr_thresholds = precision_recall_curve(y_true, y_scores)
#         aupr = auc(recall, precision)
#         aupr = round(aupr, 3)
#         # Calculate the performance metrics at different thresholds and find the optimal thresholds
#         best_threshold = 0.0
#         best_f1 = 0.0
#         best_metrics = {}
#
#         # Sensitivity and specificity at each threshold are preserved
#         sensitivities = []
#         specificities = []
#
#         for threshold in thresholds:
#             y_pred = (y_scores >= threshold).astype(int)
#             precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
#             accuracy = (y_pred == y_true).mean()
#             mcc = matthews_corrcoef(y_true, y_pred)
#             tn = ((y_pred == 0) & (y_true == 0)).sum()
#             fp = ((y_pred == 1) & (y_true == 0)).sum()
#             fn = ((y_pred == 0) & (y_true == 1)).sum()
#             tp = ((y_pred == 1) & (y_true == 1)).sum()
#             specificity = tn / (tn + fp)
#
#             sensitivities.append(recall)
#             specificities.append(specificity)
#
#             if f1 > best_f1:
#                 best_f1 = f1
#                 best_threshold = threshold
#                 best_metrics = {
#                     "accuracy": accuracy,
#                     "precision": precision,
#                     "recall": recall,
#                     "f1": f1,
#                     "mcc": mcc,
#                     "specificity": specificity
#                 }
#
#         # Displays performance metrics at optimal thresholds
#         best_metrics_str = (f"Best Threshold: {best_threshold:.4f}\n"
#                             f"Accuracy: {best_metrics['accuracy']:.4f}\n"
#                             f"Precision: {best_metrics['precision']:.4f}\n"
#                             f"Recall: {best_metrics['recall']:.4f}\n"
#                             f"Specificity: {best_metrics['specificity']:.4f}\n"
#                             f"MCC: {best_metrics['mcc']:.4f}\n"
#                             f"F1 Score: {best_metrics['f1']:.4f}")
#         print(f"Best Threshold: {best_threshold}")
#         print(f"Accuracy: {best_metrics['accuracy']:.4f}")
#         print(f"Precision: {best_metrics['precision']:.4f}")
#         print(f"Recall: {best_metrics['recall']:.4f}")  # ������
#         print(f"Specificity: {best_metrics['specificity']:.4f}")  # ������
#         print(f"MCC: {best_metrics['mcc']:.4f}")
#         print(f"F1 Score: {best_metrics['f1']:.4f}")
#         print(f"AUROC: {roc_auc_:.4f}")
#         print(f"AUPR: {aupr:.4f}")
#         model.train()
#     return best_metrics['accuracy'], best_metrics['mcc'], best_metrics['f1'], auc_, aupr_



def train_test(simData, train_data, param, state, output_folder):
    """
    Train and validate model with k-fold cross-validation if state='valid';
    Otherwise, perform testing on hold-out data.

    Args:
        simData: Heterogeneous adjacency matrix or similarity data (input to model).
        train_data: Dictionary containing train and test edges and labels.
        param: Hyperparameters and configurations object.
        state: 'valid' for training+validation, else testing.
        output_folder: Directory path to save models and metrics.

    Returns:
        Number of folds in k-fold CV if state != 'valid'.
    """

    epo_metric = []
    valid_metric = []
    all_metrics = []

    # Extract train/test edges and labels
    train_edges = train_data['train_Edges']
    train_labels = train_data['train_Labels']
    test_edges = train_data['test_Edges']
    test_labels = train_data['test_Labels']

    kfolds = param.kfold
    torch.manual_seed(42)

    if state == 'valid':
        # Setup k-fold cross validation splitting
        kf = KFold(n_splits=kfolds, shuffle=True, random_state=1)
        train_idx, valid_idx = [], []
        for train_index, valid_index in kf.split(train_edges):
            train_idx.append(train_index)
            valid_idx.append(valid_index)

        for i in range(kfolds):
            fold_id = i + 1
            # Initialize model and optimizer per fold
            model = MHMDA(param, EmbeddingM(param), EmbeddingD(param))
            model.cuda()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0)

            print(f'################Fold {fold_id} of {kfolds}################')
            edges_train, edges_valid = train_edges[train_idx[i]], train_edges[valid_idx[i]]
            labels_train, labels_valid = train_labels[train_idx[i]], train_labels[valid_idx[i]]

            # Prepare dataset and dataloader for this fold
            trainEdges = CVEdgeDataset(edges_train, labels_train)
            validEdges = CVEdgeDataset(edges_valid, labels_valid)
            trainLoader = DataLoader.DataLoader(trainEdges, batch_size=param.batchSize, shuffle=True, num_workers=0)
            validLoader = DataLoader.DataLoader(validEdges, batch_size=param.batchSize, shuffle=True, num_workers=0)

            print("-----training-----")
            for e in range(param.epoch):
                running_loss = 0.0
                epo_label = []
                epo_score = []
                print("epoch：", e + 1)
                model.train()
                start = time.time()

                for i, item in enumerate(trainLoader):
                    data, label = item
                    train_data = data.cuda()
                    true_label = label.cuda()
                    pre_score = model(simData, train_data)
                    train_loss = torch.nn.BCELoss()
                    loss = train_loss(pre_score, true_label)

                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    running_loss += loss.item()
                    print(f"After batch {i + 1}: loss= {loss:.3f};", end='\n')

                    # Accumulate batch scores and labels for epoch-level statistics
                    batch_score = pre_score.cpu().detach().numpy()
                    epo_score = np.append(epo_score, batch_score)
                    epo_label = np.append(epo_label, label.numpy())

                end = time.time()
                print('Time：%.2f \n' % (end - start))

            # Validation phase for current fold
            valid_score, valid_label = [], []
            model.eval()
            with torch.no_grad():
                print("-----validing-----")
                for i, item in enumerate(validLoader):
                    data, label = item
                    valid_data = data.cuda()
                    pre_score = model(simData, valid_data)
                    batch_score = pre_score.cpu().detach().numpy()
                    valid_score = np.append(valid_score, batch_score)
                    valid_label = np.append(valid_label, label.numpy())
                end = time.time()
                print('Time：%.2f \n' % (end - start))

                # Save fold model checkpoint
                model_path = os.path.join(output_folder, f"fold_{fold_id}.pkl")
                torch.save(model.state_dict(), model_path)

                # Evaluate validation performance
                metric = get_metrics(valid_score, valid_label)
                all_metrics.append(metric)

            # Compute and save mean metrics after all folds trained
            mean_metrics = np.mean(all_metrics, axis=0)
            metrics_path = os.path.join(output_folder, "metrics.txt")
            with open(metrics_path, 'w') as f:
                for metrics in all_metrics:
                    f.write('\t'.join(map(str, metrics)) + '\n')
                f.write("Mean Metrics:\n")
                f.write('\t'.join(map(str, mean_metrics)) + '\n')

    else:
        # Testing phase: Load saved model and evaluate on test data
        test_score, test_label = [], []
        testEdges = CVEdgeDataset(test_edges, test_labels)
        testLoader = DataLoader.DataLoader(testEdges, batch_size=param.batchSize, shuffle=False, num_workers=0)
        model = MHMDA(param, EmbeddingM(param), EmbeddingD(param))
        # Load model checkpoint for first fold by default
        model.load_state_dict(torch.load('./savemodel/circRNA-disease_datasets/fold_2.pkl'))
        model.cuda()
        model.eval()
        with torch.no_grad():
            start = time.time()
            for i, item in enumerate(testLoader):
                data, label = item
                test_data = data.cuda()
                pre_score = model(simData, test_data)
                batch_score = pre_score.cpu().detach().numpy()
                test_score = np.append(test_score, batch_score)
                test_label = np.append(test_label, label.numpy())
            end = time.time()
            print('Time：%.2f \n' % (end - start))
            # Compute test metrics
            metrics = get_metrics(test_score, test_label)
        print(np.array(valid_metric))
        cv_metric = np.mean(valid_metric, axis=0)
        print_met(cv_metric)

        return kfolds