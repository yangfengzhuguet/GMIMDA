import csv
import torch
import numpy as np
import pandas as pd
import random
import os
import pickle
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


############### SNF for nonlinear fusion of similarity matrices ##############

def read_csv(path):
    with open(path, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file)
        md_data = []
        md_data += [[float(i) for i in row] for row in reader]
        # md_data = np.array(md_data)
        return torch.tensor(md_data)
        # return md_data

# W is the matrix which needs to be normalized
def new_normalization (w):
    m = w.shape[0]
    p = np.zeros([m,m])
    for i in range(m):
        for j in range(m):
            if i == j:
                p[i][j] = 1/2
            elif np.sum(w[i,:])-w[i,i]>0:
                p[i][j] = w[i,j]/(2*(np.sum(w[i,:])-w[i,i]))
    return p

# get the KNN kernel, k is the number if first nearest neibors
def KNN_kernel (S, k):
    n = S.shape[0]
    S_knn = np.zeros([n,n])
    for i in range(n):
        sort_index = np.argsort(S[i,:])
        for j in sort_index[n-k:n]:
            if np.sum(S[i,sort_index[n-k:n]])>0:
                S_knn [i][j] = S[i][j] / (np.sum(S[i,sort_index[n-k:n]]))
    return S_knn


# updataing rules
def MiRNA_updating (S1,S2,S3,P1,P2,P3):
    it = 0
    P = (P1+P2+P3)/3
    dif = 1
    while dif>0.0000001:
        it = it + 1
        P111 = np.dot(np.dot(S1,(P2+P3)/2),S1.T)
        P111 = new_normalization(P111)
        P222 =np.dot(np.dot(S2,(P1+P3)/2),S2.T)
        P222 = new_normalization(P222)
        P333 = np.dot(np.dot(S3,(P1+P2)/2),S3.T)
        P333 = new_normalization(P333)
        P1 = P111
        P2 = P222
        P3 = P333
        P_New = (P1+P2+P3)/3
        dif = np.linalg.norm(P_New-P)/np.linalg.norm(P)
        P = P_New
    print("Iter numb1", it)
    return P

def disease_updating(S1,S2,S3, P1,P2,P3):
    it = 0
    P = (P1+P2+P3)/3
    dif = 1
    while dif> 0.0000001:
        it = it + 1
        P111 =np.dot(np.dot(S1,(P2+P3)/2), S1.T)
        P111 = new_normalization(P111)
        P222 =np.dot(np.dot(S2,(P1+P3)/2), S2.T)
        P222 = new_normalization(P222)
        P333 = np.dot(np.dot(S3,(P1+P2)/2),S3.T)
        P333 = new_normalization(P333)
        P1 = P111
        P2 = P222
        P3 = P333
        P_New = (P1+P2+P3)/3
        dif = np.linalg.norm(P_New-P)/np.linalg.norm(P)
        P = P_New
    print("Iter numb2", it)
    return P

# multi-source feature fusion through SNF
def get_syn_sim (k1, k2):

    disease_semantic_sim = read_csv('data/miR2Disease/0-1/disease_sim/disSSim.csv') # disease semantic similarity
    disease_GIP_sim = read_csv('data/miR2Disease/0-1/disease_sim/disGIPSim.csv') # disease gip similarity
    disease_cos_sim = read_csv('data/miR2Disease/0-1/disease_sim/disCosSim.csv') # disease cosine similarity

    miRNA_GIP_sim = read_csv('data/miR2Disease/0-1/miRNA_sim/miFunSim_norm.csv') # miRNA function similarity
    miRNA_cos_sim = read_csv('data/miR2Disease/0-1/miRNA_sim/miGIPSim.csv') # miRNA gip similarity
    miRNA_func_sim = read_csv('data/miR2Disease/0-1/miRNA_sim/miCosSim.csv') # miRNA cosine similarity


# normalization for miRNA similarity matrix
    mi_GIP_sim_norm = new_normalization(miRNA_GIP_sim)
    mi_cos_sim_norm = new_normalization(miRNA_cos_sim)
    mi_func_sim_norm = new_normalization(miRNA_func_sim)

# KNN for miRNA similarity matrix
    mi_GIP_knn = KNN_kernel(miRNA_GIP_sim, k1)
    mi_cos_knn = KNN_kernel(miRNA_cos_sim, k1)
    mi_func_knn = KNN_kernel(miRNA_func_sim, k1)

# iteratively update each similarity network
    Pmi= MiRNA_updating(mi_GIP_knn, mi_cos_knn, mi_func_knn, mi_GIP_sim_norm, mi_cos_sim_norm, mi_func_sim_norm)
    Pmi_final = (Pmi + Pmi.T)/2

# normalization for disease similarity matrix
    dis_sem_norm = new_normalization(disease_semantic_sim)
    dis_GIP_norm = new_normalization(disease_GIP_sim)
    dis_cos_norm = new_normalization(disease_cos_sim)

# knn for disease similarity matrix
    dis_sem_knn = KNN_kernel(disease_semantic_sim, k2)
    dis_GIP_knn = KNN_kernel(disease_GIP_sim, k2)
    dis_cos_knn = KNN_kernel(disease_cos_sim, k2)

# iteratively update each similarity network
    Pdiease = disease_updating(dis_sem_knn, dis_GIP_knn, dis_cos_knn, dis_sem_norm, dis_GIP_norm, dis_cos_norm)
    Pdiease_final = (Pdiease+Pdiease.T)/2

# obtain the final similarity matrix of miRNA and disease
    return Pmi_final, Pdiease_final

# The following code is used to obtain the fusion similarity matrix of miRNA and disease

# mi_final, di_final = get_syn_sim(78,37)
# mi_final = pd.DataFrame(mi_final)
# di_final = pd.DataFrame(di_final)
# mi_final.to_csv('mi_SNF.csv', header=False, index=False)
# di_final.to_csv('di_SNF.csv', header=False, index=False)


# ten-folds
def get_fold():
    path = "data/HMDD v2.0/association_matrix_.csv"

    Rowid = []
    Cloumnid = []
    Labels = []
    Divide = []
    Rowid_neg = []
    Cloumnid_neg = []
    Labels_neg = []
    Divide_neg = []
    pos_neg_pair_fengceng = dict()

    # Read csv and save positive and negative samples
    with open(path, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file)
        tt = 0
        for line in reader:
            for i in range(len(line)):
                if float(line[i]) == 1:
                    Divide.append("222")  # positive
                    Rowid.append(tt)
                    Cloumnid.append(i)
                    Labels.append(int(float(line[i])))
                elif float(line[i]) == 0:
                    Divide_neg.append("111")  # negative
                    Rowid_neg.append(tt)
                    Cloumnid_neg.append(i)
                    Labels_neg.append(int(float(line[i])))
                else:
                    pass
            tt = tt + 1

    # print(len(Rowid), len(Cloumnid), len(Labels), len(Divide))
    # print(Rowid[0], Cloumnid[0], Labels[0], Divide[0])

    # Integrate the 4 columns of positive and negative samples and disrupt
    Data = [Rowid, Cloumnid, Labels, Divide]
    Data_neg = [Rowid_neg, Cloumnid_neg, Labels_neg, Divide_neg]

    Data = np.array(Data).T
    Data_neg = np.array(Data_neg).T
    print(Data.shape, Data_neg.shape)

    row = list(range(Data.shape[0]))
    random.shuffle(row)
    Data = Data[row]

    # Ten fold cross-validation
    num_cross_val = 10
    for fold in range(num_cross_val):
        train_pos = np.array([x for i, x in enumerate(Data) if i % num_cross_val != fold])
        test_pos = np.array([x for i, x in enumerate(Data) if i % num_cross_val == fold])

        clo = list(range(Data_neg.shape[0]))
        random.shuffle(clo)
        num = Data.shape[0] / 10 * 9  # Take the same number of negative samples as the 9 fold positive samples
        train_neg = Data_neg[clo][:int(num), :]

        # Reset training and test labels
        for i in range(train_neg.shape[0]):
            train_neg[i][3] = "222"
        for i in range(test_pos.shape[0]):
            test_pos[i][3] = "111"

        # Delete all rows in Data_neg that are contained in train_neg.
        train_neg_set = set(map(tuple, train_neg[:, :3]))  # The first three columns are used as unique identifiers
        Data_neg_filtered = np.array([row for row in Data_neg if tuple(row[:3]) not in train_neg_set])

        # Randomly select the same number of samples as test_pos from Data_neg_filtered
        if Data_neg_filtered.shape[0] >= test_pos.shape[0]:
            sampled_neg = Data_neg_filtered[
                np.random.choice(Data_neg_filtered.shape[0], test_pos.shape[0], replace=False)]
        else:
            raise ValueError("error")

        # Final combined training and test set
        train = np.concatenate((train_pos, train_neg), axis=0)
        test = np.concatenate((test_pos, sampled_neg), axis=0)

        # # Replace -1 with 0 in the third column of the train.
        # train[:, 2][train[:, 2] == '-1'] = 0
        #
        # # Replace -1 with 0 in the third column of the test.
        # test[:, 2][test[:, 2] == '-1'] = 0

        # Upsetting the data again
        li = list(range(train.shape[0]))
        random.shuffle(li)
        train = train[li].astype(int)
        li = list(range(test.shape[0]))
        random.shuffle(li)
        test = test[li].astype(int)
        pos_neg_pair_fengceng = {'train': train, 'test': test}
        # Define the directory and file name where the data will be saved
        parent_dir = "data/miR2Disease/0-1/ten-folds_balance"
        filename = f"pos_neg_pair_10_{fold+1}.pkl"
        file_path = os.path.join(parent_dir, filename)
        # Make sure the directory exists
        os.makedirs(parent_dir, exist_ok=True)
        # Save dictionary to file
        with open(file_path, 'wb') as file:
            pickle.dump(pos_neg_pair_fengceng, file)
        print(f"pos_neg_pair_10_{fold+1} saved to {file_path}")
        # Saved files with 0: miRNA index, 1: disease index, 2: label, 3: useless
    return pos_neg_pair_fengceng
# get_fold()


import csv
import numpy as np
import random
import os
import pickle

def get_fc():
    path = "data/miR2Disease/miR2Disease_MDA01.csv"

    Rowid = []
    Cloumnid = []
    Labels = []
    Divide = []
    Rowid_neg = []
    Cloumnid_neg = []
    Labels_neg = []
    Divide_neg = []
    pos_neg_pair_fengceng = dict()

    # Read csv and save positive and negative samples
    with open(path, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file)
        tt = 0
        for line in reader:
            for i in range(len(line)):
                if float(line[i]) == 1:
                    Divide.append("222")  # positive
                    Rowid.append(tt)
                    Cloumnid.append(i)
                    Labels.append(int(float(line[i])))
                elif float(line[i]) == 0:
                    Divide_neg.append("111")  # negative
                    Rowid_neg.append(tt)
                    Cloumnid_neg.append(i)
                    Labels_neg.append(int(float(line[i])))
                else:
                    pass
            tt = tt + 1

    # Integrate the 4 columns of positive and negative samples
    Data = [Rowid, Cloumnid, Labels, Divide]
    Data_neg = [Rowid_neg, Cloumnid_neg, Labels_neg, Divide_neg]

    Data = np.array(Data).T
    Data_neg = np.array(Data_neg).T
    print(Data.shape, Data_neg.shape)

    # Shuffle the positive and negative samples
    np.random.shuffle(Data)
    np.random.shuffle(Data_neg)

    # Get the number of positive samples
    num_pos = Data.shape[0]

    # Ensure the number of negative samples matches the number of positive samples
    if Data_neg.shape[0] < num_pos:
        raise ValueError("Not enough negative samples to balance with positive samples.")

    # Randomly select negative samples to match the number of positive samples
    Data_neg = Data_neg[np.random.choice(Data_neg.shape[0], num_pos, replace=False)]

    # Split the data into training and testing sets (4/5 for training, 1/5 for testing)
    train_ratio = 0.8
    test_ratio = 0.2

    # Split positive samples
    num_train_pos = int(num_pos * train_ratio)
    train_pos = Data[:num_train_pos]
    test_pos = Data[num_train_pos:]

    # Split negative samples
    num_train_neg = int(num_pos * train_ratio)
    train_neg = Data_neg[:num_train_neg]
    test_neg = Data_neg[num_train_neg:]

    # Combine positive and negative samples for training and testing
    train = np.concatenate((train_pos, train_neg), axis=0)
    test = np.concatenate((test_pos, test_neg), axis=0)

    # Shuffle the training and testing sets
    li = list(range(train.shape[0]))
    random.shuffle(li)
    train = train[li].astype(int)
    li = list(range(test.shape[0]))
    random.shuffle(li)
    test = test[li].astype(int)

    # Save the stratified split
    pos_neg_pair_fengceng = {'train': train, 'test': test}

    # Define the directory and file name where the data will be saved
    parent_dir = "data/miR2Disease/fc"
    filename = "pos_neg_pair_stratified_balanced.pkl"
    file_path = os.path.join(parent_dir, filename)

    # Make sure the directory exists
    os.makedirs(parent_dir, exist_ok=True)

    # Save dictionary to file
    with open(file_path, 'wb') as file:
        pickle.dump(pos_neg_pair_fengceng, file)

    print(f"Stratified and balanced split saved to {file_path}")
    # Saved files with 0: miRNA index, 1: disease index, 2: label, 3: useless

    return pos_neg_pair_fengceng


# get_fc()


# for case study
def get_case():
    path = "data/HMDD v2.0/association_matrix_.csv"

    Rowid = []
    Cloumnid = []
    Labels = []
    Divide = []

    # Read csv and save positive and negative samples
    with open(path, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file)
        tt = 0
        for line in reader:
            for i in range(len(line)):
                if (float(line[i]) == 1 or float(line[i]) == 0):
                    Divide.append("222")  # positive
                    Rowid.append(tt)
                    Cloumnid.append(i)
                    Labels.append(int(float(line[i])))
                else:
                    pass
            tt = tt + 1
    Data = [Rowid, Cloumnid, Labels, Divide]
    Data = np.array(Data).T
    # Define the directory and file name where the data will be saved
    parent_dir = "data/"
    filename = f"case.pkl"
    file_path = os.path.join(parent_dir, filename)
    # Make sure the directory exists
    os.makedirs(parent_dir, exist_ok=True)
    # Save dictionary to file
    with open(file_path, 'wb') as file:
        pickle.dump(Data, file)
    return Data

# get_case()


def get_edge_index(matrix):
    edge_index = [[], []]
    for i in range(matrix.size(0)):
        for j in range(matrix.size(1)):
            if matrix[i][j] != 0:
                edge_index[0].append(i)
                edge_index[1].append(j)
    return torch.LongTensor(edge_index)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_data():
    sim_set = dict()
    # sim_mut = dict()

    # Obtaining miRNAs using SNF fusion, disease similarity matrix, edge set
    mi_final = read_csv('data/miR2Disease/0-1/miRNA_sim/mi_SNF.csv').to(device)
    mi_final_edges = get_edge_index(mi_final).to(device)

    di_final = read_csv('data/miR2Disease/0-1/disease_sim/di_SNF.csv').to(device)
    di_final_edges = get_edge_index(di_final).to(device)
    # Preservation of features fused using SNF
    sim_set['miRNA_SNF'] = {'mi_SNF': mi_final, 'mi_SNF_edges': mi_final_edges}
    sim_set['disease_SNF'] = {'di_SNF': di_final, 'di_SNF_edges': di_final_edges}


    # mi_gua = read_csv('data/miR2Disease/miRNA_sim/miGIPSim.csv').to(device)
    # mi_gua_edges = get_edge_index(mi_gua).to(device)
    # mi_cos = read_csv('data/miR2Disease/miRNA_sim/miCosSim.csv').to(device)
    # mi_cos_edges = get_edge_index(mi_cos).to(device)
    # mi_fun = read_csv('data/miR2Disease/miRNA_sim/miFunSim_norm.csv').to(device)
    # mi_fun_edges = get_edge_index(mi_fun).to(device)
    #
    # di_gua = read_csv('data/miR2Disease/disease_sim/disGIPSim.csv').to(device)
    # di_gua_edges = get_edge_index(di_gua).to(device)
    # di_cos = read_csv('data/miR2Disease/disease_sim/disCosSim.csv').to(device)
    # di_cos_edges = get_edge_index(di_cos).to(device)
    # di_sem = read_csv('data/miR2Disease/disease_sim/disSSim.csv').to(device)
    # di_sem_edges = get_edge_index(di_sem).to(device)
    #
    # sim_mut['miRNA_mut'] = {'mi_gua': mi_gua, 'mi_gua_edges': mi_gua_edges, 'mi_cos': mi_cos, 'mi_cos_edges': mi_cos_edges, 'mi_fun': mi_fun, 'mi_fun_edges': mi_fun_edges}
    # sim_mut['disease_mut'] = {'di_gua': di_gua, 'di_gua_edges': di_gua_edges, 'di_cos': di_cos, 'di_cos_edges': di_cos_edges, 'di_sem': di_sem, 'di_sem_edges': di_sem_edges}

    # Define the directory and file name where the data will be saved
    parent_dir = "data/miR2Disease"
    filename = "sim_mut.pkl"
    file_path = os.path.join(parent_dir, filename)

    # Make sure the directory exists
    os.makedirs(parent_dir, exist_ok=True)

    # Save dictionary to file
    with open(file_path, 'wb') as file:
        pickle.dump(sim_set, file)

    print(f"sim_set saved to {file_path}")
    return sim_set
# Save the results of intermediate preprocessing to a pkl file
# get_data()


# SVD
def S_V_D(mda, k):
    U, S, Vt = np.linalg.svd(mda)

    #Select the first k singular values and the corresponding vectors, S has been automatically ordered
    Sigma_k = np.diag(S[:k]) # kxk

    # Singular value weight fusion
    mi_SVD = U[:, :k].dot(np.sqrt(Sigma_k))
    di_SVD = (np.sqrt(Sigma_k).dot(Vt[:k, :])).T
    return mi_SVD, di_SVD


# NMF
def updating_U (W, A, U, V, lam):
    m, n = U.shape
    fenzi = (W*A).dot(V.T)
    fenmu = (W*(U.dot(V))).dot((V.T)) + (lam/2) *(np.ones([m, n]))

    # fenmu = (W*(U.dot(V))).dot((V.T)) + lam*U
    U_new = U
    for i in range(m):
        for j in range(n):
            U_new[i,j] = U[i, j]*(fenzi[i,j]/fenmu[i, j])
    return U_new

def updating_V(W, A, U, V, lam):
    m, n = V.shape
    fenzi = (U.T).dot(W * A)
    fenmu = (U.T).dot(W * (U.dot(V))) + (lam / 2) * (np.ones([m, n]))
    # fenmu = (U.T).dot(W*(U.dot(V)))+lam*V
    V_new = V
    for i in range(m):
        for j in range(n):
            V_new[i, j] = V[i, j] * (fenzi[i, j] / fenmu[i, j])
    return V_new


def objective_function(W, A, U, V, lam):
    m, n = A.shape
    sum_obj = 0
    for i in range(m):
        for j in range(n):
            # print("the shape of Ui", U[i,:].shape, V[:,j].shape)
            sum_obj = sum_obj + W[i, j] * (A[i, j] - U[i, :].dot(V[:, j])) + lam * (
                        np.linalg.norm(U[i, :], ord=1, keepdims=False) + np.linalg.norm(V[:, j], ord=1, keepdims=False))
    return sum_obj

def get_low_feature(k, lam, th, A):  # k is the number elements in the features, lam is the parameter for adjusting, th is the threshold for coverage state
    # get_low_feature(90, 0.01, pow(10, -4) ,A) where 90 is the embedding of miRNA and disease
    m, n = A.shape
    arr1 = np.random.randint(0, 100, size=(m, k))
    U = arr1 / 100  # miRNA
    arr2 = np.random.randint(0, 100, size=(k, n))
    V = arr2 / 100  # disease
    obj_value = objective_function(A, A, U, V, lam)
    obj_value1 = obj_value + 1
    i = 0
    diff = abs(obj_value1 - obj_value)
    while i < 1000:
        i = i + 1
        U = updating_U(A, A, U, V, lam)
        V = updating_V(A, A, U, V, lam)
        # obj_value1 = obj_value
        # obj_value = objective_function(A, A, U, V, lam)
        # diff = abs(obj_value1 - obj_value)
        # print("ite ", i, diff, obj_value1)
        # print("iter", i)
    # print(U)
    # print(V)
    return U, V.transpose()

# # obtain the embedding based SVD and NMF
#
# SVD_NMF = dict()
# mda = pd.read_csv('data/miR2Disease/miR2Disease_MDA01.csv', header=None, sep=',').to_numpy()
#
# mi_SVD, di_SVD = S_V_D(mda, 64)
# mi_NMF, di_NMF = get_low_feature(64, 0.01, pow(10, -4), mda)
#
# SVD_NMF['miRNA'] = {'mi_SVD': mi_SVD, 'mi_NMF': mi_NMF}
# SVD_NMF['disease'] = {'di_SVD': di_SVD, 'di_NMF': di_NMF}
# parent_dir = "data/miR2Disease/"
# filename = "SVD_NMF.pkl"
# file_path = os.path.join(parent_dir, filename)
#
# # Make sure the directory exists
# os.makedirs(parent_dir, exist_ok=True)
#
# # Save dictionary to file
# with open(file_path, 'wb') as file:
#     pickle.dump(SVD_NMF, file)
#
# print(f"sim_set saved to {file_path}")