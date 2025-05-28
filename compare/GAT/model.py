import torch
from torch import nn
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv
import os
import numpy as np
from KAN_ import *
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.enabled = False

# GCN
class GCN(nn.Module):
    def __init__(self, in_feat, hidden, out_feat, flag):
        super(GCN, self).__init__()

        if flag == 'miRNA':
            # GCN for miRNA
            self.mi_SNF_1 = GCNConv(in_feat, hidden)
            self.mi_SNF_gat = GATConv(64, 64)
            self.mi_SNF_2 = GCNConv(hidden, out_feat)
        else:
            # GCN for disease
            self.di_SNF_1 = GCNConv(in_feat, hidden)
            self.di_SNF_gat = GATConv(64, 64)
            self.di_SNF_2 = GCNConv(hidden, in_feat)

    def forward(self, sim_set, flag_):
        if flag_ == 'miRNA':
            mi_infeats = torch.randn(585, 64)
            # miRNA
            x_m_g1 = torch.relu(self.mi_SNF_1(mi_infeats.to(device), sim_set['miRNA_SNF']['mi_SNF_edges'].to(device),
                                sim_set['miRNA_SNF']['mi_SNF'][sim_set['miRNA_SNF']['mi_SNF_edges'][0], sim_set['miRNA_SNF']['mi_SNF_edges'][1]]))
            x_m_gat = torch.relu(self.mi_SNF_1(x_m_g1, sim_set['miRNA_SNF']['mi_SNF_edges'].to(device),
                                sim_set['miRNA_SNF']['mi_SNF'][sim_set['miRNA_SNF']['mi_SNF_edges'][0], sim_set['miRNA_SNF']['mi_SNF_edges'][1]]))
            x_m_g2 = torch.relu(self.mi_SNF_2(x_m_gat, sim_set['miRNA_SNF']['mi_SNF_edges'].to(device), sim_set['miRNA_SNF']['mi_SNF']
                                [sim_set['miRNA_SNF']['mi_SNF_edges'][0], sim_set['miRNA_SNF']['mi_SNF_edges'][1]]))

            mi_gcn_feat = (x_m_g1 + x_m_g2) / 2
            return mi_gcn_feat
        else:
            di_infeats = torch.randn(88, 64)
            # disease
            y_d_g1 = torch.relu(self.di_SNF_1(di_infeats.to(device), sim_set['disease_SNF']['di_SNF_edges'].to(device),
                                sim_set['disease_SNF']['di_SNF'][sim_set['disease_SNF']['di_SNF_edges'][0], sim_set['disease_SNF']['di_SNF_edges'][1]]))
            y_d_gat = torch.relu(self.di_SNF_1(y_d_g1, sim_set['disease_SNF']['di_SNF_edges'].to(device),
                                sim_set['disease_SNF']['di_SNF'][sim_set['disease_SNF']['di_SNF_edges'][0], sim_set['disease_SNF']['di_SNF_edges'][1]]))
            y_d_g2 = torch.relu(self.di_SNF_2(y_d_gat, sim_set['disease_SNF']['di_SNF_edges'].to(device), sim_set['disease_SNF']['di_SNF']
                                [sim_set['disease_SNF']['di_SNF_edges'][0], sim_set['disease_SNF']['di_SNF_edges'][1]]))

            di_gcn_feat = (y_d_g1 + y_d_g2) / 2
            return di_gcn_feat


# KAN
class KAN(nn.Module):
    def __init__(self):
        super(KAN, self).__init__()
        self.kanlayer1 = KANLinear(64, 32)
        self.kanlayer2 = KANLinear(32, 16)
        self.kanlayer3 = KANLinear(16, 1)
        # self.kanlayer4 = KANLinear(4, HMDD v2)
    def forward(self, mi_emb, di_emb):

        pair_feat1 = mi_emb * di_emb
        pair_feat2 = self.kanlayer1(pair_feat1)
        pair_feat3 = self.kanlayer2(pair_feat2)
        pair_feat4 = self.kanlayer3(pair_feat3)
        # pair_feat5 = self.kanlayer4(pair_feat4)
        return torch.sigmoid(pair_feat4), pair_feat3

# our model
class GUET(nn.Module):
    def __init__(self):
        super(GUET, self).__init__()
        # define GCN for miRNA and dsease
        self.gcn_miRNA = GCN(64, 64, 64, 'miRNA')
        self.gcn_disease = GCN(64, 64, 64, 'disease')



        # define LayerNorm
        self.LayerNorm = torch.nn.LayerNorm(64)

        # define kan
        self.kan = KAN()

    def forward(self, sim_set, SVD_NMF, train_miRNA_index, train_disease_index):
        # obtain embedding of miRNA and disease based on GCN
        mi_gcn_feat =self.gcn_miRNA(sim_set, 'miRNA')
        mi_gcn_feat = mi_gcn_feat[train_miRNA_index]
        di_gcn_feat = self.gcn_disease(sim_set, 'disease')
        di_gcn_feat = di_gcn_feat[train_disease_index]



        # obtain embedding of miRNA and disease based on SVD
        mi_SVD = SVD_NMF['miRNA']['mi_SVD']
        mi_SVD = torch.from_numpy(mi_SVD).to(device)
        mi_SVD = mi_SVD[train_miRNA_index]
        di_SVD = SVD_NMF['disease']['di_SVD']
        di_SVD = torch.from_numpy(di_SVD).to(device)
        di_SVD = di_SVD[train_disease_index]

        # obtain embedding of miRNA and disease based on NMF
        mi_NMF = SVD_NMF['miRNA']['mi_NMF']
        mi_NMF = torch.from_numpy(mi_NMF).to(device)
        mi_NMF = mi_NMF[train_miRNA_index]
        di_NMF = SVD_NMF['disease']['di_NMF']
        di_NMF = torch.from_numpy(di_NMF).to(device)
        di_NMF = di_NMF[train_disease_index]

        mi_final = 0.5 * mi_gcn_feat + 0.25 * mi_NMF + 0.25 * mi_SVD
        di_final = 0.5 * di_gcn_feat + 0.25 * di_NMF + 0.25 * di_SVD

        mi_final = mi_final.float()
        di_final = di_final.float()

        # Norm
        mi_final = self.LayerNorm(mi_final)
        di_final = self.LayerNorm(di_final)

        # obtain predicting scores using KAN network
        predicting_scores, p = self.kan(mi_final, di_final)

        return predicting_scores.view(-1), p
        # return predicting_scores.view(-HMDD v2)
        # RF and DecisionTree
        # return mi_final, di_final

