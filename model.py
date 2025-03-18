import torch
from torch import nn
from torch_geometric.nn import GCNConv
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
            self.mi_SNF_2 = GCNConv(hidden, out_feat)
        else:
            # GCN for disease
            self.di_SNF_1 = GCNConv(in_feat, hidden)
            self.di_SNF_2 = GCNConv(hidden, in_feat)

    def forward(self, sim_set, flag_):
        if flag_ == 'miRNA':
            mi_infeats = torch.randn(215, 64)
            # miRNA
            x_m_g1 = torch.relu(self.mi_SNF_1(mi_infeats.to(device), sim_set['miRNA_SNF']['mi_SNF_edges'].to(device),
                                sim_set['miRNA_SNF']['mi_SNF'][sim_set['miRNA_SNF']['mi_SNF_edges'][0], sim_set['miRNA_SNF']['mi_SNF_edges'][1]]))
            x_m_g2 = torch.relu(self.mi_SNF_2(x_m_g1, sim_set['miRNA_SNF']['mi_SNF_edges'].to(device), sim_set['miRNA_SNF']['mi_SNF']
                                [sim_set['miRNA_SNF']['mi_SNF_edges'][0], sim_set['miRNA_SNF']['mi_SNF_edges'][1]]))

            # miRNA
            # x_m_g1 = torch.relu(self.mi_SNF_1(mi_infeats.to(device), sim_set['miRNA_mut']['mi_gua_edges'].to(device),
            #                     sim_set['miRNA_mut']['mi_gua'][sim_set['miRNA_mut']['mi_gua_edges'][0], sim_set['miRNA_mut']['mi_gua_edges'][1]]))
            # x_m_g2 = torch.relu(self.mi_SNF_2(x_m_g1, sim_set['miRNA_mut']['mi_gua_edges'].to(device), sim_set['miRNA_mut']['mi_gua']
            #                     [sim_set['miRNA_mut']['mi_gua_edges'][0], sim_set['miRNA_mut']['mi_gua_edges'][1]]))
            #
            # x_m_c1 = torch.relu(self.mi_SNF_1(mi_infeats.to(device), sim_set['miRNA_mut']['mi_cos_edges'].to(device),
            #                     sim_set['miRNA_mut']['mi_cos'][sim_set['miRNA_mut']['mi_cos_edges'][0], sim_set['miRNA_mut']['mi_cos_edges'][1]]))
            # x_m_c2 = torch.relu(self.mi_SNF_2(x_m_c1, sim_set['miRNA_mut']['mi_cos_edges'].to(device), sim_set['miRNA_mut']['mi_cos']
            #                     [sim_set['miRNA_mut']['mi_cos_edges'][0], sim_set['miRNA_mut']['mi_cos_edges'][1]]))
            #
            # x_m_f1 = torch.relu(self.mi_SNF_1(mi_infeats.to(device), sim_set['miRNA_mut']['mi_fun_edges'].to(device),
            #                     sim_set['miRNA_mut']['mi_fun'][sim_set['miRNA_mut']['mi_fun_edges'][0], sim_set['miRNA_mut']['mi_fun_edges'][1]]))
            # x_m_f2 = torch.relu(self.mi_SNF_2(x_m_f1, sim_set['miRNA_mut']['mi_fun_edges'].to(device), sim_set['miRNA_mut']['mi_fun']
            #                     [sim_set['miRNA_mut']['mi_fun_edges'][0], sim_set['miRNA_mut']['mi_fun_edges'][1]]))


            # mi_gcn_feat = (x_m_g1 + x_m_g2 + x_m_c1 + x_m_c2 + x_m_f1 + x_m_f2) / 6
            mi_gcn_feat = (x_m_g1 + x_m_g2) / 2
            return mi_gcn_feat
        else:
            di_infeats = torch.randn(110, 64)
            # disease
            y_d_g1 = torch.relu(self.di_SNF_1(di_infeats.to(device), sim_set['disease_SNF']['di_SNF_edges'].to(device),
                                sim_set['disease_SNF']['di_SNF'][sim_set['disease_SNF']['di_SNF_edges'][0], sim_set['disease_SNF']['di_SNF_edges'][1]]))
            y_d_g2 = torch.relu(self.di_SNF_2(y_d_g1, sim_set['disease_SNF']['di_SNF_edges'].to(device), sim_set['disease_SNF']['di_SNF']
                                [sim_set['disease_SNF']['di_SNF_edges'][0], sim_set['disease_SNF']['di_SNF_edges'][1]]))

            # # disease
            # y_d_g1 = torch.relu(self.di_SNF_1(di_infeats.to(device), sim_set['disease_mut']['di_gua_edges'].to(device),
            #                     sim_set['disease_mut']['di_gua'][sim_set['disease_mut']['di_gua_edges'][0], sim_set['disease_mut']['di_gua_edges'][1]]))
            # y_d_g2 = torch.relu(self.di_SNF_2(y_d_g1, sim_set['disease_mut']['di_gua_edges'].to(device), sim_set['disease_mut']['di_gua']
            #                     [sim_set['disease_mut']['di_gua_edges'][0], sim_set['disease_mut']['di_gua_edges'][1]]))
            #
            # y_d_c1 = torch.relu(self.di_SNF_1(di_infeats.to(device), sim_set['disease_mut']['di_gua_edges'].to(device),
            #                     sim_set['disease_mut']['di_gua'][sim_set['disease_mut']['di_gua_edges'][0], sim_set['disease_mut']['di_gua_edges'][1]]))
            # y_d_c2 = torch.relu(self.di_SNF_2(y_d_c1, sim_set['disease_mut']['di_gua_edges'].to(device), sim_set['disease_mut']['di_gua']
            #                     [sim_set['disease_mut']['di_gua_edges'][0], sim_set['disease_mut']['di_gua_edges'][1]]))
            #
            # y_d_d1 = torch.relu(self.di_SNF_1(di_infeats.to(device), sim_set['disease_mut']['di_gua_edges'].to(device),
            #                     sim_set['disease_mut']['di_gua'][sim_set['disease_mut']['di_gua_edges'][0], sim_set['disease_mut']['di_gua_edges'][1]]))
            # y_d_d2 = torch.relu(self.di_SNF_2(y_d_d1, sim_set['disease_mut']['di_gua_edges'].to(device), sim_set['disease_mut']['di_gua']
            #                     [sim_set['disease_mut']['di_gua_edges'][0], sim_set['disease_mut']['di_gua_edges'][1]]))

            # di_gcn_feat = (y_d_g1 + y_d_g2 + y_d_c1 + y_d_c2 + y_d_d1 + y_d_d2) / 6
            di_gcn_feat = (y_d_g1 + y_d_g2) / 2
            return di_gcn_feat

# Game theory
class GameTheory(nn.Module):
    def __init__(self, num_miRNA_features, num_disease_features, hidden_dim):
        super(GameTheory, self).__init__()

        # define network
        self.miRNA_layer = nn.Linear(num_miRNA_features, hidden_dim)
        self.disease_layer = nn.Linear(num_disease_features, hidden_dim)

    # Calculate the Nicorette inner product
    def cosine(self, x1, x2):
        numerator = torch.einsum('ij,ij->i', x1, x2)
        denominator = torch.norm(x1, dim=1) * torch.norm(x2, dim=1)
        return numerator / denominator

    def calculate_greedy_strategy(self, strategies, rewards, miRNA_index, disease_index, num_miRNAs, num_diseases,
                                  flag):
        """Calculates the greedy best strategy for each player."""
        num_pairs = strategies.shape[0]
        best_strategies = strategies.clone().detach()

        if flag == 'miRNA':
            # Constructing a local benefit matrix
            local_num_miRNAs = len(torch.unique(miRNA_index))
            local_num_diseases = len(torch.unique(disease_index))
            payoff_matrix = torch.zeros(local_num_miRNAs, local_num_diseases, dtype=rewards.dtype,
                                        device=rewards.device)

            # Fill rewards into the local revenue matrix
            for i in range(num_pairs):
                miRNA_idx = torch.where(torch.unique(miRNA_index) == miRNA_index[i])[0].item()
                disease_idx = torch.where(torch.unique(disease_index) == disease_index[i])[0].item()
                payoff_matrix[miRNA_idx, disease_idx] = rewards[i]

            # Calculate the optimal strategy
            best_indices = torch.argmax(payoff_matrix, dim=1)
            for i in range(num_pairs):
                miRNA_idx = torch.where(torch.unique(miRNA_index) == miRNA_index[i])[0].item()
                best_strategies[i] = strategies[best_indices[miRNA_idx]].clone()

        else:  # disease
            # Constructing a local benefit matrix
            local_num_miRNAs = len(torch.unique(miRNA_index))
            local_num_diseases = len(torch.unique(disease_index))
            payoff_matrix = torch.zeros(local_num_diseases, local_num_miRNAs, dtype=rewards.dtype,
                                        device=rewards.device)

            # Fill rewards into the local revenue matrix
            for i in range(num_pairs):
                miRNA_idx = torch.where(torch.unique(miRNA_index) == miRNA_index[i])[0].item()
                disease_idx = torch.where(torch.unique(disease_index) == disease_index[i])[0].item()
                payoff_matrix[disease_idx, miRNA_idx] = rewards[i]

            # Calculate the optimal strategy

            best_indices = torch.argmax(payoff_matrix, dim=1)
            for i in range(num_pairs):
                disease_idx = torch.where(torch.unique(disease_index) == disease_index[i])[0].item()
                best_strategies[i] = strategies[best_indices[disease_idx]].clone()

        return best_strategies

    # revenue function based on nikolov_inner_product
    def payoff_function(self, miRNA_embedding, disease_embedding):
        # num_miRNAs = miRNA_embedding.shape[0]
        # num_diseases = disease_embedding.shape[0]
        # payoff_matrix = torch.zeros(num_miRNAs, num_diseases, dtype=miRNA_embedding.dtype,
        #                             device=miRNA_embedding.device)
        # for i in range(num_miRNAs):
        #     for j in range(num_diseases):
        #         payoff_matrix[i, j] = self.nikolov_inner_product(miRNA_embedding[i].unsqueeze(0),
        #                                                          disease_embedding[j].unsqueeze(0))
        return self.cosine(miRNA_embedding, disease_embedding)
        # return payoff_matrix

    def forward(self, miRNA_embeddings, disease_embeddings, miRNA_index, disease_index):
        """
        :param miRNAembeddings: the embeddings of all miRNAs
        :param disease_embeddings: the embeddings of all dseases
        :param miRNA_index: current miRNA index
        :param disease_index: current disease index
        """
        num_miRNAs = miRNA_embeddings.shape[0]
        num_diseases = disease_embeddings.shape[0]
        miRNA_embedding = torch.gather(miRNA_embeddings, 0, miRNA_index.unsqueeze(1).expand(-1, miRNA_embeddings.shape[1]))
        disease_embedding = torch.gather(disease_embeddings, 0, disease_index.unsqueeze(1).expand(-1, disease_embeddings.shape[1]))

        miRNA_embedding = self.miRNA_layer(miRNA_embedding)
        disease_embedding = self.disease_layer(disease_embedding)

        # calculate payoff
        payoff = self.payoff_function(miRNA_embedding, disease_embedding)

        # Approximating the most strategic using the greedy algorithm  (Cross-cutting strategy update)
        rewards = payoff

        best_miRNA_strategies = self.calculate_greedy_strategy(miRNA_embedding, rewards, miRNA_index, disease_index,
                                                               num_miRNAs, num_diseases, 'miRNA')
        best_disease_strategies = self.calculate_greedy_strategy(disease_embedding, rewards, miRNA_index, disease_index,
                                                                 num_miRNAs, num_diseases, 'disease')

        # Calculating the Nash equilibrium loss
        nash_loss_miRNA = torch.mean((miRNA_embedding - best_miRNA_strategies) **2)
        nash_loss_disease = torch.mean((disease_embedding - best_disease_strategies) **2)
        nash_loss = (nash_loss_miRNA + nash_loss_disease) / 2
        return nash_loss, best_miRNA_strategies, best_disease_strategies

# MLP
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.acti_func = torch.sigmoid
        self.linear1 = nn.Linear(64, 32, bias=True)
        self.linear2 = nn.Linear(32, 16, bias=True)
        self.linear3 = nn.Linear(16, 1, bias=True)
    def forward(self, mi_emb, di_emb):
        pair_feat1 = mi_emb * di_emb
        pair_feat2 = self.linear1(pair_feat1)
        pair_feat3 = self.linear2(pair_feat2)
        pair_feat4 = self.linear3(pair_feat3)
        return torch.sigmoid(pair_feat4), pair_feat3

# KAN
class KAN(nn.Module):
    def __init__(self):
        super(KAN, self).__init__()
        self.kanlayer1 = KANLinear(64, 32)
        self.kanlayer2 = KANLinear(32, 16)
        self.kanlayer3 = KANLinear(16, 1)
        # self.kanlayer4 = KANLinear(4, 1)
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

        # define the GameTheory
        self.GameTheory = GameTheory(64, 64, 64)

        # define the kronecker product
        # self.kron_product = KroneckerFusion(495, 64) # miRNA
        # self.kron_product = KroneckerFusion(383, 64) # disease

        # define LayerNorm
        self.LayerNorm = torch.nn.LayerNorm(64)

        # define kan
        self.kan = KAN()

        # define MLP for contrastive experiment
        self.mlp = MLP()

    def forward(self, sim_set, SVD_NMF, train_miRNA_index, train_disease_index):
        # obtain embedding of miRNA and disease based on GCN
        mi_gcn_feat =self.gcn_miRNA(sim_set, 'miRNA')
        di_gcn_feat = self.gcn_disease(sim_set, 'disease')

        # obtain the payoff based on GameTheory
        nash_loss, best_miRNA_strategies, best_disease_strategies = self.GameTheory(mi_gcn_feat, di_gcn_feat, train_miRNA_index, train_disease_index)

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

        mi_final = 0.5 * best_miRNA_strategies + 0.25 * mi_NMF + 0.25 * mi_SVD
        di_final = 0.5 * best_disease_strategies + 0.25 * di_NMF + 0.25 * di_SVD
        # mi_gcn_feat = mi_gcn_feat[train_miRNA_index]
        # di_gcn_feat = di_gcn_feat[train_disease_index]
        # mi_final = 0.5 * mi_gcn_feat + 0.25 * mi_SVD + 0.25 * mi_NMF
        # di_final = 0.5 * di_gcn_feat + 0.25 * di_SVD + 0.25 * di_NMF
        mi_final = mi_final.float()
        di_final = di_final.float()

        # Norm
        mi_final = self.LayerNorm(mi_final)
        di_final = self.LayerNorm(di_final)

        # obtain predicting scores using KAN network
        predicting_scores, p = self.kan(mi_final, di_final)

        return predicting_scores.view(-1), nash_loss, p
        # return predicting_scores.view(-1)
        # RF and DecisionTree
        # return mi_final, di_final

