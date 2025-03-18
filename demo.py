import torch
import torch.nn as nn
import torch.nn.functional as F


class StrategyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(StrategyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class GameTheory(nn.Module):
    def __init__(self, num_miRNA_features, num_disease_features, hidden_dim, strategy_dim, label_type='classification'):
        super(GameTheory, self).__init__()
        self.miRNA_layer = nn.Linear(num_miRNA_features, hidden_dim)
        self.disease_layer = nn.Linear(num_disease_features, hidden_dim)
        self.miRNA_strategy_net = StrategyNetwork(hidden_dim, hidden_dim, strategy_dim)
        self.disease_strategy_net = StrategyNetwork(hidden_dim, hidden_dim, strategy_dim)
        self.label_type = label_type  # 'regression' or 'classification'
        if self.label_type != 'classification':
            raise ValueError("For binary classification, label_type must be 'classification'.")

    def nikolov_inner_product(self, x1, x2):
        numerator = torch.einsum('ij,ij->i', x1, x2)
        denominator = torch.norm(x1, dim=1) * torch.norm(x2, dim=1)
        return numerator / denominator

    def payoff_function(self, miRNA_embedding, disease_embedding):
        return self.nikolov_inner_product(miRNA_embedding, disease_embedding)

    def nash_loss(self, predicted_payoff, true_labels):
        """
        Nash loss based on the difference between predicted and true labels
        """
        return F.binary_cross_entropy_with_logits(predicted_payoff, true_labels)

    def calculate_greedy_strategy(self, strategies, rewards, miRNA_index, disease_index, num_miRNAs, num_diseases):
        """Calculates the greedy best strategy for each player."""
        num_pairs = strategies.shape[0]
        best_strategies = strategies.clone().detach()

        # 构建收益矩阵
        payoff_matrix = torch.zeros(num_miRNAs, num_diseases, dtype=rewards.dtype, device=rewards.device)
        payoff_matrix[miRNA_index, disease_index] = rewards

        # 找到每个 miRNA 的最优策略
        best_indices = torch.argmax(payoff_matrix, dim=1)

        for i in range(num_pairs):
            best_strategies[i] = strategies[best_indices[miRNA_index[i]]].clone()
        return best_strategies

    def forward(self, miRNA_embeddings, disease_embeddings, miRNA_index, disease_index, true_labels):
        """
        :param miRNA_embeddings: the embeddings of all miRNAs
        :param disease_embeddings: the embeddings of all diseases
        :param miRNA_index: current miRNA index (shape: [batch_size])
        :param disease_index: current disease index (shape: [batch_size])
        :param true_labels: true miRNA-disease interaction labels (shape: [batch_size])
        """
        num_miRNAs = miRNA_embeddings.shape[0]
        num_diseases = disease_embeddings.shape[0]
        miRNA_embedding = torch.gather(miRNA_embeddings, 0,
                                       miRNA_index.unsqueeze(1).expand(-1, miRNA_embeddings.shape[1]))
        disease_embedding = torch.gather(disease_embeddings, 0,
                                         disease_index.unsqueeze(1).expand(-1, disease_embeddings.shape[1]))

        miRNA_embedding = self.miRNA_layer(miRNA_embedding)
        disease_embedding = self.disease_layer(disease_embedding)

        miRNA_strategies = self.miRNA_strategy_net(miRNA_embedding)
        disease_strategies = self.disease_strategy_net(disease_embedding)

        predicted_payoff = self.payoff_function(miRNA_strategies, disease_strategies)

        # 使用贪心算法近似最优策略
        rewards = predicted_payoff
        best_miRNA_strategies = self.calculate_greedy_strategy(miRNA_strategies, rewards, miRNA_index, disease_index,
                                                               num_miRNAs, num_diseases)
        best_disease_strategies = self.calculate_greedy_strategy(disease_strategies, rewards, miRNA_index,
                                                                 disease_index, num_miRNAs, num_diseases)

        # 计算纳什均衡损失
        nash_loss_miRNA = torch.mean((miRNA_strategies - best_miRNA_strategies) ** 2)
        nash_loss_disease = torch.mean((disease_strategies - best_disease_strategies) ** 2)
        nash_loss = (nash_loss_miRNA + nash_loss_disease) / 2

        # 计算标签损失
        label_loss = self.nash_loss(predicted_payoff, true_labels)

        # 总损失
        loss = nash_loss + label_loss

        return predicted_payoff, loss


# 示例代码
num_miRNA_features = 64
num_disease_features = 64
hidden_dim = 32
strategy_dim = 16
label_type = 'classification'  # 标签类型是分类

model = GameTheory(num_miRNA_features, num_disease_features, hidden_dim, strategy_dim, label_type)
miRNA_embeddings = torch.randn(495, num_miRNA_features)
disease_embeddings = torch.randn(383, num_disease_features)
miRNA_index = torch.randint(0, 495, (8000,))
disease_index = torch.randint(0, 383, (8000,))
true_labels = torch.randint(0, 2, (8000,)).float()  # 模拟二分类标签

predicted_payoff, loss = model(miRNA_embeddings, disease_embeddings, miRNA_index, disease_index, true_labels)
print(loss)
