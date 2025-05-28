import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--parent_dir', type=str, default='circRNA-disease_datasets', help='The parent_dir of os.path')
    parser.add_argument('--parent_dir_', type=str, default='circRNA-disease_datasets/5-fold-balance', help='The parent_dir_sub of os.path')

    parser.add_argument('--lr', type=float, default=0.001, help='The learning rate')
    parser.add_argument('--we_decay', type=float, default=1e-5, help='The weight decay')
    parser.add_argument('--epoch', type=int, default=50, help='The train epoch')

    return parser.parse_args()

