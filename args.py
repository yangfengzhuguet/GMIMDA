import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--parent_dir', type=str, default='data/miR2Disease', help='The parent_dir of os.path')
    parser.add_argument('--parent_dir_', type=str, default='data/miR2Disease/ten-folds_balance', help='The parent_dir_sub of os.path')

    parser.add_argument('--lr', type=float, default=0.001, help='The learning rate')
    parser.add_argument('--we_decay', type=float, default=1e-5, help='The weight decay')
    parser.add_argument('--epoch', type=int, default=70, help='The train epoch')

    return parser.parse_args()

