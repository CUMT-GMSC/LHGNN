import argparse


def parse():
    parser = argparse.ArgumentParser()

    dataset_name = 'email-Enron'
    # dataset_name = 'email-Eu'
    # dataset_name = 'contact-high-school'
    # dataset_name = 'contact-primary-school'
    # dataset_name = 'NDC-classes'
    # dataset_name = 'congress-bills'
    parser.add_argument('--dataset_name', type=str, default=dataset_name)

    neg_sample = 'pcns'

    parser.add_argument('--neg_sample', type=str, default=neg_sample, help='负采样方法')

    hgnn_type = 'LHGNN'

    parser.add_argument('--hgnn_type', type=str, default=hgnn_type, help='选择哪个超图神经网络')

    # HNHN的参数
    # alpha在原论文中的取值是[-3,0.5]
    parser.add_argument('--edge_alpha', type=float, default=0.5, help='边度数的幂值')
    # beta在原论文中的取值是[-2.5,1]
    parser.add_argument('--node_beta', type=float, default=0.5, help='节点度数的幂值')

    # 超图神经网络的节点嵌入维度，初始的，隐藏层和输出层
    parser.add_argument('--initial_dim', type=int, default=64, help='初始嵌入维度')
    parser.add_argument('--hidden_dim', type=int, default=64, help='隐藏层嵌入维度')
    parser.add_argument('--output_dim', type=int, default=32, help='超图神经网络的输出维度')

    # 其他训练参数
    parser.add_argument('--batch_size', type=int, default=32, help='批量大小')
    parser.add_argument('--epoch', type=int, default=100, help='epoch数值')
    parser.add_argument('--patient_epoch', type=int, default=30, help='patient_epoch数值')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0, help='权重衰减')
    parser.add_argument('--drop_out', type=float, default=0, help='dropout值')

    # agg_mode = 'mean'
    # agg_mode = 'std'
    # agg_mode = 'max'
    # agg_mode = 'min'
    # agg_mode = 'attention'
    # agg_mode = 'maxmin'
    agg_mode = 'mean_std'
    # agg_mode = 'norm_maxmin'
    parser.add_argument('--agg_mode', type=str, default=agg_mode, help='节点聚合方式')

    # 如果是基于自注意力的聚合器，还有几个参数，默认值是cash里面的参数
    parser.add_argument('--n_head', type=int, default=2, help='注意力头数')

    return parser.parse_args()