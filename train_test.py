import torch
import torch.nn.functional as F
from utils import measure
import dhg

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 一个batch的训练，输入一个batch的训练超边和训练标签， 两个模型，基础超图, 输入节点初始特征
def train_batch(edges, labels, rep_model, agg_model, lr, Hg, X, agg_mode, weight_decay):
    rep_model.train()
    agg_model.train()
    optimizer = torch.optim.Adam(list(rep_model.parameters()) + list(agg_model.parameters()), lr=lr,  weight_decay=weight_decay)
    optimizer.zero_grad()

    # Hg_batch = dhg.Hypergraph(num_v=num_v, e_list=edges, merge_op='sum', device=device)
    # H_T = Hg_batch.H_T
    # embeddings = torch.sparse.mm(H_T, adj)
    # edge_embeddings = embeddings.to_dense().to(device)

    final_embeddings = rep_model(X, Hg)
    embeddings = []
    for edge in edges:
        embedding = final_embeddings[list(edge), :]
        embeddings.append(embedding)

    preds = agg_model(embeddings, agg_mode)
    labels = labels.to(torch.long)
    # 计算交叉熵损失
    loss = F.cross_entropy(preds, labels)
    loss.backward()
    optimizer.step()
    return loss.item()

def test(edges, labels, Hg_train_pos, rep_model, agg_model, X, agg_mode):
    rep_model.eval()
    agg_model.eval()

    with torch.no_grad():
        final_embeddings = rep_model(X, Hg_train_pos)
        embeddings = []
        for edge in edges:
            embedding = final_embeddings[list(edge), :]
            embeddings.append(embedding)

        preds = torch.sigmoid(agg_model(embeddings, agg_mode))
        labels = torch.tensor(labels, dtype=torch.long).to(device)

        auc_roc, ap, accuracy, f1 = measure(labels, preds[:, 1])

        TP = 0
        TN = 0
        FP = 0
        FN = 0
        predictions = (preds[:, 1].clone().detach() >= 0.5).int()
        for index in range(len(predictions)):
            prediction = predictions[index]
            test = labels[index]
            if (prediction == 1 and test == 1):
                TP = TP + 1
            if (prediction == 1 and test == 0):
                FP = FP + 1
            if (prediction == 0 and test == 1):
                FN = FN + 1
            if (prediction == 0 and test == 0):
                TN = TN + 1

        return auc_roc, ap, accuracy, f1, TP, TN, FP, FN