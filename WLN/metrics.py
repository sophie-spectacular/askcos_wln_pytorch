import torch
import torch.nn.functional as F

def wln_loss(batch_size):
    def loss(y_true, y_pred):
        flat_label = y_true.reshape(-1)
        bond_mask = (flat_label != -1).float()
        flat_label = torch.clamp(flat_label, min=0.).float()
        flat_score = y_pred.reshape(-1)
        l = F.binary_cross_entropy_with_logits(flat_score, flat_label, reduction='none')
        return torch.sum(l * bond_mask) / batch_size
    return loss

def get_batch_size(tensor):
    return tensor.shape[0]

def top_k_acc(y_true_g, y_pred, k):
    y_true = y_true_g.int()
    bond_mask = (y_true == -1).float() * 10000.0
    masked_pred = y_pred - bond_mask
    top_k = torch.topk(masked_pred, k=k, dim=-1).indices.long()

    # Gather the true labels at the top_k indices
    match = torch.gather(y_true, -1, top_k)
    match = (match == 1).sum(dim=-1).float()
    y_true_sum = (y_true == 1).float().sum(dim=-1)
    correct = (match == y_true_sum).int()
    return correct.sum().float() / correct.numel()

def top_10_acc(y_true, y_pred):
    return top_k_acc(y_true, y_pred, k=10)
def top_20_acc(y_true, y_pred):
    return top_k_acc(y_true, y_pred, k=20)
def top_100_acc(y_true, y_pred):
    return top_k_acc(y_true, y_pred, k=100)