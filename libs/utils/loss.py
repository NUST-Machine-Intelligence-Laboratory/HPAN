import torch
import torch.nn.functional as F


def batch_cos_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # calculate cosine similarity between a and b
    # a: [batch,num_a,channel]
    # b: [batch,num_b,channel]
    # return: [batch,num_a,num_b]
    assert a.shape[0] == b.shape[0], 'batch size of a and b must be equal'
    assert a.shape[2] == b.shape[2], 'channel of a and b must be equal'
    cos_esp = 1e-8
    a_norm = a.norm(dim=2, keepdim=True)
    b_norm = b.norm(dim=2, keepdim=True)
    cos_sim = torch.bmm(a, b.permute(0, 2, 1))
    cos_sim = cos_sim / (torch.bmm(a_norm, b_norm.permute(0, 2, 1)) + cos_esp)
    return cos_sim


def cross_entropy_loss(pred, mask):
    batch, q_num = mask.shape[0], mask.shape[1]
    pred = pred.reshape(batch * q_num, -1)
    mask = mask.reshape(batch * q_num, -1)
    loss = F.binary_cross_entropy(pred, mask)
    return loss


def mask_iou_loss(pred, mask):
    batch, q_num = mask.shape[0], mask.shape[1]
    pred = pred.reshape(batch * q_num, -1)
    mask = mask.reshape(batch * q_num, -1)
    inter = torch.minimum(pred, mask).sum()
    union = torch.maximum(pred, mask).sum()
    loss = 1 - inter / union
    return loss


def proto_dist_loss(proto_token):
    device = proto_token.device
    proto_num = proto_token.shape[1]
    negative_ = (torch.ones(proto_num) - 2 * torch.eye(proto_num)).to(device)
    cos_sim = batch_cos_sim(proto_token, proto_token)
    cos_dist = cos_sim * negative_
    loss = cos_dist.exp().mean(dim=2).log().mean()
    return loss
