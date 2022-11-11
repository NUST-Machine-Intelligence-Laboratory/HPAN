
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import resnet34, ResNet34_Weights
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import resnet101, ResNet101_Weights
from torchvision.models import resnet152, ResNet152_Weights
import time
from thop import profile


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


def get_prior_mask(q_feat: torch.Tensor, s_feat: torch.Tensor,
                   s_mask: torch.Tensor, batch) -> torch.Tensor:
    cos_esp = 1e-8
    q_frame, c, h, w = q_feat.shape
    s_frame, _, _, _ = s_feat.shape
    q_num = q_frame // batch
    s_num = s_frame // batch

    s_mask = F.interpolate(s_mask, (h, w), mode='bilinear', align_corners=True)
    s_feat = s_feat * s_mask

    q_feat = q_feat.reshape(batch, q_num, c, h, w)
    q_feat = q_feat.permute(0, 1, 3, 4, 2).contiguous()
    q_feat = q_feat.reshape(batch, q_num * h * w, c)

    s_feat = s_feat.reshape(batch, s_num, c, h, w)
    s_feat = s_feat.permute(0, 1, 3, 4, 2).contiguous()
    s_feat = s_feat.reshape(batch, s_num * h * w, c)

    cos_sim = batch_cos_sim(q_feat, s_feat)

    cos_sim = torch.max(cos_sim, dim=2)[0]
    cos_sim = cos_sim.reshape(batch, q_num, h * w)

    cos_sim_min = torch.min(cos_sim, dim=2, keepdim=True)[0]
    cos_sim_max = torch.max(cos_sim, dim=2, keepdim=True)[0]
    cos_sim = (cos_sim - cos_sim_min) / (cos_sim_max - cos_sim_min + cos_esp)

    cos_sim = cos_sim.reshape(q_frame, 1, h, w)
    return cos_sim


class Kmeans_Clustering():
    def __init__(self, max_iter=20):
        self.max_iter = max_iter

    def run(self, data: torch.Tensor, centroids=None, num_cnt=5):
        self.num_cnt = num_cnt
        if centroids is None:
            centroids = self.init_centroids(data)
        else:
            assert centroids.shape[0] == self.num_cnt

        for i in range(self.max_iter):
            clusters = self.assign_clusters(data, centroids)
            centroids = self.update_centroids(data, clusters)
        return clusters, centroids

    def init_centroids(self, data: torch.Tensor):
        return data[torch.randperm(data.size(0))[:self.num_cnt]]

    def assign_clusters(self, data: torch.Tensor, centroids: torch.Tensor):
        dist = data.mm(centroids.t())
        clusters = dist.argmax(dim=1)
        return clusters

    def update_centroids(self, data: torch.Tensor, clusters: torch.Tensor):
        centroids = []
        for i in range(self.num_cnt):
            if clusters[clusters == i].shape[0] == 0:
                centroids.append(data[torch.randperm(data.size(0))[0]])
            else:
                centroids.append(data[clusters == i].mean(dim=0))
        return torch.stack(centroids)


class Encoder(nn.Module):
    def __init__(self, type='resnet50'):
        super(Encoder, self).__init__()
        if type == 'resnet18':
            resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        elif type == 'resnet34':
            resnet = resnet34(weights=ResNet34_Weights.DEFAULT)
        elif type == 'resnet50':
            resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        elif type == 'resnet101':
            resnet = resnet101(weights=ResNet101_Weights.DEFAULT)
        elif type == 'resnet152':
            resnet = resnet152(weights=ResNet152_Weights.DEFAULT)
        else:
            raise Exception('No such model')

        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu,
                                    resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, in_f):
        in_f = self.layer0(in_f)
        l1_f = self.layer1(in_f)
        l2_f = self.layer2(l1_f)
        l3_f = self.layer3(l2_f)
        l4_f = self.layer4(l3_f)

        return [l1_f, l2_f, l3_f, l4_f]


class GraphAttention(nn.Module):
    def __init__(self, h_dim=None):
        super(GraphAttention, self).__init__()
        self.with_projection = h_dim is not None
        if self.with_projection:
            self.linear_q = nn.Linear(h_dim, h_dim)
            self.linear_k = nn.Linear(h_dim, h_dim)
            self.linear_v = nn.Linear(h_dim, h_dim)

    def forward(self, q_node, k_node, v_node):
        # q_node: [batch, num_q, channel_q]
        # k_node: [batch, num_k, channel_q]
        # v_node: [batch, num_k, channel_v]
        # return: [batch, num_q, channel_v]
        assert q_node.shape[0] == k_node.shape[0] and q_node.shape[
            0] == v_node.shape[0]
        assert k_node.shape[1] == v_node.shape[1]
        assert q_node.shape[2] == k_node.shape[2]

        if self.with_projection:
            q_node = self.linear_q(q_node)
            k_node = self.linear_k(k_node)
            v_node = self.linear_v(v_node)

        cos_sim = batch_cos_sim(q_node, k_node)
        sum_sim = cos_sim.sum(dim=2, keepdim=True)
        edge_weight = cos_sim / sum_sim
        edge_feature = torch.bmm(edge_weight, v_node)
        return edge_feature


class Attention(nn.Module):
    def __init__(self, h_dim=None):
        super(Attention, self).__init__()
        self.with_projection = h_dim is not None
        if self.with_projection:
            self.linear_q = nn.Linear(h_dim, h_dim)
            self.linear_k = nn.Linear(h_dim, h_dim)
            self.linear_v = nn.Linear(h_dim, h_dim)

    def forward(self, q_token, k_token, v_token):
        # q_token: [batch, num_q, channel_q]
        # k_token: [batch, num_k, channel_q]
        # v_token: [batch, num_k, channel_v]
        # return: [batch, num_q, channel_v]
        assert q_token.shape[0] == k_token.shape[0] and q_token.shape[
            0] == v_token.shape[0], 'batch size of q, k, v must be equal'
        assert k_token.shape[1] == v_token.shape[
            1], 'token num of k and v must be equal'
        assert q_token.shape[2] == k_token.shape[
            2], 'channel of q and k must be equal'

        if self.with_projection:
            q_token = self.linear_q(q_token)
            k_token = self.linear_k(k_token)
            v_token = self.linear_v(v_token)

        qk = torch.bmm(q_token, k_token.permute(0, 2, 1))
        qk = qk / (qk.shape[1]**0.5)
        qk = F.softmax(qk, dim=2)
        out_token = torch.bmm(qk, v_token)
        return out_token


class AgentAttention(nn.Module):
    def __init__(self, h_dim=None):
        super(AgentAttention, self).__init__()
        self.with_projection = h_dim is not None
        if self.with_projection:
            self.linear_q = nn.Linear(h_dim, h_dim)
            self.linear_k = nn.Linear(h_dim, h_dim)
            self.linear_v = nn.Linear(h_dim, h_dim)

        self.as_attn = Attention()
        self.qa_attn = Attention()

    def forward(self, q_token, a_token, s_token):
        # q_token: [batch, num_q, channel]
        # a_token: [batch, num_a, channel]
        # s_token: [batch, num_s, channel]
        # return: [batch, num_q, channel]
        batch = q_token.shape[0]
        num_q = q_token.shape[1]
        num_a = a_token.shape[1]
        num_s = s_token.shape[1]
        if self.with_projection:
            a_token_q = self.linear_q(a_token).view(batch, num_a, -1)
            s_token_k = self.linear_k(s_token).view(batch, num_s, -1)
            s_token_v = self.linear_v(s_token).view(batch, num_s, -1)
            q_token_q = self.linear_q(q_token).view(batch, num_q, -1)
            a_token_k = self.linear_k(a_token).view(batch, num_a, -1)

        as_token = a_token.view(batch, num_a, -1) + self.as_attn(
            a_token_q, s_token_k, s_token_v)
        qa_token = q_token.view(batch, num_q, -1) + self.qa_attn(
            q_token_q, a_token_k, as_token)
        return qa_token


class PrototypeAgentNetwork(nn.Module):
    def __init__(self,
                 in_dim,
                 h_dim,
                 lambda_self=0.8,
                 lambda_co=0.2,
                 proto_per_frame=5,
                 with_self_attn=True):
        super(PrototypeAgentNetwork, self).__init__()
        self.lambda_self = lambda_self
        self.lambda_co = lambda_co
        self.s_proto_attn = GraphAttention(h_dim)
        self.q_proto_attn = GraphAttention(h_dim)
        self.co_proto_attn = GraphAttention(h_dim)
        self.cluster = Kmeans_Clustering()
        self.proto_per_frame = proto_per_frame

        self.h_dim = h_dim
        self.conv_q = nn.Conv2d(in_dim, h_dim, kernel_size=1)
        self.conv_s = nn.Conv2d(in_dim, h_dim, kernel_size=1)
        self.with_self_attn = with_self_attn
        self.qs_attn = AgentAttention(h_dim)
        if with_self_attn:
            self.qq_attn = AgentAttention(h_dim)

    def get_token(self, feat, mask, token_num):
        n, c, h, w = feat.shape
        feat = feat.permute(0, 2, 3, 1).contiguous()
        feat = feat.reshape(n * h * w, c)
        mask = mask.reshape(n * h * w)
        threshold = mask.mean()
        token = feat[mask >= threshold]
        if token.shape[0] < token_num:
            token_add = feat[torch.sort(
                mask, descending=True)[1][:token_num - token.shape[0]]]
            token = torch.cat([token, token_add], dim=0)

        # cluster
        if token_num > 1:
            _, token = self.cluster.run(token, num_cnt=token_num)
        else:
            token = token.mean(dim=0, keepdim=True)
        return token

    def get_query_proto(self, q_feat, q_mask, batch):
        q_frame, c, h, w = q_feat.shape
        q_num = q_frame // batch
        q_feat = q_feat.reshape(batch, q_num, c, h, w)
        q_mask = q_mask.reshape(batch, q_num, 1, h, w)
        q_token = []
        for idx in range(batch):
            feat = q_feat[idx]
            mask = q_mask[idx]
            token = self.get_token(feat,
                                   mask,
                                   token_num=self.proto_per_frame * q_num)
            q_token.append(token)
        q_token = torch.stack(q_token)
        return q_token

    def get_support_proto(self, s_feat, s_mask, batch):
        s_frame = s_feat.shape[0]
        s_num = s_frame // batch
        s_feat = s_feat.unsqueeze(1)
        s_mask = s_mask.unsqueeze(1)
        s_tokens = []
        for batch_idx in range(batch):
            s_token = []
            for frame_idx in range(s_num):
                idx = batch_idx * s_num + frame_idx
                feat = s_feat[idx]
                mask = s_mask[idx]
                token = self.get_token(feat,
                                       mask,
                                       token_num=self.proto_per_frame)
                s_token.append(token)
            s_tokens.append(torch.cat(s_token))
        s_tokens = torch.stack(s_tokens)
        return s_tokens

    def forward(self,
                q_feat,
                s_feat,
                q_mask,
                s_mask,
                batch,
                with_proto_loss=False):
        # q_feat: [q_frame, channel, h, w]
        # s_feat: [s_frame, channel, h, w]
        # q_mask: [q_frame, 1, h, w]
        # s_mask: [s_frame, 1, h, w]
        # batch: int

        q_frame, c, h, w = q_feat.shape
        s_frame, _, _, _ = s_feat.shape
        q_num = q_frame // batch
        s_num = s_frame // batch

        q_feat = q_feat * q_mask
        s_feat = s_feat * s_mask
        q_feat = self.conv_q(q_feat)
        s_feat = self.conv_s(s_feat)
        c = self.h_dim

        q_proto = self.get_query_proto(q_feat, q_mask, batch)
        s_proto = self.get_support_proto(s_feat, s_mask, batch)

        qq_proto = q_proto + self.lambda_self * self.q_proto_attn(
            q_proto, q_proto, q_proto)
        ss_proto = s_proto + self.lambda_self * self.s_proto_attn(
            s_proto, s_proto, s_proto)
        a_token = ss_proto + self.lambda_co * self.co_proto_attn(
            ss_proto, qq_proto, qq_proto)
        q_token = q_feat.reshape(batch, q_num, c, h, w)
        q_token = q_token.permute(0, 1, 3, 4, 2).contiguous()
        q_token = q_token.reshape(batch, q_num * h * w, c)

        s_token = s_feat.reshape(batch, s_num, c, h, w)
        s_token = s_token.permute(0, 1, 3, 4, 2).contiguous()
        s_token = s_token.reshape(batch, s_num * h * w, c)

        qs_token = self.qs_attn(q_token, a_token, s_token)
        qs_token = qs_token.reshape(batch, q_num, h, w, c)
        qs_token = qs_token.permute(0, 1, 4, 2, 3).contiguous()
        qs_token = qs_token.reshape(batch * q_num, c, h, w)

        if self.with_self_attn:
            qq_token = self.qq_attn(q_token, a_token, q_token)
            qq_token = qq_token.reshape(batch, q_num, h, w, c)
            qq_token = qq_token.permute(0, 1, 4, 2, 3).contiguous()
            qq_token = qq_token.reshape(batch * q_num, c, h, w)
            out_token = torch.cat([qs_token, qq_token], dim=1)
        else:
            out_token = qs_token

        if with_proto_loss:
            return out_token, a_token
        else:
            return out_token


class DomainAgentNetwork(nn.Module):
    def __init__(self, in_dim, h_dim, with_self_attn=True) -> None:
        super(DomainAgentNetwork, self).__init__()
        self.h_dim = h_dim
        self.conv_q = nn.Conv2d(in_dim, h_dim, kernel_size=1)
        self.conv_s = nn.Conv2d(in_dim, h_dim, kernel_size=1)
        self.with_self_attn = with_self_attn
        self.qs_attn = AgentAttention(h_dim)
        if with_self_attn:
            self.qq_attn = AgentAttention(h_dim)

    def forward(self, q_feat, s_feat, q_mask, s_mask, batch):
        # q_feat: [q_frame, channel, h, w]
        # s_feat: [s_frame, channel, h, w]
        # q_mask: [q_frame, 1, h, w]
        # s_mask: [s_frame, 1, h, w]
        # batch: int

        q_frame, c, h, w = q_feat.shape
        s_frame, _, _, _ = s_feat.shape
        q_num = q_frame // batch
        s_num = s_frame // batch
        agent_idx = q_num // 2

        q_feat = q_feat * q_mask
        s_feat = s_feat * s_mask
        q_feat = self.conv_q(q_feat)
        s_feat = self.conv_s(s_feat)
        c = self.h_dim

        q_token = q_feat.reshape(batch, q_num, c, h, w)
        q_token = q_token.permute(0, 1, 3, 4, 2).contiguous()
        q_token = q_token.reshape(batch, q_num, h * w, c)

        s_token = s_feat.reshape(batch, s_num, c, h, w)
        s_token = s_token.permute(0, 1, 3, 4, 2).contiguous()
        s_token = s_token.reshape(batch, s_num, h * w, c)

        a_token = q_token[:, agent_idx:agent_idx + 1]

        qs_token = self.qs_attn(q_token, a_token, s_token)
        qs_token = qs_token.reshape(batch, q_num, h, w, c)
        qs_token = qs_token.permute(0, 1, 4, 2, 3).contiguous()
        qs_token = qs_token.reshape(batch * q_num, c, h, w)

        if self.with_self_attn:
            qq_token = self.qq_attn(q_token, a_token, q_token)
            qq_token = qq_token.reshape(batch, q_num, h, w, c)
            qq_token = qq_token.permute(0, 1, 4, 2, 3).contiguous()
            qq_token = qq_token.reshape(batch * q_num, c, h, w)
            out_token = torch.cat([qs_token, qq_token], dim=1)
        else:
            out_token = qs_token

        return out_token


class ResBlock(nn.Module):
    def __init__(self, in_dim):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1)

    def forward(self, x):
        r = F.relu(self.conv1(x))
        r = F.relu(self.conv2(r))
        return x + r


class Refine(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Refine, self).__init__()
        self.convFS = nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1)
        self.ResFS = ResBlock(out_dim)
        self.ResMM = ResBlock(out_dim)

    def forward(self, f, pm):
        s = self.ResFS(self.convFS(f))
        m = s + F.interpolate(
            pm, size=s.shape[2:], mode='bilinear', align_corners=True)
        m = self.ResMM(m)
        return m


class Fusion(nn.Module):
    def __init__(self, in_dim, h_dim):
        super(Fusion, self).__init__()
        self.conv_1 = nn.Conv2d(in_dim[0], h_dim, kernel_size=1)
        self.convFS = nn.Conv2d(h_dim + in_dim[1],
                                h_dim,
                                kernel_size=3,
                                padding=1)
        self.ResMM = ResBlock(h_dim)

    def forward(self, f_1, f_2):
        f_1 = self.conv_1(f_1)
        f_x = self.convFS(torch.cat([f_1, f_2], dim=1))
        f_x = F.relu(f_x)
        f_x = self.ResMM(f_x)
        return f_x


class Decoder(nn.Module):
    def __init__(self, in_dim, h_dim):
        super(Decoder, self).__init__()
        self.fusion_3 = Fusion([in_dim[1], in_dim[0]], h_dim)
        self.RF2 = Refine(in_dim[2], h_dim)
        self.RF1 = Refine(in_dim[3], h_dim)
        self.predict = nn.Conv2d(h_dim, 1, kernel_size=3, padding=1)

    def forward(self, attn_l3, f_l3, f_l2, f_l1, out_size):
        f_l3 = self.fusion_3(f_l3, attn_l3)
        f_x = self.RF2(f_l2, f_l3)
        f_x = self.RF1(f_l1, f_x)
        pred_map = self.predict(f_x)
        pred_map = F.interpolate(pred_map,
                                 size=out_size,
                                 mode='bilinear',
                                 align_corners=True)
        return torch.sigmoid(pred_map)


class HPAN(nn.Module):
    def __init__(self,
                 backbone_type='resnet50',
                 with_prior_mask=True,
                 with_proto_attn=True,
                 proto_with_self_attn=True,
                 proto_per_frame=5,
                 with_domain_attn=True,
                 domain_with_self_attn=True):
        super(HPAN, self).__init__()
        print('with_proto_attn:', with_proto_attn)
        print('with_domain_attn:', with_domain_attn)
        assert with_proto_attn | with_domain_attn
        self.with_prior_mask = with_prior_mask
        self.with_proto_attn = with_proto_attn
        self.proto_with_self_attn = proto_with_self_attn
        self.with_domain_attn = with_domain_attn
        self.domain_with_self_attn = domain_with_self_attn
        if backbone_type == 'resnet18':
            encoder_c = [64, 128, 256, 512]
        elif backbone_type == 'resnet34':
            encoder_c = [64, 128, 256, 512]
        elif backbone_type == 'resnet50':
            encoder_c = [256, 512, 1024, 2048]
        elif backbone_type == 'resnet101':
            encoder_c = [256, 512, 1024, 2048]
        elif backbone_type == 'resnet152':
            encoder_c = [256, 512, 1024, 2048]

        h_dim = 256

        d_in_dim = 0
        if self.with_proto_attn:
            d_in_dim += 2 * h_dim if self.proto_with_self_attn else h_dim
        if self.with_domain_attn:
            d_in_dim += 2 * h_dim if self.domain_with_self_attn else h_dim
        print('decoder_in_dim:', d_in_dim)

        self.encoder = Encoder(backbone_type)
        self.proto_net = PrototypeAgentNetwork(
            encoder_c[2],
            h_dim,
            with_self_attn=self.proto_with_self_attn,
            proto_per_frame=proto_per_frame)
        self.domain_net = DomainAgentNetwork(
            encoder_c[2], h_dim, with_self_attn=self.domain_with_self_attn)
        self.decoder = Decoder(
            [d_in_dim, encoder_c[2], encoder_c[1], encoder_c[0]], h_dim)

    def forward(self,
                query_video: torch.Tensor,
                support_image: torch.Tensor,
                support_mask: torch.Tensor,
                with_prior_mask_loss=False,
                with_proto_loss=False):
        batch, q_num, in_c, in_h, in_w = query_video.shape
        _, s_num, _, _, _ = support_image.shape
        q_frame = batch * q_num
        s_frame = batch * s_num
        query_video = query_video.reshape(q_frame, in_c, in_h, in_w)
        support_image = support_image.reshape(s_frame, in_c, in_h, in_w)
        support_mask = support_mask.reshape(s_frame, 1, in_h, in_w)

        # extract low level features
        in_f = torch.cat((query_video, support_image), dim=0)
        encoder_f = self.encoder(in_f)
        q_f_l1 = encoder_f[0][:q_frame]
        q_f_l2 = encoder_f[1][:q_frame]
        q_f_l3 = encoder_f[2][:q_frame]
        q_f_l4 = encoder_f[3][:q_frame]
        s_f_l3 = encoder_f[2][q_frame:]
        s_f_l4 = encoder_f[3][q_frame:]

        _, _, l3_h, l3_w = q_f_l3.shape
        _, _, l4_h, l4_w = q_f_l4.shape

        # prior_mask
        if self.with_prior_mask:
            prior_mask = get_prior_mask(q_f_l4, s_f_l4, support_mask, batch)
            prior_mask = prior_mask.reshape(q_frame, 1, l4_h, l4_w)
        else:
            prior_mask = torch.ones(
                (q_frame, 1, l4_h, l4_w)).to(support_mask.device)

        # get l3 mask
        q_m_l3 = F.interpolate(prior_mask, (l3_h, l3_w),
                               mode='bilinear',
                               align_corners=True)
        s_m_l3 = F.interpolate(support_mask, (l3_h, l3_w),
                               mode='bilinear',
                               align_corners=True)

        if self.with_proto_attn:
            if with_proto_loss:
                p_attn_f, proto_token = self.proto_net(
                    q_f_l3,
                    s_f_l3,
                    q_m_l3,
                    s_m_l3,
                    batch,
                    with_proto_loss=with_proto_loss)
            else:
                p_attn_f = self.proto_net(q_f_l3, s_f_l3, q_m_l3, s_m_l3,
                                          batch)
        if self.with_domain_attn:
            d_attn_f = self.domain_net(q_f_l3, s_f_l3, q_m_l3, s_m_l3, batch)

        if self.with_proto_attn & self.with_domain_attn:
            attn_f_l3 = torch.cat((p_attn_f, d_attn_f), dim=1)
        elif self.with_proto_attn:
            attn_f_l3 = p_attn_f
        elif self.with_domain_attn:
            attn_f_l3 = d_attn_f

        pred_map = self.decoder(attn_f_l3, q_f_l3, q_f_l2, q_f_l1,
                                (in_h, in_w))

        pred_map = pred_map.reshape(batch, q_num, 1, in_h, in_w)

        if with_prior_mask_loss:
            prior_mask = F.interpolate(prior_mask, [in_h, in_w],
                                       mode='bilinear',
                                       align_corners=True)
            prior_mask = prior_mask.reshape(batch, q_num, 1, in_h, in_w)

        if with_prior_mask_loss & with_proto_loss:
            return pred_map, prior_mask, proto_token
        elif with_prior_mask_loss:
            return pred_map, prior_mask
        elif with_proto_loss:
            return pred_map, proto_token
        else:
            return pred_map


if __name__ == '__main__':
    backbone_type = 'resnet50'
    with_prior_mask = True
    with_proto_attn = True
    proto_with_self_attn = True
    proto_per_frame = 15
    with_domain_attn = False
    domain_with_self_attn = False
    with_prior_mask_loss = False
    with_proto_loss = False
    model = HPAN(backbone_type=backbone_type,
                 with_prior_mask=with_prior_mask,
                 with_proto_attn=with_proto_attn,
                 proto_with_self_attn=proto_with_self_attn,
                 proto_per_frame=proto_per_frame,
                 with_domain_attn=with_domain_attn,
                 domain_with_self_attn=domain_with_self_attn)
    model.cuda()
    s_num = 40
    q_num = 40
    query_video = torch.rand(1, q_num, 3, 241, 425).cuda()
    support_mask = torch.rand(1, s_num, 1, 241, 425).cuda()
    support_image = torch.rand(1, s_num, 3, 241, 425).cuda()

    total_ops, total_params = profile(model,
                                      inputs=(query_video, support_image,
                                              support_mask),
                                      verbose=False,
                                      custom_ops={'torch': torch})

    start_time = time.time()
    for i in range(100):
        with torch.no_grad():
            model(query_video, support_image, support_mask)
    cost_time = time.time() - start_time
    total_time = cost_time / 100

    print('total_ops: {:.4f}Gflops'.format(total_ops / 1e9))
    print('total_params: {:.4f}M'.format(total_params / 1e6))
    print('total_time: {:.4f}ms'.format(total_time * 1000))
