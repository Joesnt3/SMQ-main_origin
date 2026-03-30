# =============================================================================
# ms_tcn.py — ASFormer-based two-stage model
# Replaces the original MS-TCN encoder/decoder used by SMQ.
# =============================================================================

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionHelper(nn.Module):
    def __init__(self):
        super(AttentionHelper, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def scalar_dot_att(self, proj_query, proj_key, proj_val, padding_mask):
        _, c1, _ = proj_query.shape
        energy = torch.bmm(proj_query.permute(0, 2, 1), proj_key)
        attention = energy / math.sqrt(c1)
        attention = attention + torch.log(padding_mask + 1e-6)
        attention = self.softmax(attention)
        attention = attention * padding_mask
        attention = attention.permute(0, 2, 1)
        out = torch.bmm(proj_val, attention)
        return out, attention


class AttLayer(nn.Module):
    def __init__(self, q_dim, k_dim, v_dim, r1, r2, r3, stage, att_type):
        super(AttLayer, self).__init__()
        self.query_conv = nn.Conv1d(in_channels=q_dim, out_channels=q_dim // r1, kernel_size=1)
        self.key_conv = nn.Conv1d(in_channels=k_dim, out_channels=k_dim // r2, kernel_size=1)
        self.value_conv = nn.Conv1d(in_channels=v_dim, out_channels=v_dim // r3, kernel_size=1)
        self.conv_out = nn.Conv1d(in_channels=v_dim // r3, out_channels=v_dim, kernel_size=1)

        self.stage = stage
        self.att_type = att_type
        assert self.stage in ["encoder", "decoder"]
        assert self.att_type == "normal_att"

        self.att_helper = AttentionHelper()

    def forward(self, x1, x2, mask):
        query = self.query_conv(x1)
        key = self.key_conv(x1)

        if self.stage == "decoder":
            value = self.value_conv(x2)
        else:
            value = self.value_conv(x1)

        return self._normal_self_att(query, key, value, mask)

    def _normal_self_att(self, q, k, v, mask):
        bsz, _, seq_len = q.size()
        padding_mask = torch.ones((bsz, 1, seq_len), device=q.device) * mask[:, 0:1, :]
        output, _ = self.att_helper.scalar_dot_att(q, k, v, padding_mask)
        output = self.conv_out(F.relu(output))
        return output * mask[:, 0:1, :]


class ConvFeedForward(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(ConvFeedForward, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.layer(x)


class AttModule(nn.Module):
    def __init__(self, dilation, in_channels, out_channels, r1, r2, att_type, stage, alpha):
        super(AttModule, self).__init__()
        self.feed_forward = ConvFeedForward(dilation, in_channels, out_channels)
        self.instance_norm = nn.InstanceNorm1d(in_channels, track_running_stats=False)
        self.att_layer = AttLayer(in_channels, in_channels, out_channels, r1, r1, r2, stage=stage, att_type=att_type)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()
        self.alpha = alpha

    def forward(self, x, f, mask):
        out = self.feed_forward(x)
        out = self.alpha * self.att_layer(self.instance_norm(out), f, mask) + out
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out) * mask[:, 0:1, :]


class Encoder(nn.Module):
    def __init__(
        self,
        num_layers,
        r1,
        r2,
        num_f_maps,
        input_dim,
        num_classes,
        channel_masking_rate,
        att_type,
        alpha,
    ):
        super(Encoder, self).__init__()
        self.conv_1x1 = nn.Conv1d(input_dim, num_f_maps, 1)
        self.layers = nn.ModuleList(
            [AttModule(2**i, num_f_maps, num_f_maps, r1, r2, att_type, "encoder", alpha) for i in range(num_layers)]
        )

        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)
        self.dropout = nn.Dropout2d(p=channel_masking_rate)
        self.channel_masking_rate = channel_masking_rate

    def forward(self, x, mask):
        if self.channel_masking_rate > 0:
            x = x.unsqueeze(2)
            x = self.dropout(x)
            x = x.squeeze(2)

        feature = self.conv_1x1(x)
        for layer in self.layers:
            feature = layer(feature, None, mask)

        out = self.conv_out(feature) * mask[:, 0:1, :]
        return out, feature


class Decoder(nn.Module):
    def __init__(self, num_layers, r1, r2, num_f_maps, input_dim, num_classes, att_type, alpha):
        super(Decoder, self).__init__()
        self.conv_1x1 = nn.Conv1d(input_dim, num_f_maps, 1)
        self.layers = nn.ModuleList(
            [AttModule(2**i, num_f_maps, num_f_maps, r1, r2, att_type, "decoder", alpha) for i in range(num_layers)]
        )
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x, fencoder, mask):
        feature = self.conv_1x1(x)
        for layer in self.layers:
            feature = layer(feature, fencoder, mask)

        out = self.conv_out(feature) * mask[:, 0:1, :]
        return out, feature


class MultiStageModel(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, target_dim1, target_dim2):
        super(MultiStageModel, self).__init__()
        # Use ASFormer encoder/decoder blocks while keeping the old constructor/API.
        self.stage1 = Encoder(
            num_layers=num_layers,
            r1=2,
            r2=2,
            num_f_maps=num_f_maps,
            input_dim=dim,
            num_classes=target_dim1,
            channel_masking_rate=0.0,
            att_type="normal_att",
            alpha=1.0,
        )
        self.stage2 = Decoder(
            num_layers=num_layers,
            r1=2,
            r2=2,
            num_f_maps=num_f_maps,
            input_dim=target_dim1,
            num_classes=target_dim2,
            att_type="normal_att",
            alpha=1.0,
        )

    def forward(self, x, mask):
        out, encoder_feature = self.stage1(x, mask)
        out, _ = self.stage2(out, encoder_feature * mask[:, 0:1, :], mask)
        return out