import torch
import torch.nn as nn
import torch.nn.functional as F


class GCCG(torch.nn.Module):
    def __init__(self, context_size=5, output_channel=128):
        super(GCCG, self).__init__()
        self.context_size = context_size
        self.pad = context_size // 2
        self.conv = torch.nn.Conv2d(
            self.context_size * self.context_size,
            output_channel * 2,
            3,
            padding=(1, 1),
            bias=True,
            padding_mode="zeros",
        )
        self.conv1 = torch.nn.Conv2d(
            output_channel * 2,
            output_channel,
            3,
            padding=(1, 1),
            bias=True,
            padding_mode="zeros",
        )
        # additional layer
        # self.conv2 = torch.nn.Conv2d(
        #     output_channel,
        #     output_channel,
        #     3,
        #     padding=(1, 1),
        #     bias=True,
        #     padding_mode="zeros",
        # )

    def self_similarity(self, feature_normalized):
        b, c, h, w = feature_normalized.size()
        feature_pad = F.pad(
            feature_normalized, (self.pad, self.pad, self.pad, self.pad), "constant", 0
        )
        output = torch.zeros(
            [self.context_size * self.context_size, b, h, w],
            dtype=feature_normalized.dtype,
            requires_grad=feature_normalized.requires_grad,
        )
        if feature_normalized.is_cuda:
            output = output.cuda(feature_normalized.get_device())
        for c in range(self.context_size):
            for r in range(self.context_size):
                output[c * self.context_size + r] = (
                    feature_pad[:, :, r : (h + r), c : (w + c)] * feature_normalized
                ).sum(1)

        output = output.transpose(0, 1).contiguous()
        return output

    def forward(self, feature):
        feature_normalized = F.normalize(feature, p=2, dim=1)
        ss = self.self_similarity(feature_normalized)

        ss1 = F.relu(self.conv(ss))
        ss2 = F.relu(self.conv1(ss1))
        output = torch.cat((ss, ss1, ss2), 1)
        return output