r""" Hypercorrelation Squeeze Network """
from functools import reduce
from operator import add

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet
from torchvision.models import vgg

from .base.feature import extract_feat_vgg, extract_feat_res, extract_feat_chossed, extract_feat_vgg_dense
from .base.correlation import Correlation
from .learner import HPNLearner
from .base.CRM import GCCG
from .base.FEM import FEM

class FECANet(nn.Module):
    def __init__(self, backbone, use_original_imgsize):
        super(FECANet, self).__init__()

        # 1. Backbone network initialization
        self.backbone_type = backbone
        self.use_original_imgsize = use_original_imgsize

        if backbone == 'vgg16':
            self.backbone = vgg.vgg16(pretrained=True)
            self.feat_ids = [17, 19, 21, 24, 26, 28, 30]
            self.extract_feats = extract_feat_vgg
            nbottlenecks = [2, 2, 3, 3, 3, 1]

        elif backbone == 'resnet50':
            self.backbone = resnet.resnet50(pretrained=True)
            self.feat_ids = list(range(4, 17))
            self.extract_feats = extract_feat_res
            nbottlenecks = [3, 4, 6, 3]

        else:
            raise Exception('Unavailable backbone: %s' % backbone)

        self.bottleneck_ids = reduce(add, list(map(lambda x: list(range(x)), nbottlenecks)))
        self.lids = reduce(add, [[i + 1] * x for i, x in enumerate(nbottlenecks)])
        self.stack_ids = torch.tensor(self.lids).bincount().__reversed__().cumsum(dim=0)[:3]
        self.backbone.eval()
        if self.backbone_type == 'resnet50':
            self.hpn_learner = HPNLearner([4, 7, 5])  # resnet dense
        if self.backbone_type == 'vgg16':
            self.hpn_learner = HPNLearner([2, 4, 4])  # vgg dense
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.ss = GCCG(3, output_channel=64)  # dim = [201]
        dims = [512, 1024, 2048]
        vgg_dims = [256, 512, 512]

        # FEM initialization on Single GPU parallel training
        if torch.cuda.device_count() == 1:
            if self.backbone_type == 'resnet50':
                self.fem = [FEM(dims[i]).cuda() for i in range(3)]
            if self.backbone_type == 'vgg16':
                self.fem = [FEM(vgg_dims[i]).cuda() for i in range(3)]

        # FEM initialization on Multi GPU parallel training
        if torch.cuda.device_count() > 1:
            self.fem_1 = FEM(dims[0]).cuda()
            self.fem_2 = FEM(dims[1]).cuda()
            self.fem_3 = FEM(dims[2]).cuda()

    def forward(self, query_img, support_img, support_mask, history_mask_pred):
        with torch.no_grad():
            query_feats = self.extract_feats(query_img, self.backbone)
            support_feats = self.extract_feats(support_img, self.backbone)

            # intermediate feature extraction for resnet
            if self.backbone_type == 'resnet50':
                # Dense integral correlation generation for VGG
                query_feats_dense = extract_feat_chossed(query_img, self.backbone, self.feat_ids, self.bottleneck_ids, self.lids)
                support_feats_dense = extract_feat_chossed(support_img, self.backbone, self.feat_ids, self.bottleneck_ids,
                                                   self.lids)
                support_feats = self.mask_feature(support_feats, support_mask.clone())
                corr_dense = Correlation.multilayer_correlation_dense(query_feats_dense, support_feats_dense,
                                                                      self.stack_ids)

            # intermediate feature extraction for vgg
            if self.backbone_type == 'vgg16':
                # Dense integral correlation generation for VGG
                query_feats_dense = extract_feat_vgg_dense(query_img, self.backbone, self.feat_ids, self.bottleneck_ids,
                                                     self.lids)
                support_feats_dense = extract_feat_vgg_dense(support_img, self.backbone, self.feat_ids, self.bottleneck_ids,
                                                       self.lids)


                support_feats[2] = F.interpolate(support_feats[2], (13, 13), mode='bilinear', align_corners=True)
                query_feats[2] = F.interpolate(query_feats[2], (13, 13), mode='bilinear', align_corners=True)
                support_feats_dense[-1] = F.interpolate(support_feats_dense[-1], (13, 13), mode='bilinear', align_corners=True)
                query_feats_dense[-1] = F.interpolate(query_feats_dense[-1], (13, 13), mode='bilinear', align_corners=True)
                support_feats = self.mask_feature(support_feats, support_mask.clone())

                corr_dense = Correlation.multilayer_correlation_dense(query_feats_dense, support_feats_dense,
                                                                      self.stack_ids)
        # Single GPU parallel training
        if torch.cuda.device_count() == 1:
            for i in range(3):
                  query_feats[i], support_feats[i] = self.fem[i](query_feats[i], support_feats[i])

        # Multi GPU parallel training
        if torch.cuda.device_count() > 1:
            query_feats[0], support_feats[0] = self.fem_1(query_feats[0], support_feats[0])
            query_feats[1], support_feats[1] = self.fem_2(query_feats[1], support_feats[1])
            query_feats[2], support_feats[2] = self.fem_3(query_feats[2], support_feats[2])

        # Global context correlation generation
        similarity_s, similarity_q = [], []
        for i in range(len(query_feats)):
            similarity_q.append(self.ss(query_feats[i]))
            similarity_s.append(self.ss(support_feats[i]))
        corr_self_simi = Correlation.multilayer_correlation(similarity_q, similarity_s, self.stack_ids)

        # concat Dense integral correlation and Global context correlation in three layer
        for i in range(len(corr_dense)):
            corr_dense[i] = torch.cat([corr_dense[i], corr_self_simi[i]], dim=1)

        logit_mask = self.hpn_learner(corr_dense, history_mask_pred)

        return logit_mask

    def mask_feature(self, features, support_mask):
        for idx, feature in enumerate(features):
            mask = F.interpolate(support_mask.unsqueeze(1).float(), feature.size()[2:], mode='bilinear', align_corners=True)
            features[idx] = features[idx] * mask
        return features

    def predict_mask_nshot(self, batch, nshot, dataset):

        # Perform multiple prediction given (nshot) number of different support sets
        logit_mask_agg = 0
        logit_mask_avg = 0
        for s_idx in range(nshot):
            logit_mask = self(batch['query_img'], batch['support_imgs'][:, s_idx],
                              batch['support_masks'][:, s_idx], batch['history_mask'])
            if dataset.use_original_imgsize:
                # EXP2_1 update history_mask
                pred_softmax = F.softmax(logit_mask, dim=1).detach().cpu()
                for j in range(batch['query_img'].shape[0]):
                    sub_index = batch['idx'][j]
                    # pred_softmax = F.interpolate(logit_mask, pred_size, mode='bilinear', align_corners=True)
                    dataset.history_mask_list[sub_index] = pred_softmax[j]
            else:
                logit_mask = F.interpolate(logit_mask, batch['support_imgs'].size()[-2:], mode='bilinear', align_corners=True)

            logit_mask_avg += F.softmax(logit_mask, dim=1).detach().cpu()
            if self.use_original_imgsize:
                org_qry_imsize = tuple([batch['org_query_imsize'][1].item(), batch['org_query_imsize'][0].item()])
                logit_mask = F.interpolate(logit_mask, org_qry_imsize, mode='bilinear', align_corners=True)

            logit_mask_agg += logit_mask.argmax(dim=1).clone()
            if nshot == 1: return logit_mask_agg

        # Average & quantize predictions given threshold (=0.5)
        bsz = logit_mask_agg.size(0)
        max_vote = logit_mask_agg.view(bsz, -1).max(dim=1)[0]
        max_vote = torch.stack([max_vote, torch.ones_like(max_vote).long()])
        max_vote = max_vote.max(dim=0)[0].view(bsz, 1, 1)
        pred_mask = logit_mask_agg.float() / max_vote

        threshold = 0.5
        pred_mask[pred_mask < threshold] = 0
        pred_mask[pred_mask >= threshold] = 1

        return pred_mask

    def compute_objective(self, logit_mask, gt_mask):
        logit_mask = F.interpolate(logit_mask, gt_mask.size()[-2:], mode='bilinear', align_corners=True)
        bsz = logit_mask.size(0)
        logit_mask = logit_mask.view(bsz, 2, -1)
        gt_mask = gt_mask.view(bsz, -1).long()

        return self.cross_entropy_loss(logit_mask, gt_mask)

    def train_mode(self):
        self.train()
        self.backbone.eval()  # to prevent BN from learning data statistics with exponential averaging
