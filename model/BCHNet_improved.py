r""" Improved BCHNet Framework for CD-FSS """

from functools import reduce
from operator import add

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import resnet
from torchvision.models import vgg
from .base.feature import extract_feat_vgg, extract_feat_res
from .base.correlation import Correlation
from .learner import HPNLearner
from .base.enhanced_frequency_filter import EnhancedFrequencyFilter, AdaptiveFrequencyFilter
from .base.attention import (
    SpatialChannelAttention, 
    CrossAttention, 
    AdaptivePrototypeRefinement,
    PyramidPooling
)
from .base.losses import CombinedLoss, PrototypeConsistencyLoss


class ImprovedBCHNetwork(nn.Module):
    def __init__(self, backbone, use_enhanced_filter=True, use_attention=True):
        super(ImprovedBCHNetwork, self).__init__()

        # 1. Backbone network initialization
        self.backbone_type = backbone
        self.use_enhanced_filter = use_enhanced_filter
        self.use_attention = use_attention

        if backbone == 'resnet50':
            self.backbone = resnet.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
            self.feat_ids = list(range(4, 17))
            self.extract_feats = extract_feat_res
            nbottlenecks = [3, 4, 6, 3]

        elif backbone == 'vgg16':
            print("Sorry. You need to redesign it by yourself.")

        else:
            raise Exception('Unavailable backbone: %s' % backbone)

        self.bottleneck_ids = reduce(add, list(map(lambda x: list(range(x)), nbottlenecks)))
        self.lids = reduce(add, [[i + 1] * x for i, x in enumerate(nbottlenecks)])
        self.stack_ids = torch.tensor(self.lids).bincount().__reversed__().cumsum(dim=0)[:3]
        self.backbone.eval()
        self.hpn_learner = HPNLearner(list(reversed(nbottlenecks[-3:])))
        
        # Enhanced loss function
        self.combined_loss = CombinedLoss(
            ce_weight=1.0,
            dice_weight=0.5,
            edge_weight=0.3,
            consistency_weight=0.2
        )
        self.proto_consistency_loss = PrototypeConsistencyLoss()

        # Enhanced Frequency domain filter
        if use_enhanced_filter:
            self.FDF = EnhancedFrequencyFilter(shape=(1, 2048, 13, 13))
        else:
            self.FDF = AdaptiveFrequencyFilter(shape=(1, 2048, 13, 13))
        
        # Attention modules
        if use_attention:
            self.query_attention = SpatialChannelAttention(2048, reduction=16)
            self.support_attention = SpatialChannelAttention(2048, reduction=16)
            self.cross_attention = CrossAttention(dim=2048, num_heads=8)
            self.prototype_refiner = AdaptivePrototypeRefinement(2048)
            self.pyramid_pooling = PyramidPooling(2048, pool_sizes=[1, 2, 3, 6])

    def forward(self, query_img, support_img, support_mask):

        # ################################ Features Extracting #########################################################
        with torch.no_grad():
            query_feats = self.extract_feats(query_img, self.backbone, self.feat_ids, self.bottleneck_ids, self.lids)
            support_feats = self.extract_feats(support_img, self.backbone, self.feat_ids, self.bottleneck_ids, self.lids)
        
        # Apply attention to high-level features
        if self.use_attention:
            query_feats[-1] = self.query_attention(query_feats[-1])
            support_feats[-1] = self.support_attention(support_feats[-1])
        
        fg_support_feats, prototypes_f, prototypes_b = self.mask_feature(support_feats, support_mask.clone())
        # ################################ Features Extracting #########################################################

        # ################################ Query Self-similarity Matching ##############################################
        sim_mat = self.similarity(query_feats, prototypes_f, prototypes_b)
        P_f1, P_b2 = self.SSP_func(query_feats, sim_mat)
        M_self = self.self_similarity(query_feats, P_f1)
        Pf_self = self.masked_average_pooling(query_feats, M_self)
        mask_rough = self.query_similarity_func(query_feats, Pf_self, P_b2)
        M_rough = []
        for idx, mask_f in enumerate(mask_rough):
            mask_b = 1 - mask_f
            cat_mask = torch.cat((mask_b.unsqueeze(1), mask_f.unsqueeze(1)), dim=1)
            M_rough.append(cat_mask)
        fg_query_feats = self.mask_query_feature(query_feats, mask_rough)
        # ############################### Query Self-similar Matching ##################################################

        # ################################ Cross Attention Enhancement #################################################
        if self.use_attention:
            # Apply cross attention between query and support
            query_feats[-1] = query_feats[-1] + self.cross_attention(query_feats[-1], support_feats[-1])
            
            # Refine prototypes with adaptive mechanism
            fg_support_feats[-1] = self.prototype_refiner(fg_support_feats[-1], prototypes_f[-1])
            
            # Apply pyramid pooling for multi-scale context
            query_feats[-1] = self.pyramid_pooling(query_feats[-1])
        # ################################ Cross Attention Enhancement #################################################

        # ################################ Frequency Domain Filter #####################################################
        query_feats = self.filter(query_feats)
        fg_support_feats = self.filter(fg_support_feats)
        support_feats = self.filter(support_feats)
        fg_query_feats = self.filter(fg_query_feats)
        # ################################ Frequency Domain Filter ####################################################

        # ############################### Bidirectional Hypercorrelation Construction ##################################
        corr_1 = Correlation.multilayer_correlation(query_feats, fg_support_feats, self.stack_ids)
        corr_2 = Correlation.multilayer_correlation(support_feats, fg_query_feats, self.stack_ids)
        logit_mask_1 = self.hpn_learner(corr_1)
        logit_mask_1 = F.interpolate(logit_mask_1, support_img.size()[2:], mode='bilinear', align_corners=True)
        logit_mask_2 = self.hpn_learner(corr_2)
        logit_mask_2 = F.interpolate(logit_mask_2, support_img.size()[2:], mode='bilinear', align_corners=True)
        # ############################### Bidirectional Hypercorrelation Construction ##################################

        return logit_mask_1, logit_mask_2, M_rough

    def mask_feature(self, features, support_mask):

        eps = 1e-6
        prototypes_f = []
        prototypes_b = []
        fg_features = []
        for idx, feature in enumerate(features):

            mask = F.interpolate(support_mask.unsqueeze(1).float(), feature.size()[2:], mode='bilinear', align_corners=True)
            bg_mask = 1 - mask

            feature_f = feature * mask
            feature_b = feature * bg_mask

            proto_f = feature_f.sum((2, 3))
            label_sum = mask.sum((2, 3))
            proto_f = proto_f / (label_sum + eps)
            prototypes_f.append(proto_f)

            proto_b = feature_b.sum((2, 3))
            label_sum = bg_mask.sum((2, 3))
            proto_b = proto_b / (label_sum + eps)
            prototypes_b.append(proto_b)

            fg_features.append(feature_f)

        return fg_features, prototypes_f, prototypes_b

    def similarity(self, features, fg_proto, bg_proto):

        out = []
        for idx, feature in enumerate(features):
            similarity_fg = F.cosine_similarity(features[idx], fg_proto[idx].unsqueeze(2).unsqueeze(3), dim=1)
            similarity_bg = F.cosine_similarity(features[idx], bg_proto[idx].unsqueeze(2).unsqueeze(3), dim=1)
            out_cat = torch.cat((similarity_bg[:, None, ...], similarity_fg[:, None, ...]), dim=1) * 10.0
            out.append(out_cat)

        return out

    def SSP_func(self, features, sim_mat):

        Pq_f = []
        Pq_sspb = []
        for i, feature_q in enumerate(features):

            bs = features[i].shape[0]
            ch = features[i].shape[1]

            pred_1 = sim_mat[i].softmax(1)
            pred_1 = pred_1.view(bs, 2, -1)

            pred_fg = pred_1[:, 1]
            pred_bg = pred_1[:, 0]

            fg_ls = []
            bg_local_ls = []

            for epi in range(bs):
                fg_thres = 0.7
                bg_thres = 0.6
                cur_feat = feature_q[epi].view(ch, -1)
                f_h, f_w = feature_q[epi].shape[-2:]

                if i <= 3:
                    k = 12
                elif i <= 9:
                    k = 6
                elif i <= 12:
                    k = 3

                if (pred_fg[epi] > fg_thres).sum() > 0:
                    fg_feat = cur_feat[:, (pred_fg[epi] > fg_thres)]
                else:
                    fg_feat = cur_feat[:, torch.topk(pred_fg[epi], k).indices]

                if (pred_bg[epi] > bg_thres).sum() > 0:
                    bg_feat = cur_feat[:, (pred_bg[epi] > bg_thres)]
                else:
                    bg_feat = cur_feat[:, torch.topk(pred_bg[epi], k).indices]

                fg_proto = fg_feat.mean(-1)
                fg_ls.append(fg_proto.unsqueeze(0))

                bg_feat_norm = bg_feat / torch.norm(bg_feat, 2, 0, True)
                cur_feat_norm = cur_feat / torch.norm(cur_feat, 2, 0, True)
                cur_feat_norm_t = cur_feat_norm.t()
                bg_sim = torch.matmul(cur_feat_norm_t, bg_feat_norm) * 2.0
                bg_sim = bg_sim.softmax(-1)
                bg_proto_local = torch.matmul(bg_sim, bg_feat.t())
                bg_proto_local = bg_proto_local.t().view(ch, f_h, f_w).unsqueeze(0)
                bg_local_ls.append(bg_proto_local)

            Pq_f_i = torch.cat(fg_ls, 0).unsqueeze(-1).unsqueeze(-1)
            Pq_sspb_i = torch.cat(bg_local_ls, 0)
            Pq_f.append(Pq_f_i)
            Pq_sspb.append(Pq_sspb_i)

        return Pq_f, Pq_sspb

    def self_similarity(self, features, fg_proto):

        M_self =[]
        for i, feature_q in enumerate(features):
            similarity_fg = F.cosine_similarity(feature_q, fg_proto[i], dim=1)
            threshold = 0.6
            foreground_mask = (similarity_fg >= threshold).float()
            M_self_fg = foreground_mask
            M_self_bg = 1 - foreground_mask
            M_self_i = torch.cat((M_self_bg.unsqueeze(1), M_self_fg.unsqueeze(1)), dim=1)
            M_self.append(M_self_i)

        return M_self

    def masked_average_pooling(self, features, masks):

        prototypes_self_f = []
        eps = 1e-6
        for idx, mask in enumerate(masks):
            pred_mask = mask.argmax(dim=1)
            pred_mask = pred_mask.unsqueeze(1).float()
            fg_feature = features[idx] * pred_mask
            proto_f = fg_feature.sum((2, 3))
            label_sum = pred_mask.sum((2, 3))
            proto_f = proto_f / (label_sum + eps)
            prototypes_self_f.append(proto_f.unsqueeze(-1).unsqueeze(-1))
        return prototypes_self_f

    def query_similarity_func(self, features, fg_proto, bg_proto):

        out = []
        for idx, feature in enumerate(features):
            similarity_fg = F.cosine_similarity(features[idx], fg_proto[idx], dim=1)
            similarity_bg = F.cosine_similarity(features[idx], bg_proto[idx], dim=1)
            out_cat = torch.cat((similarity_bg.unsqueeze(1), similarity_fg.unsqueeze(1)), dim=1) * 10.0
            out_cat = out_cat.argmax(dim=1)
            out.append(out_cat)

        return out

    def mask_query_feature(self, features, rough_masks):

        fg_features = []

        for idx, feature in enumerate(features):
            mask = rough_masks[idx].unsqueeze(1)
            feature_f = feature * mask
            fg_features.append(feature_f)

        return fg_features

    def filter(self, feats):
        apm_feats = []
        for idx, feature in enumerate(feats):
            if idx <= 9:
                apm_feats.append(feature)
            elif idx <= 12:
                x = self.FDF(feature)
                apm_feats.append(x)
        return apm_feats

    def predict_mask_nshot(self, batch, nshot):
        logit_mask_agg = 0
        logit_mask_orig = []
        bg_logit_mask_orig = []

        for s_idx in range(nshot):
            logit_mask, bg_logit_mask, _ = self(batch['query_img'], batch['support_imgs'][:, s_idx], batch['support_masks'][:, s_idx])
            logit_mask_agg += logit_mask.argmax(dim=1)
            logit_mask_orig.append(logit_mask)
            bg_logit_mask_orig.append(bg_logit_mask)
            if nshot == 1: return logit_mask_agg, logit_mask_orig, bg_logit_mask_orig
        bsz = logit_mask_agg.size(0)
        max_vote = logit_mask_agg.view(bsz, -1).max(dim=1)[0]
        max_vote = torch.stack([max_vote, torch.ones_like(max_vote).long()])
        max_vote = max_vote.max(dim=0)[0].view(bsz, 1, 1)
        pred_mask = logit_mask_agg.float() / max_vote
        pred_mask[pred_mask < 0.5] = 0
        pred_mask[pred_mask >= 0.5] = 1

        return pred_mask, logit_mask_orig, bg_logit_mask_orig

    def final_predict_mask_nshot(self, batch, nshot):
        logit_mask_agg = 0
        logit_mask_orig = []

        for s_idx in range(nshot):
            logit_mask = self.final_predict(batch['query_img'], batch['support_imgs'][:, s_idx], batch['support_masks'][:, s_idx])
            logit_mask_agg += logit_mask.argmax(dim=1)
            logit_mask_orig.append(logit_mask)
            if nshot == 1: return logit_mask_agg, logit_mask_orig
        bsz = logit_mask_agg.size(0)
        max_vote = logit_mask_agg.view(bsz, -1).max(dim=1)[0]
        max_vote = torch.stack([max_vote, torch.ones_like(max_vote).long()])
        max_vote = max_vote.max(dim=0)[0].view(bsz, 1, 1)
        pred_mask = logit_mask_agg.float() / max_vote
        pred_mask[pred_mask < 0.5] = 0
        pred_mask[pred_mask >= 0.5] = 1

        return pred_mask, logit_mask_orig

    def final_predict(self, query_img, support_img, support_mask):

        with torch.no_grad():
            query_feats = self.extract_feats(query_img, self.backbone, self.feat_ids, self.bottleneck_ids, self.lids)
            support_feats = self.extract_feats(support_img, self.backbone, self.feat_ids, self.bottleneck_ids,
                                               self.lids)
        
        if self.use_attention:
            query_feats[-1] = self.query_attention(query_feats[-1])
            support_feats[-1] = self.support_attention(support_feats[-1])
        
        fg_support_feats, prototypes_f, prototypes_b = self.mask_feature(support_feats, support_mask.clone())

        query_feats = self.filter(query_feats)
        fg_support_feats = self.filter(fg_support_feats)

        corr_1 = Correlation.multilayer_correlation(query_feats, fg_support_feats, self.stack_ids)
        logit_mask_1 = self.hpn_learner(corr_1)
        logit_mask_1 = F.interpolate(logit_mask_1, support_img.size()[2:], mode='bilinear', align_corners=True)

        return logit_mask_1
    
    def predict_mask_nshot_support(self, batch, nshot):

        logit_mask_agg = 0
        logit_mask_orig = []

        for s_idx in range(nshot):
            _, logit_mask, _ = self(batch['support_imgs'][:, s_idx], batch['support_imgs'][:, s_idx], batch['support_masks'][:, s_idx])
            logit_mask_agg += logit_mask.argmax(dim=1)
            logit_mask_orig.append(logit_mask)
            if nshot == 1: return logit_mask_agg, logit_mask_orig
        bsz = logit_mask_agg.size(0)
        max_vote = logit_mask_agg.view(bsz, -1).max(dim=1)[0]
        max_vote = torch.stack([max_vote, torch.ones_like(max_vote).long()])
        max_vote = max_vote.max(dim=0)[0].view(bsz, 1, 1)
        pred_mask = logit_mask_agg.float() / max_vote
        pred_mask[pred_mask < 0.5] = 0
        pred_mask[pred_mask >= 0.5] = 1

        return pred_mask, logit_mask_orig

    def compute_objective(self, logit_mask, gt_mask, logit_mask_2=None):
        """Enhanced loss computation with multiple components"""
        bsz = logit_mask.size(0)
        
        # Resize gt_mask if needed
        if logit_mask.shape[2:] != gt_mask.shape[1:]:
            gt_mask_resized = F.interpolate(
                gt_mask.unsqueeze(1).float(), 
                logit_mask.shape[2:], 
                mode='nearest'
            ).squeeze(1).long()
        else:
            gt_mask_resized = gt_mask.long()
        
        # Use combined loss
        total_loss, loss_dict = self.combined_loss(logit_mask, gt_mask_resized, logit_mask_2)
        
        return total_loss
    
    def compute_objective_finetuning(self, logit_mask, gt_mask, nshot):
        loss = 0.0
        for idx in range(nshot):
            bsz = gt_mask.shape[0]
            gt_resized = F.interpolate(
                gt_mask[:, idx].unsqueeze(1).float(),
                logit_mask[idx].shape[2:],
                mode='nearest'
            ).squeeze(1).long()
            loss += self.compute_objective(logit_mask[idx], gt_resized)
        return loss/nshot

    def train_mode(self):
        self.train()
        self.backbone.eval()

    def pred_mask_loss(self, pred_mask, gt):
        loss =  0.0
        for mask in pred_mask:
            gt_mask = F.interpolate(gt.unsqueeze(1).float(), mask.size()[2:], mode='bilinear', align_corners=True)
            loss = loss + self.compute_objective(mask.float(), gt_mask.squeeze(1))
        loss = loss / 13.0
        return loss
