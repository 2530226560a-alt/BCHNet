r""" Enhanced Loss Functions for BCHNet """

import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeAwareLoss(nn.Module):
    """Edge-Aware Loss for Better Boundary Segmentation"""
    
    def __init__(self, edge_weight=2.0):
        super(EdgeAwareLoss, self).__init__()
        self.edge_weight = edge_weight
        
        # Sobel kernels for edge detection
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        
    def get_edge_map(self, mask):
        """Extract edge map from mask using Sobel operator"""
        if self.sobel_x.device != mask.device:
            self.sobel_x = self.sobel_x.to(mask.device)
            self.sobel_y = self.sobel_y.to(mask.device)
        
        mask_float = mask.float().unsqueeze(1) if mask.dim() == 3 else mask.float()
        
        edge_x = F.conv2d(mask_float, self.sobel_x, padding=1)
        edge_y = F.conv2d(mask_float, self.sobel_y, padding=1)
        edge = torch.sqrt(edge_x ** 2 + edge_y ** 2)
        edge = (edge > 0.1).float()
        
        return edge.squeeze(1)
    
    def forward(self, pred, target):
        """
        pred: [B, 2, H, W] - predicted logits
        target: [B, H, W] - ground truth mask
        """
        # Standard cross entropy loss
        ce_loss = F.cross_entropy(pred, target.long(), reduction='none')
        
        # Get edge map from target
        edge_map = self.get_edge_map(target)
        
        # Weight loss by edge map
        edge_weight_map = 1.0 + self.edge_weight * edge_map
        weighted_loss = ce_loss * edge_weight_map
        
        return weighted_loss.mean()


class DiceLoss(nn.Module):
    """Dice Loss for Segmentation"""
    
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, pred, target):
        """
        pred: [B, 2, H, W] - predicted logits
        target: [B, H, W] - ground truth mask
        """
        pred_soft = F.softmax(pred, dim=1)[:, 1, :, :]  # Get foreground probability
        target_float = target.float()
        
        intersection = (pred_soft * target_float).sum(dim=(1, 2))
        union = pred_soft.sum(dim=(1, 2)) + target_float.sum(dim=(1, 2))
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        return 1.0 - dice.mean()


class FocalLoss(nn.Module):
    """Focal Loss for Handling Class Imbalance"""
    
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, pred, target):
        """
        pred: [B, 2, H, W] - predicted logits
        target: [B, H, W] - ground truth mask
        """
        ce_loss = F.cross_entropy(pred, target.long(), reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        return focal_loss.mean()


class ConsistencyLoss(nn.Module):
    """Consistency Loss for Bidirectional Predictions"""
    
    def __init__(self):
        super(ConsistencyLoss, self).__init__()
        
    def forward(self, pred1, pred2):
        """
        pred1, pred2: [B, 2, H, W] - two predictions that should be consistent
        """
        pred1_soft = F.softmax(pred1, dim=1)
        pred2_soft = F.softmax(pred2, dim=1)
        
        # KL divergence for consistency
        kl_loss = F.kl_div(
            F.log_softmax(pred1, dim=1),
            pred2_soft,
            reduction='batchmean'
        ) + F.kl_div(
            F.log_softmax(pred2, dim=1),
            pred1_soft,
            reduction='batchmean'
        )
        
        return kl_loss * 0.5


class PrototypeConsistencyLoss(nn.Module):
    """Prototype Consistency Loss for Better Prototype Learning"""
    
    def __init__(self):
        super(PrototypeConsistencyLoss, self).__init__()
        
    def forward(self, query_proto, support_proto):
        """
        Encourage consistency between query and support prototypes
        query_proto: [B, C] - query prototype
        support_proto: [B, C] - support prototype
        """
        # Normalize prototypes
        query_proto_norm = F.normalize(query_proto, p=2, dim=1)
        support_proto_norm = F.normalize(support_proto, p=2, dim=1)
        
        # Cosine similarity loss
        similarity = (query_proto_norm * support_proto_norm).sum(dim=1)
        loss = 1.0 - similarity.mean()
        
        return loss


class CombinedLoss(nn.Module):
    """Combined Loss Function with Multiple Components"""
    
    def __init__(self, 
                 ce_weight=1.0,
                 dice_weight=0.5,
                 edge_weight=0.3,
                 focal_weight=0.0,
                 consistency_weight=0.2):
        super(CombinedLoss, self).__init__()
        
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.edge_weight = edge_weight
        self.focal_weight = focal_weight
        self.consistency_weight = consistency_weight
        
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()
        self.edge_loss = EdgeAwareLoss()
        self.focal_loss = FocalLoss()
        self.consistency_loss = ConsistencyLoss()
        
    def forward(self, pred, target, pred2=None):
        """
        pred: [B, 2, H, W] - primary prediction
        target: [B, H, W] - ground truth
        pred2: [B, 2, H, W] - optional secondary prediction for consistency
        """
        total_loss = 0.0
        loss_dict = {}
        
        # Cross entropy loss
        if self.ce_weight > 0:
            ce = self.ce_loss(pred, target.long())
            total_loss += self.ce_weight * ce
            loss_dict['ce'] = ce.item()
        
        # Dice loss
        if self.dice_weight > 0:
            dice = self.dice_loss(pred, target)
            total_loss += self.dice_weight * dice
            loss_dict['dice'] = dice.item()
        
        # Edge-aware loss
        if self.edge_weight > 0:
            edge = self.edge_loss(pred, target)
            total_loss += self.edge_weight * edge
            loss_dict['edge'] = edge.item()
        
        # Focal loss
        if self.focal_weight > 0:
            focal = self.focal_loss(pred, target)
            total_loss += self.focal_weight * focal
            loss_dict['focal'] = focal.item()
        
        # Consistency loss
        if pred2 is not None and self.consistency_weight > 0:
            consistency = self.consistency_loss(pred, pred2)
            total_loss += self.consistency_weight * consistency
            loss_dict['consistency'] = consistency.item()
        
        return total_loss, loss_dict
