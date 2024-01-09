import numpy as np
import torch
import torch.nn as nn
from pointnet2 import pointnet2_utils
import torch.nn.functional as F
from torch.autograd import Variable
from util import transform_point_cloud
from chamfer_loss import *
from utils import pairwise_distance_batch, PointNet, Pointer, get_knn_index, Discriminator, feature_extractor, compute_rigid_transformation, get_keypoints
from torch import einsum
from pointnet2_ops.pointnet2_utils import furthest_point_sample, gather_operation, grouping_operation

def pdist(A, B, dist_type='L2'):
  if dist_type == 'L2':
    D2 = torch.sum((A.unsqueeze(1) - B.unsqueeze(0)).pow(2), 2)
    return torch.sqrt(D2 + 1e-7)
  elif dist_type == 'SquareL2':
    return torch.sum((A.unsqueeze(1) - B.unsqueeze(0)).pow(2), 2)
  else:
    raise NotImplementedError('Not implemented')

def square_distance(src, dst):
    """
    Calculate Squared distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm;
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def query_knn(nsample, xyz, new_xyz, include_self=True):
    """Find k-NN of new_xyz in xyz"""
    pad = 0 if include_self else 1
    sqrdists = square_distance(new_xyz, xyz)  # B, S, N
    _, idx = sqrdists.topk(nsample, largest=False)
    return idx.int()

class FeedbackTransformer(nn.Module):
    def __init__(self, in_channel, dim=256, n_knn=16, pos_hidden_dim=64, attn_hidden_multiplier=4):
        super(FeedbackTransformer, self).__init__()
        self.n_knn = n_knn

        self.pos_mlp = nn.Sequential(
            nn.Conv2d(14, pos_hidden_dim, 1),
            nn.BatchNorm2d(pos_hidden_dim),
            nn.ReLU(),
            nn.Conv2d(pos_hidden_dim, in_channel, 1)
        )

        self.attn_mlp = nn.Sequential(
            nn.Conv2d(in_channel, dim * attn_hidden_multiplier, 1),
            nn.BatchNorm2d(dim * attn_hidden_multiplier),
            nn.ReLU(),
            nn.Conv2d(dim * attn_hidden_multiplier, in_channel, 1)
        )

    def forward(self, pcd_gad, feat, pcd_feadb_gad, feat_feadb):

        b, _, num_point = pcd_gad.shape
        pcd = pcd_gad[:,0:3,:]
        pcd_feadb = pcd_feadb_gad[:,0:3,:]

        fusion_pcd = torch.cat((pcd, pcd_feadb), dim=2)
        fusion_feat = torch.cat((feat, feat_feadb), dim=2) 

        fusion_gad = torch.cat((pcd_gad, pcd_feadb_gad), dim=2) 

        key_point = pcd
        key_feat = feat
        key_gad = pcd_gad

        key_point_idx = query_knn(self.n_knn, fusion_pcd.transpose(2,1).contiguous(), key_point.transpose(2,1).contiguous(), include_self=True) 

        group_point = grouping_operation(fusion_pcd, key_point_idx) 
        group_feat = grouping_operation(fusion_feat, key_point_idx) 

        group_gad = grouping_operation(fusion_gad, key_point_idx) 
        pos_gad = key_gad.reshape((b, -1, num_point, 1)) - group_gad 

        qk_rel = key_feat.reshape((b, -1, num_point, 1)) - group_feat 
        pos_rel = key_point.reshape((b, -1, num_point, 1)) - group_point 

        pos_embedding = self.pos_mlp(pos_gad)
        sample_weight = self.attn_mlp(qk_rel + pos_embedding) 
        sample_weight = torch.softmax(sample_weight, -1) 

        group_feat = group_feat + pos_embedding  
        refined_feat = einsum('b c i j, b c i j -> b c i', sample_weight, group_feat)
        
        return refined_feat

class SVDHead(nn.Module):
    def __init__(self, args):
        super(SVDHead, self).__init__()
        self.num_keypoints = args.n_keypoints
        self.weight_function = Discriminator(args)
        self.fuse = Pointer()
        self.nn_margin = args.nn_margin
        
    def forward(self, *input):
        """
            Args:
                src: Source point clouds. Size (B, 3, N)
                tgt: target point clouds. Size (B, 3, M)
                src_embedding: Features of source point clouds. Size (B, C, N)
                tgt_embedding: Features of target point clouds. Size (B, C, M)
                src_idx: Nearest neighbor indices. Size [B * N * k]
                k: Number of nearest neighbors.
                src_knn: Coordinates of nearest neighbors. Size [B, N, K, 3]
                i: i-th iteration.
                tgt_knn: Coordinates of nearest neighbors. Size [B, M, K, 3]
                src_idx1: Nearest neighbor indices. Size [B * N * k]
                idx2:  Nearest neighbor indices. Size [B, M, k]
                k1: Number of nearest neighbors.
            Returns:
                R/t: rigid transformation.
                src_keypoints, tgt_keypoints: Selected keypoints of source and target point clouds. Size (B, 3, num_keypoint)
                src_keypoints_knn, tgt_keypoints_knn: KNN of keypoints. Size [b, 3, num_kepoints, k]
                loss_scl: Spatial Consistency loss.
        """
        src = input[0]
        tgt = input[1]
        src_embedding = input[2]
        tgt_embedding = input[3]
        src_idx = input[4]
        k = input[5]
        src_knn = input[6] # [b, n, k, 3]
        i = input[7]
        tgt_knn = input[8] # [b, n, k, 3]
        src_idx1 = input[9] # [b * n * k1]
        idx2 = input[10] #[b, m, k1]
        k1 = input[11]

        batch_size, num_dims_src, num_points = src.size()
        batch_size, _, num_points_tgt = tgt.size()
        batch_size, _, num_points = src_embedding.size()

        # Matching Matrix Generator
        distance_map = pairwise_distance_batch(src_embedding, tgt_embedding) #[b, n, m]
        scores = torch.softmax(-distance_map, dim=2) #[b, n, m]  
        
        src_knn_scores = scores.view(batch_size * num_points, -1)[src_idx1, :]
        src_knn_scores = src_knn_scores.view(batch_size, num_points, k1, num_points) # [b, n, k, m]
        src_knn_scores = pointnet2_utils.gather_operation(src_knn_scores.view(batch_size * num_points, k1, num_points),\
            idx2.view(batch_size, 1, num_points * k1).repeat(1, num_points, 1).view(batch_size * num_points, num_points * k1).int()).view(batch_size,\
                num_points, k1, num_points, k1)[:, :, 1:, :, 1:].sum(-1).sum(2) / (k1-1) 

        src_knn_scores = self.nn_margin - src_knn_scores
        refined_distance_map = torch.exp(src_knn_scores) * distance_map
        refined_matching_map = torch.softmax(-refined_distance_map, dim=2) # [b, n, m] 

        src_corr = torch.matmul(tgt, refined_matching_map.transpose(2, 1).contiguous())# [b,3,n] Eq. (4)

        # Overlap Prediction
        src_knn_corr = src_corr.transpose(2,1).contiguous().view(batch_size * num_points, -1)[src_idx, :]
        src_knn_corr = src_knn_corr.view(batch_size, num_points, k, num_dims_src)#[b, n, k, 3]

        # edge features of the pseudo target neighborhoods and the source neighborhoods 
        knn_distance = src_corr.transpose(2,1).contiguous().unsqueeze(2) - src_knn_corr #[b, n, k, 3]
        src_knn_distance = src.transpose(2,1).contiguous().unsqueeze(2) - src_knn #[b, n, k, 3]
        
        weight = self.weight_function(knn_distance, src_knn_distance)#[b, 1, n] 

        # compute rigid transformation 
        R, t = compute_rigid_transformation(src, src_corr, weight) # weighted SVD


        # choose k keypoints with highest weights
        src_topk_idx, src_keypoints, tgt_keypoints = get_keypoints(src, src_corr, weight, self.num_keypoints)


        idx_tgt_corr = torch.argmax(refined_matching_map, dim=-1).int() # [b, n]
        identity = torch.eye(num_points_tgt).cuda().unsqueeze(0).repeat(batch_size, 1, 1) # [b, m, m]
        one_hot_number = pointnet2_utils.gather_operation(identity, idx_tgt_corr) # [b, m, n]
        src_keypoints_idx = src_topk_idx.repeat(1, num_points_tgt, 1) # [b, m, num_keypoints]
        keypoints_one_hot = torch.gather(one_hot_number, dim = 2, index = src_keypoints_idx).transpose(2,1).reshape(batch_size * self.num_keypoints, num_points_tgt)
        #[b, m, num_keypoints] - [b, num_keypoints, m] - [b * num_keypoints, m]
        predicted_keypoints_scores = torch.gather(refined_matching_map.transpose(2, 1), dim = 2, index = src_keypoints_idx).transpose(2,1).reshape(batch_size * self.num_keypoints, num_points_tgt)
        loss_scl = (-torch.log(predicted_keypoints_scores + 1e-15) * keypoints_one_hot).sum(1).mean()

        # neighorhood information
        src_keypoints_idx2 = src_topk_idx.unsqueeze(-1).repeat(1, 3, 1, k) #[b, 3, num_keypoints, k]
        tgt_keypoints_knn = torch.gather(knn_distance.permute(0,3,1,2), dim = 2, index = src_keypoints_idx2) #[b, 3, num_kepoints, k]

        src_transformed = transform_point_cloud(src, R, t.view(batch_size, 3))
        src_transformed_knn_corr = src_transformed.transpose(2,1).contiguous().view(batch_size * num_points, -1)[src_idx, :]
        src_transformed_knn_corr = src_transformed_knn_corr.view(batch_size, num_points, k, num_dims_src) #[b, n, k, 3]

        knn_distance2 = src_transformed.transpose(2,1).contiguous().unsqueeze(2) - src_transformed_knn_corr #[b, n, k, 3]
        src_keypoints_knn = torch.gather(knn_distance2.permute(0,3,1,2), dim = 2, index = src_keypoints_idx2) #[b, 3, num_kepoints, k]
        return R, t.view(batch_size, 3), src_keypoints, tgt_keypoints, src_keypoints_knn, tgt_keypoints_knn, loss_scl

class LossFunction(nn.Module):
    def __init__(self, args):
        super(LossFunction, self).__init__()
        self.criterion2 = ChamferLoss()
        self.criterion = nn.MSELoss(reduction='sum')
        self.GAL = GlobalAlignLoss()
        self.margin = args.loss_margin

    def forward(self, *input):
        """
            Compute global registration loss and neighorhood consistency loss
            Args:
                src_keypoints: Keypoints of source point clouds. Size (B, 3, num_keypoint)
                tgt_keypoints: Keypoints of target point clouds. Size (B, 3, num_keypoint)
                rotation_ab: Size (B, 3, 3)
                translation_ab: Size (B, 3)
                src_keypoints_knn: [b, 3, num_kepoints, k]
                tgt_keypoints_knn: [b, 3, num_kepoints, k]
                k: Number of nearest neighbors.
                src_transformed: Transformed source point clouds. Size (B, 3, N)
                tgt: Target point clouds. Size (B, 3, M)
        """
        src_keypoints = input[0]
        tgt_keypoints = input[1]
        rotation_ab = input[2]
        translation_ab = input[3]
        src_keypoints_knn = input[4]
        tgt_keypoints_knn = input[5]
        k = input[6]
        src_transformed = input[7]
        tgt = input[8]

        batch_size = src_keypoints.size()[0]

        global_alignment_loss = self.GAL(src_transformed.permute(0, 2, 1), tgt.permute(0, 2, 1), self.margin) 
        
        transformed_srckps_forward = transform_point_cloud(src_keypoints, rotation_ab, translation_ab)
        keypoints_loss = self.criterion(transformed_srckps_forward, tgt_keypoints)
        knn_consensus_loss = self.criterion(src_keypoints_knn, tgt_keypoints_knn)
        neighborhood_consensus_loss = knn_consensus_loss/k + keypoints_loss

        return neighborhood_consensus_loss, global_alignment_loss

class LossFunction_kitti(nn.Module):
    def __init__(self, args):
        super(LossFunction_kitti, self).__init__()
        self.criterion2 = ChamferLoss()
        self.criterion = nn.MSELoss(reduction='none')
        self.GAL = GlobalAlignLoss()
        self.margin = args.loss_margin

    def forward(self, *input):
        """
            Compute global registration loss and neighorhood consistency loss
            Args:
                src_keypoints: Selected keypoints of source point clouds. Size (B, 3, num_keypoint)
                tgt_keypoints: Selected keypoints of target point clouds. Size (B, 3, num_keypoint)
                rotation_ab: Size (B, 3, 3)
                translation_ab: Size (B, 3)
                src_keypoints_knn: [b, 3, num_kepoints, k]
                tgt_keypoints_knn: [b, 3, num_kepoints, k]
                k: Number of nearest neighbors.
                src_transformed: Transformed source point clouds. Size (B, 3, N)
                tgt: Target point clouds. Size (B, 3, M)
        """
        src_keypoints = input[0]
        tgt_keypoints = input[1]
        rotation_ab = input[2]
        translation_ab = input[3]
        src_keypoints_knn = input[4]
        tgt_keypoints_knn = input[5]
        k = input[6]
        src_transformed = input[7]
        tgt = input[8]

        global_alignment_loss = self.GAL(src_transformed.permute(0, 2, 1), tgt.permute(0, 2, 1), self.margin) 
        
        transformed_srckps_forward = transform_point_cloud(src_keypoints, rotation_ab, translation_ab)
        keypoints_loss = self.criterion(transformed_srckps_forward, tgt_keypoints).sum(1).sum(1).mean()
        knn_consensus_loss = self.criterion(src_keypoints_knn, tgt_keypoints_knn).sum(1).sum(1).mean()
        neighborhood_consensus_loss = knn_consensus_loss + keypoints_loss

        return neighborhood_consensus_loss, global_alignment_loss

class RIENET(nn.Module):
    def __init__(self, args):
        super(RIENET, self).__init__()
        self.emb_nn = feature_extractor(args=args)
        self.single_point_embed = PointNet()
        self.forwards = SVDHead(args=args)
        self.iter = args.n_iters 
        if args.dataset == 'kitti':
            self.loss = LossFunction_kitti(args)
        else:
            self.loss = LossFunction(args)
        self.list_k1 = args.list_k1 # [5,5,5]
        self.list_k2 = args.list_k2 # [5,5,5]

    def forward(self, *input):
        """ 
            feature extraction.
            Args:
                src = input[0]: Source point clouds. Size [B, 3, N]
                tgt = input[1]: Target point clouds. Size [B, 3, N]
        """

        src = input[0] # [B, 3, 768]
        tgt = input[1] # [B, 3, 768]
        src_embedding = input[2]
        tgt_embedding = input[3]
        src_knn = input[4]
        tgt_knn = input[5]
        src_idx = input[6]
        tgt_idx = input[7]
        src_idx1 = input[8]
        i = input[9]
        t = input[10]
        batch_size, _, _ = src.size()

        rotation_ab_pred_i, translation_ab_pred_i, src_keypoints, tgt_keypoints, src_keypoints_knn, tgt_keypoints_knn, spatial_consistency_loss_i\
            = self.forwards(src, tgt, src_embedding, tgt_embedding, src_idx, self.list_k1[i], src_knn, i, tgt_knn,\
                src_idx1, tgt_idx, self.list_k2[i])

        neighborhood_consensus_loss_i, global_alignment_loss_i = self.loss(src_keypoints, tgt_keypoints,\
                rotation_ab_pred_i, translation_ab_pred_i, src_keypoints_knn, tgt_keypoints_knn, self.list_k2[i], src, tgt)
 
        return rotation_ab_pred_i, translation_ab_pred_i, global_alignment_loss_i, neighborhood_consensus_loss_i, spatial_consistency_loss_i
    
class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.iter = args.n_iters 
        self.time = args.time
        self.rienet = RIENET(args=args)
        self.emb_nn = feature_extractor(args=args)

        self.list_k1 = args.list_k1 # [5,5,5]
        self.list_k2 = args.list_k2 # [5,5,5]

        self.attention = FeedbackTransformer(in_channel=args.ft_in_channel, dim=64)  # in_channel=256  kitti_in_channel=512


    def forward(self, *input):
        
        tgt_feat_state = []
        rotation_ab_pred_state = []
        translation_ab_pred_state = []
        global_alignment_loss_state = []
        consensus_loss_state = []
        spatial_consistency_loss_state = []

        for t in range(self.time):
            src = input[0] # [B, 3, 768]
            tgt = input[1] # [B, 3, 768]
            batch_size, _, _ = src.size()
            rotation_ab_pred = torch.eye(3, device=src.device, dtype=torch.float32).view(1, 3, 3).repeat(batch_size, 1, 1)
            translation_ab_pred = torch.zeros(3, device=src.device, dtype=torch.float32).view(1, 3).repeat(batch_size, 1)
            global_alignment_loss, consensus_loss, spatial_consistency_loss, contrastive_overlap_loss = 0.0, 0.0, 0.0, 0.0

            tgt_feat_list = []
        
            for i in range(self.iter):

                src_embedding, src_idx, src_knn, _ = self.emb_nn(src, self.list_k1[i])
                tgt_embedding, _, tgt_knn, _ = self.emb_nn(tgt, self.list_k1[i])
                src_idx1, _ = get_knn_index(src, self.list_k2[i])
                _, tgt_idx = get_knn_index(tgt, self.list_k2[i])

                # Geometry-Aware Descriptor
                _, src_nearest = get_knn_index(src, 2) # [B, 768, 2]
                _, tgt_nearest = get_knn_index(tgt, 2) # [B, 768, 2]
                src_nearest_1 = src_nearest[:, :, 0] # [B, 768]
                src_nearest_2 = src_nearest[:, :, 1] # [B, 768]
                tgt_nearest_1 = tgt_nearest[:, :, 0] # [B, 768]
                tgt_nearest_2 = tgt_nearest[:, :, 1] # [B, 768]
                src_nearest_1_xyz = torch.gather(src, 2, src_nearest_1.unsqueeze(1).repeat(1, 3, 1)) # [B, 3, 768]
                src_nearest_2_xyz = torch.gather(src, 2, src_nearest_2.unsqueeze(1).repeat(1, 3, 1)) # [B, 3, 768]
                tgt_nearest_1_xyz = torch.gather(tgt, 2, tgt_nearest_1.unsqueeze(1).repeat(1, 3, 1)) # [B, 3, 768]
                tgt_nearest_2_xyz = torch.gather(tgt, 2, tgt_nearest_2.unsqueeze(1).repeat(1, 3, 1)) # [B, 3, 768]
                src_edge1 = src - src_nearest_1_xyz # [B, 3, 768]
                src_edge2 = src - src_nearest_2_xyz # [B, 3, 768]
                tgt_edge1 = tgt - tgt_nearest_1_xyz # [B, 3, 768]
                tgt_edge2 = tgt - tgt_nearest_2_xyz # [B, 3, 768]
                src_edge1_length = torch.norm(src_edge1, dim=1, keepdim=True) # [B, 1, 768]
                src_edge2_length = torch.norm(src_edge2, dim=1, keepdim=True) # [B, 1, 768]
                tgt_edge1_length = torch.norm(tgt_edge1, dim=1, keepdim=True) # [B, 1, 768]
                tgt_edge2_length = torch.norm(tgt_edge2, dim=1, keepdim=True) # [B, 1, 768]
                src_normal = torch.cross(src_edge1, src_edge2, dim=1) # [B, 3, 768]
                tgt_normal = torch.cross(tgt_edge1, tgt_edge2, dim=1) # [B, 3, 768]
                
                
                src_gad = torch.cat((src, src_edge1, src_edge2, src_edge1_length, src_edge2_length, src_normal), dim=1) # [B, 14, 768]
                tgt_gad = torch.cat((tgt, tgt_edge1, tgt_edge2, tgt_edge1_length, tgt_edge2_length, tgt_normal), dim=1)


                if t == 0: 
                    src_embedding = self.attention(src_gad, src_embedding, tgt_gad, tgt_embedding)
                    tgt_embedding = self.attention(tgt_gad, tgt_embedding, src_gad, src_embedding)
                    tgt_feat_list.append(tgt_embedding)

                else:
                    src_embedding = self.attention(src_gad, src_embedding, tgt_gad, tgt_feat_state[t-1][i])
                    tgt_embedding = self.attention(tgt_gad, tgt_feat_state[t-1][i], src_gad, src_embedding)
                    tgt_feat_list.append(tgt_embedding)

                rotation_ab_pred_i, translation_ab_pred_i, global_alignment_loss_i, neighborhood_consensus_loss_i, spatial_consistency_loss_i = self.rienet(src, tgt, src_embedding, tgt_embedding, src_knn, tgt_knn, src_idx, tgt_idx, src_idx1, i, t)

                rotation_ab_pred = torch.matmul(rotation_ab_pred_i, rotation_ab_pred)
                translation_ab_pred = torch.matmul(rotation_ab_pred_i, translation_ab_pred.unsqueeze(2)).squeeze(2) \
                                    + translation_ab_pred_i

                src = transform_point_cloud(src, rotation_ab_pred_i, translation_ab_pred_i)

                global_alignment_loss += global_alignment_loss_i
                consensus_loss += neighborhood_consensus_loss_i
                spatial_consistency_loss += spatial_consistency_loss_i
            
            tgt_feat_state.append(tgt_feat_list)
            
            rotation_ab_pred_state.append(rotation_ab_pred)
            translation_ab_pred_state.append(translation_ab_pred)
            global_alignment_loss_state.append(global_alignment_loss)
            consensus_loss_state.append(consensus_loss)
            spatial_consistency_loss_state.append(spatial_consistency_loss)

        return rotation_ab_pred_state[-1], translation_ab_pred_state[-1], (sum(global_alignment_loss_state))/self.time, (sum(consensus_loss_state))/self.time, (sum(spatial_consistency_loss_state))/self.time
