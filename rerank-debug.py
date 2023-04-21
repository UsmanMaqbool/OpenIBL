import torch
import numpy as np


__all__ = ['re_ranking']

import numpy as np
import gc

def re_ranking(q_g_dist, q_q_dist, g_g_dist, k1=20, k2=6, lambda_value=0.3):
    
    # The following naming, e.g. gallery_num, is different from outer scope.
    # Don't care about it.

    original_dist = np.concatenate(
      [np.concatenate([q_q_dist, q_g_dist], axis=1),
       np.concatenate([q_g_dist.T, g_g_dist], axis=1)],
      axis=0)
    # original_dist = torch.cat(
    #   [torch.cat([q_q_dist, q_g_dist], axis=1),
    #    torch.cat([q_g_dist.T, g_g_dist], axis=1)],
    #   axis=0)
    
    del q_q_dist, g_g_dist
    print("line 48")
    original_dist = np.power(original_dist, 2).astype(np.float32)
    # original_dist = torch.pow(original_dist, 2)
    print("line 51")
    original_dist = np.transpose(1. * original_dist/np.max(original_dist,axis = 0))
    # original_dist = torch.transpose(1. * original_dist/torch.max(original_dist,axis = 0))
    print("line 54")
    V = np.zeros_like(original_dist).astype(np.float32)
    V = np.zeros_like(original_dist).astype(np.float32)
    print("line 59")
    initial_rank = np.argsort(original_dist).astype(np.int32)
    initial_rank = np.argsort(original_dist).astype(np.int32)

    query_num = q_g_dist.shape[0]
    gallery_num = q_g_dist.shape[0] + q_g_dist.shape[1]
    all_num = gallery_num

    for i in range(all_num):
        # k-reciprocal neighbors
        forward_k_neigh_index = initial_rank[i,:k1+1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index,:k1+1]
        fi = np.where(backward_k_neigh_index==i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[candidate,:int(np.around(k1/2.))+1]
            candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index,:int(np.around(k1/2.))+1]
            fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
            if len(np.intersect1d(candidate_k_reciprocal_index,k_reciprocal_index))> 2./3*len(candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index,candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i,k_reciprocal_expansion_index])
        V[i,k_reciprocal_expansion_index] = 1.*weight/np.sum(weight)
    original_dist = original_dist[:query_num,]
    if k2 != 1:
        V_qe = np.zeros_like(V,dtype=np.float32)
        for i in range(all_num):
            V_qe[i,:] = np.mean(V[initial_rank[i,:k2],:],axis=0)
        V = V_qe
        del V_qe
    del initial_rank
    invIndex = []
    for i in range(gallery_num):
        invIndex.append(np.where(V[:,i] != 0)[0])

    jaccard_dist = np.zeros_like(original_dist,dtype = np.float32)


    for i in range(query_num):
        temp_min = np.zeros(shape=[1,gallery_num],dtype=np.float32)
        indNonZero = np.where(V[i,:] != 0)[0]
        indImages = []
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0,indImages[j]] = temp_min[0,indImages[j]]+ np.minimum(V[i,indNonZero[j]],V[indImages[j],indNonZero[j]])
        jaccard_dist[i] = 1-temp_min/(2.-temp_min)

    final_dist = jaccard_dist*(1-lambda_value) + original_dist*lambda_value
    del original_dist
    del V
    del jaccard_dist
    final_dist = final_dist[:query_num,query_num:]
    gc.collect()
    return final_dist

def re_ranking_t(q_g_dist, q_q_dist, g_g_dist, k1=20, k2=6, lambda_value=0.3):
    
    # The following naming, e.g. gallery_num, is different from outer scope.
    # Don't care about it.

    original_dist = torch.cat((torch.cat((q_q_dist, q_g_dist),1), torch.cat((q_g_dist.T, g_g_dist),1)), 0)
    
    del q_q_dist, g_g_dist
    print("line 48")
    original_dist = torch.pow(original_dist, 2)
    print("line 51")
    aa = 1. * original_dist/torch.max(original_dist,0).values
    # original_dist = np.transpose(1. * original_dist/np.max(original_dist,axis = 0))
    original_dist = torch.transpose(1. * original_dist/torch.max(original_dist,0).values, 0, 1)
    print("line 54")
    V = torch.zeros_like(original_dist)
    print("line 59")
    initial_rank = torch.argsort(original_dist)

    
    q_g_dist = q_g_dist.cpu().numpy()
    initial_rank = initial_rank.cpu().numpy()
    original_dist = original_dist.cpu().numpy()
    V = V.cpu().numpy()

    query_num = q_g_dist.shape[0]
    gallery_num = q_g_dist.shape[0] + q_g_dist.shape[1]
    all_num = gallery_num
    
    

    for i in range(all_num):
        # k-reciprocal neighbors
        forward_k_neigh_index = initial_rank[i,:k1+1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index,:k1+1]
        fi = np.where(backward_k_neigh_index==i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[candidate,:int(np.around(k1/2.))+1]
            candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index,:int(np.around(k1/2.))+1]
            fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
            if len(np.intersect1d(candidate_k_reciprocal_index,k_reciprocal_index))> 2./3*len(candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index,candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i,k_reciprocal_expansion_index])
        V[i,k_reciprocal_expansion_index] = 1.*weight/np.sum(weight)
    original_dist = original_dist[:query_num,]
    if k2 != 1:
        V_qe = np.zeros_like(V,dtype=np.float32)
        for i in range(all_num):
            V_qe[i,:] = np.mean(V[initial_rank[i,:k2],:],axis=0)
        V = V_qe
        del V_qe
    del initial_rank
    invIndex = []
    for i in range(gallery_num):
        invIndex.append(np.where(V[:,i] != 0)[0])

    jaccard_dist = np.zeros_like(original_dist,dtype = np.float32)


    for i in range(query_num):
        temp_min = np.zeros(shape=[1,gallery_num],dtype=np.float32)
        indNonZero = np.where(V[i,:] != 0)[0]
        indImages = []
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0,indImages[j]] = temp_min[0,indImages[j]]+ np.minimum(V[i,indNonZero[j]],V[indImages[j],indNonZero[j]])
        jaccard_dist[i] = 1-temp_min/(2.-temp_min)

    final_dist = jaccard_dist*(1-lambda_value) + original_dist*lambda_value
    del original_dist
    del V
    del jaccard_dist
    final_dist = final_dist[:query_num,query_num:]
    gc.collect()
    return final_dist


aa = torch.load('tensor.pt')
distmat = aa['distmat']
distmat_qq = aa['distmat_qq']
distmat_gg = aa['distmat_gg']
print("loaded")

distmat_jac = re_ranking(distmat.numpy(), distmat_qq.numpy(), distmat_gg.numpy(),
                                                    k1=20, k2=1, lambda_value=0.1)
print("done using numpy")
distmat.cuda()
distmat_qq.cuda()
distmat_gg.cuda()
distmat_jac_torch = re_ranking_t(distmat, distmat_qq, distmat_gg,
                                                    k1=20, k2=1, lambda_value=0.1)
print("done using numpy")