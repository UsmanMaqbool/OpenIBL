from __future__ import print_function, absolute_import
import time
from collections import OrderedDict
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.metrics import pairwise_distances

import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

from .pca import PCA
from .utils.meters import AverageMeter
from .utils.rerank import re_ranking
from .utils.dist_utils import synchronize
from .utils.serialization import write_json
from .utils.data.preprocessor import Preprocessor
from .utils import to_torch

import code


def extract_cnn_feature(model, inputs, vlad=True, gpu=None):
    model.eval()
    inputs = to_torch(inputs).cuda(gpu)
    outputs = model(inputs)
    if (isinstance(outputs, list) or isinstance(outputs, tuple)):
        x_pool, x_vlad = outputs
        if vlad:
            outputs = F.normalize(x_vlad, p=2, dim=-1)
        else:
            outputs = F.normalize(x_pool, p=2, dim=-1)
    else:
        outputs = F.normalize(outputs, p=2, dim=-1)
    return outputs

def extract_features(model, data_loader, dataset, print_freq=100,
                vlad=True, pca=None, gpu=None, sync_gather=False):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    features = []

    if (pca is not None):
        pca.load(gpu=gpu)

    end = time.time()
    with torch.no_grad():
        for i, (imgs, fnames, _, _, _) in enumerate(data_loader):
            data_time.update(time.time() - end)
            outputs = extract_cnn_feature(model, imgs, vlad, gpu=gpu)
            if (pca is not None):
                outputs = pca.infer(outputs)
            outputs = outputs.data.cpu()

            features.append(outputs)
                
            batch_time.update(time.time() - end)
            end = time.time()

            if ((i + 1) % print_freq == 0 and rank==0):
                print('Extract Features: [{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      .format(i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg))

    if (pca is not None):
        del pca

    if (sync_gather):

        features_dict = OrderedDict()        

        #551x32x32768
        #dataset: 17608
        # code.interact(local=locals())
        
        chunk_start = 0
        chunk_end = 0
        for k in range(len(features)):
            l = len(features[k]) #32
            #below should run till 32
            chunk_start = chunk_end
            chunk_end = chunk_start+l
            
            # print(f"chunk start:{chunk_start} chuck end: {chunk_end} featuresize: {l}")
            for fname, output in zip(dataset[chunk_start:chunk_end], features[k]):
                features_dict[fname[0]] = output
        print("Length of features_dict:", len(features_dict))
        
    else:
        # broadcast features in sequence
        # cost more time but less GPU memory
        bc_features = torch.cat(features).cuda(gpu)
        features_dict = OrderedDict()
        for k in range(world_size):
            bc_features.data.copy_(torch.cat(features))
            if (rank==0):
                print("gathering features from rank no.{}".format(k))
            dist.broadcast(bc_features, k)
            l = bc_features.cpu().size(0)
            for fname, output in zip(dataset[k*l:(k+1)*l], bc_features.cpu()):
                features_dict[fname[0]] = output
        del bc_features, features

    return features_dict

def features_pairwise_distance(model, data_loader, sub_set, query_len, gallery_len, print_freq=100,
                vlad=True, pca=None, gpu=None, sync_gather=False, metric=None):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
                                
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    features = []
    features_dict = torch.zeros(query_len, gallery_len)
    
    q_chunk = 32*1500
    db_chunk = 32*2500
    
    q_db_pos_list = []    
    
    for i in range(0, query_len, q_chunk):
        for j in range(0, gallery_len, db_chunk):
            start_q = i
            end_q = min(i + q_chunk, query_len)
            start_db = j
            end_db = min(j + db_chunk, gallery_len)
            # print(f"Query: {start_q}:{end_q} and DB {start_db}:{end_db}")
            
            # Append a tuple with the values to q_db_pos_list
            q_db_pos_list.append((start_q, end_q, start_db, end_db,end_q-start_q+end_db-start_db))
    
    
    
    q_db_pos = 0
    mark_chunk = q_db_pos_list[q_db_pos][4]
    i_pos = (mark_chunk/32)
    extra_feature = 0
    
    start_x = 0
    start_y = 0
    x_top = 0
    k = 0
    j = 0
    
    if (pca is not None):
        pca.load(gpu=gpu)

    end = time.time()
    with torch.no_grad():
        for i, (imgs, fnames, _, _, _) in enumerate(data_loader):
            data_time.update(time.time() - end)
            outputs = extract_cnn_feature(model, imgs, vlad, gpu=gpu)
            if (pca is not None):
                outputs = pca.infer(outputs)
            outputs = outputs.data.cpu()

            features.append(outputs)
            
            if((i == i_pos-1 and i != 0) or i == len(data_loader)-1):
                print(i)
                start_q, end_q, start_db, end_db, _ = q_db_pos_list[q_db_pos]
                print(f"Query: {start_q}:{end_q} and DB {start_db}:{end_db}")

                q_chunk = end_q-start_q
                db_chunk = end_db-start_db
                features_cat = torch.cat(features)
                if(extra_feature):
                    features_cat = torch.cat((feature_rem, features_cat), dim=0)
                    print("~~~~~~~~~~~~~~Concatenating~~~~~~~~~~~~~~~~~")
                    print(f"Previous: {feature_rem.shape} Current: {features_cat.shape} ")
    
                x = features_cat[0:q_chunk]
                y = features_cat[q_chunk:q_chunk+db_chunk]
                feature_rem = features_cat[q_chunk+db_chunk:]
                print(f"~~~~~~Next:~~~~~~~{feature_rem.shape[0]}~~~~~~~~~~~~~~~~~")
                if(not feature_rem.shape[0]):
                    extra_feature = 0
                  
                features = []
                m, n = x.size(0), y.size(0)
                x = x.view(m, -1) #7416, 32768
                y = y.view(n, -1) # 10000, 32768
                if metric is not None:
                    x = metric.transform(x)
                    y = metric.transform(y)
                dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                    torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
                dist_m.addmm_(x, y.t(), beta=1, alpha=-2)
                
                # torch.Size([2048, 4096])
                features_dict[start_q:end_q, start_db:end_db] = dist_m

                # estimate next position
                q_db_pos = q_db_pos + 1
                if(q_db_pos < len(q_db_pos_list)):
                    mark_chunk = q_db_pos_list[q_db_pos][4] - feature_rem.shape[0]
                    print(f"mark_chunk {mark_chunk} =  {q_db_pos_list[q_db_pos][4]} - {feature_rem.shape[0]}")
                    if (mark_chunk%32)==0:
                        print("yes, next is fully convertable")
                        i_pos = i_pos+(mark_chunk/32)
                        print(f"new position {i_pos}")
                        
                    else:
                        i_pos = i_pos+round(mark_chunk/32)+1
                        extra_feature = mark_chunk%32
                        print(f"For next: {mark_chunk%32} need to keep")

                # code.interact(local=locals())

            batch_time.update(time.time() - end)
            end = time.time()

            if ((i + 1) % print_freq == 0 and rank==0):
                print('Extract Features: [{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      .format(i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg))
        # code.interact(local=locals())

    return features_dict

def pairwise_distance(features, query=None, gallery=None, metric=None):
    if query is None and gallery is None:
        n = len(features)
        x = torch.cat(list(features.values()))
        x = x.view(n, -1)
        if metric is not None:
            x = metric.transform(x)
        dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True) * 2
        dist_m = dist_m.expand(n, n) - 2 * torch.mm(x, x.t())
        return dist_m, None, None

    # if (dist.get_rank()==0):
        # print ("===> Start calculating pairwise distances")
        
    x = torch.cat([features[f].unsqueeze(0) for f, _, _, _ in query], 0) #7416, 32768
    y = torch.cat([features[f].unsqueeze(0) for f, _, _, _ in gallery], 0) # 10000, 32768
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1) #7416, 32768
    y = y.view(n, -1) # 10000, 32768
    if metric is not None:
        x = metric.transform(x)
        y = metric.transform(y)
    dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
           torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    # torch.Size([7416, 10000])
    dist_m.addmm_(x, y.t(), beta=1, alpha=-2)
    # torch.Size([7416, 10000])
    return dist_m

def spatial_nms(pred, db_ids, topN):
    assert(len(pred)==len(db_ids))
    pred_select = pred[:topN]
    pred_pids = [db_ids[i] for i in pred_select]
    # find unique
    seen = set()
    seen_add = seen.add
    pred_pids_unique = [i for i, x in enumerate(pred_pids) if not (x in seen or seen_add(x))]
    return [pred_select[i] for i in pred_pids_unique]

def evaluate_all(distmat, gt, gallery, recall_topk=[1, 5, 10], nms=False):
    sort_idx = np.argsort(distmat, axis=1)
    del distmat
    db_ids = [db[1] for db in gallery]

    if (dist.get_rank()==0):
        print("===> Start calculating recalls")
    correct_at_n = np.zeros(len(recall_topk))
    # len(sort_idx) = 22168
    # len(gt)       = 22016
    print(f"len(sort_idx) = {len(sort_idx)} and len(gt) = {len(gt)}")
    for qIx, pred in enumerate(sort_idx):
        if (nms):
            pred = spatial_nms(pred.tolist(), db_ids, max(recall_topk)*12)
        # code.interact(local=locals())
        for i, n in enumerate(recall_topk):
            # if in top N then also in top NN, where NN > N
            if np.any(np.in1d(pred[:n], gt[qIx])):
                correct_at_n[i:] += 1
                break
    recalls = correct_at_n / len(gt)
    del sort_idx

    if (dist.get_rank()==0):
        print('Recall Scores:')
        for i, k in enumerate(recall_topk):
            print('  top-{:<4}{:12.1%}'.format(k, recalls[i]))
    return recalls


class Evaluator(object):
    def __init__(self, model):
        super(Evaluator, self).__init__()
        self.model = model
        self.rank = dist.get_rank()

    def evaluate(self, query_loader, dataset, query, gallery, ground_truth, gallery_loader=None, \
                    vlad=True, pca=None, rerank=False, gpu=None, sync_gather=False, \
                    nms=False, rr_topk=25, lambda_value=0):
        if (gallery_loader is not None):
            features = extract_features(self.model, query_loader, query,
                                        vlad=vlad, pca=pca, gpu=gpu, sync_gather=sync_gather)
            features_db = extract_features(self.model, gallery_loader, gallery,
                                        vlad=vlad, pca=pca, gpu=gpu, sync_gather=sync_gather)
            features.update(features_db)
        else:
            features = extract_features(self.model, query_loader, dataset,
                            vlad=vlad, pca=pca, gpu=gpu, sync_gather=sync_gather)

        print(f"lenght of features 244: {len(features)}")
        print(len(query))
        print(len(gallery))
        # distmat = pairwise_distance(features, query, gallery)

        # Split features, query, and gallery into chunks
        n_query = len(query)
        n_gallery = len(gallery)

        # Define the batch size for all datasets
        batch_size = 50  # Adjust this based on available memory

        # Initialize an empty distance matrix
        distmat = np.zeros((n_query, n_gallery))

        for i in range(0, n_query, batch_size):
            query_batch = query[i:i+batch_size]

            for j in range(0, n_gallery, batch_size):
                gallery_batch = gallery[j:j+batch_size]

                # Compute distances for the current batches
                
                dist_batch = pairwise_distance(features, query_batch, gallery_batch)

                # Update the distance matrix with the computed distances
                i_start, i_end = i, min(i + batch_size, n_query)
                j_start, j_end = j, min(j + batch_size, n_gallery)
                distmat[i_start:i_end, j_start:j_end] = dist_batch

        print(f"ground truth lenght {len(ground_truth)}")

        
        recalls = evaluate_all(distmat, ground_truth, gallery, nms=nms)
        if (not rerank):
            return recalls

        if (self.rank==0):
            print('Applying re-ranking ...')
            distmat_gg = pairwise_distance(features, gallery, gallery)
            distmat_qq = pairwise_distance(features, query, query)
            distmat = re_ranking(distmat.numpy(), distmat_qq.numpy(), distmat_gg.numpy(),
                                k1=rr_topk, k2=1, lambda_value=lambda_value)

        return evaluate_all(distmat, ground_truth, gallery, nms=nms)
