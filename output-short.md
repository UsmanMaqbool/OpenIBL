
## Clustering:
```sh
╰ ./scripts/cluster.sh vgg16
==========
Args:Namespace(arch='vgg16', batch_size=64, data_dir='/mnt/ssd/usman_ws/OpenIBL/examples/data', dataset='pitts', height=480, logs_dir='/mnt/ssd/usman_ws/OpenIBL/examples/../logs', num_clusters=64, print_freq=10, resume='', seed=43, width=640, workers=8)
==========
Pittsburgh dataset loaded
  subset        | # pids | # images
  ---------------------------------
  train_query   |   311  |     7320
  train_gallery |   417  |    10000
  val_query     |   319  |     7608
  val_gallery   |   417  |    10000
  test_query    |   286  |     6816
  test_gallery  |   417  |    10000
====> Extracting Descriptors
==> Batch (1/8)
==> Batch (2/8)
==> Batch (3/8)
==> Batch (4/8)
==> Batch (5/8)
==> Batch (6/8)
==> Batch (7/8)
==> Batch (8/8)
====> Clustering
====> Storing centroids (64, 512)
====> Done!
```

## Sare Ind Original Implementation

```sh
╰ ./scripts/train_baseline_dist.sh sare_ind
/home/leo/anaconda3/envs/pytorchtutorial/lib/python3.7/site-packages/torch/distributed/launch.py:186: FutureWarning: The module torch.distributed.launch is deprecated
and will be removed in future. Use torchrun.
Note that --use_env is set by default in torchrun.
If your script expects `--local_rank` argument to be set, please
change it to read from `os.environ['LOCAL_RANK']` instead. See 
https://pytorch.org/docs/stable/distributed.html#launch-utility for 
further instructions

  FutureWarning,
Use GPU: 0 for training, rank no.0 of world_size 1
==========
Args:Namespace(arch='vgg16', cache_size=1000, data_dir='/mnt/ssd/usman_ws/OpenIBL/examples/data', dataset='pitts', deterministic=False, epochs=10, eval_step=1, features=4096, gpu=0, height=480, init_dir='/mnt/ssd/usman_ws/OpenIBL/examples/../logs', iters=0, launcher='pytorch', layers='conv5', logs_dir='/media/leo/2C737A9872F69ECF/why-so-deepv2-data/pittsburgh/netvlad-run/pitts30k-vgg16/conv5-sare_ind-lr0.001-tuple1', loss_type='sare_ind', lr=0.001, margin=0.1, momentum=0.9, neg_num=10, neg_pool=1000, ngpus_per_node=1, nowhiten=False, num_clusters=64, print_freq=10, rank=0, rerank=False, resume='', scale='30k', seed=43, step_size=5, sync_gather=True, syncbn=True, tcp_port='6010', test_batch_size=32, tuple_size=1, vlad=True, weight_decay=0.001, width=640, workers=2, world_size=1)
==========
Pittsburgh dataset loaded
  subset        | # pids | # images
  ---------------------------------
  train_query   |   311  |     7320
  train_gallery |   417  |    10000
  val_query     |   319  |     7608
  val_gallery   |   417  |    10000
  test_query    |   286  |     6816
  test_gallery  |   417  |    10000
Loading centroids from /mnt/ssd/usman_ws/OpenIBL/examples/../logs/vgg16_pitts_64_desc_cen.hdf5
Test the initial model:
===> Start calculating pairwise distances
/mnt/ssd/usman_ws/OpenIBL/examples/ibl/evaluators.py:129: UserWarning: This overload of addmm_ is deprecated:
	addmm_(Number beta, Number alpha, Tensor mat1, Tensor mat2)
Consider using one of the following signatures instead:
	addmm_(Tensor mat1, Tensor mat2, *, Number beta, Number alpha) (Triggered internally at  /opt/conda/conda-bld/pytorch_1640811797118/work/torch/csrc/utils/python_arg_parser.cpp:1050.)
  dist_m.addmm_(1, -2, x, y.t())
===> Start calculating recalls
Recall Scores:
  top-1          80.0%
  top-5          93.3%
  top-10         96.0%
[W reducer.cpp:1303] Warning: find_unused_parameters=True was specified in DDP constructor, but did not find any unused parameters in the forward pass. This flag results in an extra traversal of the autograd graph every iteration,  which can adversely affect performance. If your model indeed never has any unused parameters in the forward pass, consider turning this flag off. Note that this warning may be a false positive if your model has flow control causing later iterations to have unused parameters. (function operator())
===> Start calculating pairwise distances
===> Start calculating recalls
Recall Scores:
  top-1          85.5%
  top-5          93.4%
  top-10         95.5%

 * Finished epoch   0 recall@1: 85.5%  recall@5: 93.4%  recall@10: 95.5%  best@5: 93.4% *
Recall Scores:
  top-1          82.1%
  top-5          92.1%
  top-10         94.7%

 * Finished epoch   1 recall@1: 82.1%  recall@5: 92.1%  recall@10: 94.7%  best@5: 93.4%
Recall Scores:
  top-1          79.6%
  top-5          91.3%
  top-10         93.9%

 * Finished epoch   2 recall@1: 79.6%  recall@5: 91.3%  recall@10: 93.9%  best@5: 93.4%
Recall Scores:
  top-1          73.5%
  top-5          87.2%
  top-10         90.7%

 * Finished epoch   3 recall@1: 73.5%  recall@5: 87.2%  recall@10: 90.7%  best@5: 93.4% 
Recall Scores:
  top-1          67.6%
  top-5          83.8%
  top-10         88.8%

 * Finished epoch   4 recall@1: 67.6%  recall@5: 83.8%  recall@10: 88.8%  best@5: 93.4% 
 
calculating PCA parameters...
/mnt/ssd/usman_ws/OpenIBL/examples/ibl/pca.py:53: UserWarning: torch.symeig is deprecated in favor of torch.linalg.eigh and will be removed in a future PyTorch release.
The default behavior has changed from using the upper triangular portion of the matrix by default to using the lower triangular portion.
L, _ = torch.symeig(A, upper=upper)
should be replaced with
L = torch.linalg.eigvalsh(A, UPLO='U' if upper else 'L')
and
L, V = torch.symeig(A, eigenvectors=True)
should be replaced with
L, V = torch.linalg.eigh(A, UPLO='U' if upper else 'L') (Triggered internally at  /opt/conda/conda-bld/pytorch_1640811797118/work/aten/src/ATen/native/BatchLinearAlgebra.cpp:2499.)
  L, U = torch.symeig(x2, eigenvectors=True)
================= PCA RESULT ==================
U: (32768, 4096)
lams: (4096,)
mu: (32768, 1)
Utmu: (4096, 1)
===============================================
Testing on Pitts30k-test:
load PCA parameters...
Extract Features: [10/526]	Time 0.270 (1.111)	Data 0.000 (0.853)	
Extract Features: [20/526]	Time 0.248 (0.928)	Data 0.000 (0.670)	
Extract Features: [30/526]	Time 0.264 (0.871)	Data 0.000 (0.613)	
Extract Features: [40/526]	Time 0.300 (0.850)	Data 0.000 (0.590)	
Extract Features: [50/526]	Time 0.300 (0.834)	Data 0.000 (0.571)	
Extract Features: [60/526]	Time 0.248 (0.813)	Data 0.000 (0.551)	
Extract Features: [70/526]	Time 0.287 (0.805)	Data 0.000 (0.541)	
Extract Features: [80/526]	Time 0.248 (0.804)	Data 0.000 (0.541)	
Extract Features: [90/526]	Time 0.249 (0.793)	Data 0.000 (0.530)	
Extract Features: [100/526]	Time 0.249 (0.787)	Data 0.000 (0.523)	
Extract Features: [110/526]	Time 0.248 (0.787)	Data 0.000 (0.523)	
Extract Features: [120/526]	Time 0.296 (0.787)	Data 0.000 (0.523)	
Extract Features: [130/526]	Time 0.313 (0.782)	Data 0.000 (0.518)	
Extract Features: [140/526]	Time 0.248 (0.780)	Data 0.000 (0.516)	
Extract Features: [150/526]	Time 0.250 (0.775)	Data 0.000 (0.511)	
Extract Features: [160/526]	Time 0.249 (0.776)	Data 0.000 (0.512)	
Extract Features: [170/526]	Time 0.248 (0.777)	Data 0.000 (0.513)	
Extract Features: [180/526]	Time 0.288 (0.772)	Data 0.023 (0.508)	
Extract Features: [190/526]	Time 0.413 (0.770)	Data 0.165 (0.507)	
Extract Features: [200/526]	Time 0.577 (0.768)	Data 0.330 (0.505)	
Extract Features: [210/526]	Time 0.526 (0.766)	Data 0.278 (0.502)	
Extract Features: [220/526]	Time 0.499 (0.761)	Data 0.249 (0.498)	
Extract Features: [230/526]	Time 0.678 (0.758)	Data 0.429 (0.494)	
Extract Features: [240/526]	Time 0.755 (0.754)	Data 0.507 (0.491)	
Extract Features: [250/526]	Time 0.608 (0.750)	Data 0.341 (0.486)	
Extract Features: [260/526]	Time 1.504 (0.752)	Data 1.257 (0.488)	
Extract Features: [270/526]	Time 1.191 (0.754)	Data 0.941 (0.491)	
Extract Features: [280/526]	Time 1.188 (0.756)	Data 0.892 (0.492)	
Extract Features: [290/526]	Time 1.265 (0.758)	Data 1.017 (0.495)	
Extract Features: [300/526]	Time 1.051 (0.759)	Data 0.804 (0.495)	
Extract Features: [310/526]	Time 1.136 (0.764)	Data 0.887 (0.501)	
Extract Features: [320/526]	Time 1.099 (0.765)	Data 0.807 (0.501)	
Extract Features: [330/526]	Time 1.233 (0.763)	Data 0.953 (0.499)	
Extract Features: [340/526]	Time 1.184 (0.760)	Data 0.882 (0.496)	
Extract Features: [350/526]	Time 0.911 (0.757)	Data 0.604 (0.493)	
Extract Features: [360/526]	Time 0.940 (0.752)	Data 0.685 (0.488)	
Extract Features: [370/526]	Time 0.884 (0.749)	Data 0.630 (0.484)	
Extract Features: [380/526]	Time 0.921 (0.746)	Data 0.667 (0.480)	
Extract Features: [390/526]	Time 1.150 (0.743)	Data 0.893 (0.477)	
Extract Features: [400/526]	Time 1.180 (0.743)	Data 0.924 (0.477)	
Extract Features: [410/526]	Time 0.969 (0.742)	Data 0.715 (0.476)	
Extract Features: [420/526]	Time 1.162 (0.741)	Data 0.861 (0.474)	
Extract Features: [430/526]	Time 1.129 (0.740)	Data 0.874 (0.473)	
Extract Features: [440/526]	Time 1.212 (0.739)	Data 0.921 (0.471)	
Extract Features: [450/526]	Time 1.042 (0.739)	Data 0.777 (0.471)	
Extract Features: [460/526]	Time 0.959 (0.737)	Data 0.649 (0.469)	
Extract Features: [470/526]	Time 1.115 (0.735)	Data 0.859 (0.466)	
Extract Features: [480/526]	Time 0.939 (0.732)	Data 0.679 (0.463)	
Extract Features: [490/526]	Time 0.947 (0.730)	Data 0.642 (0.460)	
Extract Features: [500/526]	Time 0.869 (0.727)	Data 0.563 (0.457)	
Extract Features: [510/526]	Time 0.817 (0.724)	Data 0.563 (0.454)	
Extract Features: [520/526]	Time 0.724 (0.722)	Data 0.437 (0.452)	
===> Start calculating pairwise distances
===> Start calculating recalls
Recall Scores:
  top-1          86.9%
  top-5          93.0%
  top-10         94.8% 
 
 
```

## Sare Ind (your implementation)

```
╰ ./scripts/train_baseline_dist.sh sare_ind
/home/leo/anaconda3/envs/pytorchtutorial/lib/python3.7/site-packages/torch/distributed/launch.py:186: FutureWarning: The module torch.distributed.launch is deprecated
and will be removed in future. Use torchrun.
Note that --use_env is set by default in torchrun.
If your script expects `--local_rank` argument to be set, please
change it to read from `os.environ['LOCAL_RANK']` instead. See 
https://pytorch.org/docs/stable/distributed.html#launch-utility for 
further instructions

  FutureWarning,
Use GPU: 0 for training, rank no.0 of world_size 1
==========
Args:Namespace(arch='vgg16', cache_size=1000, data_dir='/mnt/ssd/usman_ws/OpenIBL/examples/data', dataset='pitts', deterministic=False, epochs=10, eval_step=1, features=4096, gpu=0, height=480, init_dir='/mnt/ssd/usman_ws/OpenIBL/examples/../logs', iters=0, launcher='pytorch', layers='conv5', logs_dir='/media/leo/2C737A9872F69ECF/why-so-deepv2-data/pittsburgh/netvlad-run/pitts30k-vgg16/conv5-sare_ind-lr0.001-tuple1', loss_type='sare_ind', lr=0.001, margin=0.1, momentum=0.9, neg_num=10, neg_pool=1000, ngpus_per_node=1, nowhiten=False, num_clusters=64, print_freq=10, rank=0, rerank=False, resume='', scale='30k', seed=43, step_size=5, sync_gather=True, syncbn=True, tcp_port='6010', test_batch_size=32, tuple_size=1, vlad=True, weight_decay=0.001, width=640, workers=2, world_size=1)
==========
Pittsburgh dataset loaded
  subset        | # pids | # images
  ---------------------------------
  train_query   |   311  |     7320
  train_gallery |   417  |    10000
  val_query     |   319  |     7608
  val_gallery   |   417  |    10000
  test_query    |   286  |     6816
  test_gallery  |   417  |    10000
Loading centroids from /mnt/ssd/usman_ws/OpenIBL/examples/../logs/vgg16_pitts_64_desc_cen.hdf5
Test the initial model:
===> Start calculating pairwise distances
/mnt/ssd/usman_ws/OpenIBL/examples/ibl/evaluators.py:129: UserWarning: This overload of addmm_ is deprecated:
	addmm_(Number beta, Number alpha, Tensor mat1, Tensor mat2)
Consider using one of the following signatures instead:
	addmm_(Tensor mat1, Tensor mat2, *, Number beta, Number alpha) (Triggered internally at  /opt/conda/conda-bld/pytorch_1640811797118/work/torch/csrc/utils/python_arg_parser.cpp:1050.)
  dist_m.addmm_(1, -2, x, y.t())
===> Start calculating recalls
Recall Scores:
  top-1          80.0%
  top-5          93.3%
  top-10         96.0%
[W reducer.cpp:1303] Warning: find_unused_parameters=True was specified in DDP constructor, but did not find any unused parameters in the forward pass. This flag results in an extra traversal of the autograd graph every iteration,  which can adversely affect performance. If your model indeed never has any unused parameters in the forward pass, consider turning this flag off. Note that this warning may be a false positive if your model has flow control causing later iterations to have unused parameters. (function operator())
===> Start calculating pairwise distances
===> Start calculating recalls
Recall Scores:
  top-1          81.5%
  top-5          91.8%
  top-10         94.4%

 * Finished epoch   0 recall@1: 81.5%  recall@5: 91.8%  recall@10: 94.4%  best@5: 91.8% *

===> Start calculating pairwise distances
===> Start calculating recalls
Recall Scores:
  top-1          83.6%
  top-5          93.1%
  top-10         95.5%

 * Finished epoch   1 recall@1: 83.6%  recall@5: 93.1%  recall@10: 95.5%  best@5: 93.1% *

===> Start calculating pairwise distances
===> Start calculating recalls
Recall Scores:
  top-1          83.8%
  top-5          92.5%
  top-10         94.8%

 * Finished epoch   2 recall@1: 83.8%  recall@5: 92.5%  recall@10: 94.8%  best@5: 93.1%

===> Start calculating pairwise distances
===> Start calculating recalls
Recall Scores:
  top-1          85.7%
  top-5          94.4%
  top-10         96.5%

 * Finished epoch   3 recall@1: 85.7%  recall@5: 94.4%  recall@10: 96.5%  best@5: 94.4% *

===> Start calculating pairwise distances
===> Start calculating recalls
Recall Scores:
  top-1          85.7%
  top-5          94.0%
  top-10         96.0%

 * Finished epoch   4 recall@1: 85.7%  recall@5: 94.0%  recall@10: 96.0%  best@5: 94.4%

===> Start calculating pairwise distances
===> Start calculating recalls
Recall Scores:
  top-1          84.7%
  top-5          93.5%
  top-10         95.8%

 * Finished epoch   5 recall@1: 84.7%  recall@5: 93.5%  recall@10: 95.8%  best@5: 94.4%

===> Start calculating pairwise distances
===> Start calculating recalls
Recall Scores:
  top-1          85.2%
  top-5          93.7%
  top-10         95.9%

 * Finished epoch   6 recall@1: 85.2%  recall@5: 93.7%  recall@10: 95.9%  best@5: 94.4%

===> Start calculating pairwise distances
===> Start calculating recalls
Recall Scores:
  top-1          85.3%
  top-5          93.7%
  top-10         96.2%

 * Finished epoch   7 recall@1: 85.3%  recall@5: 93.7%  recall@10: 96.2%  best@5: 94.4%

===> Start calculating pairwise distances
===> Start calculating recalls
Recall Scores:
  top-1          85.7%
  top-5          93.9%
  top-10         95.9%

 * Finished epoch   8 recall@1: 85.7%  recall@5: 93.9%  recall@10: 95.9%  best@5: 94.4%

===> Start calculating pairwise distances
===> Start calculating recalls
Recall Scores:
  top-1          85.0%
  top-5          93.5%
  top-10         95.9%

 * Finished epoch   9 recall@1: 85.0%  recall@5: 93.5%  recall@10: 95.9%  best@5: 94.4%

Performing PCA reduction on the best model:
=> Loaded checkpoint '/media/leo/2C737A9872F69ECF/why-so-deepv2-data/pittsburgh/netvlad-run/pitts30k-vgg16/conv5-sare_ind-lr0.001-tuple1/model_best.pth.tar'
calculating PCA parameters...
/mnt/ssd/usman_ws/OpenIBL/examples/ibl/pca.py:53: UserWarning: torch.symeig is deprecated in favor of torch.linalg.eigh and will be removed in a future PyTorch release.
The default behavior has changed from using the upper triangular portion of the matrix by default to using the lower triangular portion.
L, _ = torch.symeig(A, upper=upper)
should be replaced with
L = torch.linalg.eigvalsh(A, UPLO='U' if upper else 'L')
and
L, V = torch.symeig(A, eigenvectors=True)
should be replaced with
L, V = torch.linalg.eigh(A, UPLO='U' if upper else 'L') (Triggered internally at  /opt/conda/conda-bld/pytorch_1640811797118/work/aten/src/ATen/native/BatchLinearAlgebra.cpp:2499.)
  L, U = torch.symeig(x2, eigenvectors=True)
================= PCA RESULT ==================
U: (32768, 4096)
lams: (4096,)
mu: (32768, 1)
Utmu: (4096, 1)
===============================================
Testing on Pitts30k-test:
load PCA parameters...
===> Start calculating pairwise distances
===> Start calculating recalls
Recall Scores:
  top-1          86.8%
  top-5          93.4%
  top-10         95.2%
expr: syntax error: missing argument after ‘-’                                             
zsh_weather:[:6: unknown condition: -gt
```