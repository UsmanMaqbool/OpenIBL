## GraphVLAD

![GitHub Icon](https://img.icons8.com/fluent/24/000000/github.png): GitHub [Official source](https://github.com/yxgeee/OpenIBL) | [My forked](https://github.com/UsmanMaqbool/OpenIBL)
![YouTube Icon](https://img.icons8.com/fluent/24/000000/pdf.png): Paper
![YouTube Icon](https://img.icons8.com/fluent/24/000000/youtube-play.png): YouTube

## Todo 📝

- [ ] Bugs
    - [ ] Todo
- [ ] Install
    - [ ] Todo
- [ ] Training
    - [ ] Todo
- [ ] Testing
    - [ ] Todo


## Clustering
init directory should have `vd16_offtheshelf_conv5_3_max.pth` file.
```shell
./scripts/cluster.sh vgg16 /home/leo/usman_ws/datasets/openibl-init

## Transfer to Hipergator
### 29 July
rsync -ah --progress -e 'ssh -p 2222' ~/usman_ws/datasets/openibl-init/vgg16_pitts_64_desc_cen.hdf5 m.maqboolbhutta@hpg.rc.ufl.edu:/home/m.maqboolbhutta/usman_ws/datasets/openibl-init/
```

## Training and Testing 🚀
### GraphVLAD
#### PC
```sh 
# Single
./scripts/train_baseline_dist.sh graphvlad triplet vgg16 pitts 30k
# Training all
./scripts/all/train_baseline_dist_all.sh graphvlad vgg16 pitts 30k
```
#### Slurm

```sh
# Single
sbatch --j graphvlad-triplet-29jul ./scripts/leo/train_baseline_slurm.sh graphvlad vgg16 pitts 30k triplet
# 0620-v9-graphvlad-sort-gal 
## Training
sbatch --j netvlad-sare-ind scripts/train_baseline_slurm_all.sh graphvlad sare_ind vgg16 pitts 30k

## Testing
### 29July,2024
sbatch --j graphvlad-test scripts/leo/test_slurm_all.sh graphvlad vgg16 pitts 30k /home/m.maqboolbhutta/usman_ws/models/openibl/fastscnn/vgg16-graphvlad-triplet-pitts30k-lr0.01-tuple4-29-Jul
```



#### Vanilla
#### SFRS




##### PC

```sh
### Data: 
## Training
./scripts/train_sfrs_dist.sh graphvlad sare_ind vgg16 pitts 30k

## Testing

### Download to PC
#### /home/m.maqboolbhutta/usman_ws/models/openibl/vgg16-graphvlad-sare_ind-pitts30k-lr0.001-tuple4-06-Jul/checkpoint3_4.pth.tar
./scripts/test_dist.sh graphvlad /home/leo/usman_ws/models/openibl/graphvlad/checkpoint3_4.pth.tar

```
##### Slurm

```sh
# fastscnn-v2 - 1Aug
## Training
sbatch --j fastscnn-v2 scripts/leo/train_sfrs_slurm_all.sh graphvlad vgg16 pitts 30k

### Testing
sbatch --j graphvlad-sare-ind scripts/test_slurm_all.sh graphvlad vgg16 pitts 30k /home/m.maqboolbhutta/usman_ws/models/openibl/graphvlad/

```

#### Results

##### Copy
```sh
rsync -ah --progress -e 'ssh -p 2222' m.maqboolbhutta@hpg.rc.ufl.edu:/home/m.maqboolbhutta/usman_ws/models/openibl/vgg16-graphvlad-sare_ind-pitts30k-lr0.001-tuple4-17-Jul ~/usman_ws/models/openibl/graphvlad/hipergator/
```

##### July 15: SareIND : v1.1graphvlad-sfrs | Hipergator
Location: /home/m.maqboolbhutta/usman_ws/models/openibl/vgg16-graphvlad-sare_ind-pitts30k-lr0.001-tuple4-15-Jul
```sh
# tested on PC
## location: /home/leo/usman_ws/models/openibl/graphvlad/hipergator/vgg16-graphvlad-sare_ind-pitts30k-lr0.001-tuple4-15-Jul 
./scripts/test_dist_all.sh graphvlad /home/leo/usman_ws/models/openibl/graphvlad/hipergator/vgg16-graphvlad-sare_ind-pitts30k-lr0.001-tuple4-17-Jul 

```

## Debug

### PC

- GraphVLAD Train
- Recent
    ```json
    {
        // Use IntelliSense to learn about possible attributes.
        // Hover to view descriptions of existing attributes.
        // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
        "version": "0.2.0",
        "configurations": [
            {
                "name": "OpenIBL Train Debug",
                "type": "python",
                "request": "launch",
                "program": "/home/leo/anaconda3/envs/openibl/lib/python3.8/site-packages/torch/distributed/launch.py",
                "console": "integratedTerminal",
                "args": [
                "--nproc_per_node=1 ",
                "--master_port=6050",
                "--use_env",
                "examples/netvlad_img.py",
                "--launcher=pytorch",
                "-d", "pitts",
                "--scale", "30k",
                "-a", "vgg16",
                "--layers", "conv5",
                "--vlad",
                "--syncbn",
                "--sync-gather",
                "--width", "640",
                "--height", "480",
                "--tuple-size", "1",
                "-j", "1",
                "--neg-num", "10",
                "--test-batch-size", "32",
                "--margin", "0.1",
                "--lr", "0.001",
                "--weight-decay", "0.001",
                "--loss-type", "triplet",
                "--eval-step", "1",
                "--epochs", "5",
                "--step-size", "5",
                "--cache-size", "1000",
                "--logs-dir", "/home/leo/usman_ws/models/openibl/debug",
                "--data-dir", "/home/leo/usman_ws/codes/OpenIBL/examples/data/",
                "--init-dir", "/home/leo/usman_ws/datasets/openibl-init",
                "--fast-scnn", "/home/leo/usman_ws/datasets/fast_scnn/fast_scnn_citys.pth",
                "--method", "graphvlad",
                ],
            }
        ]
    }
    ```
- Previous
    ```json
    {
        // Use IntelliSense to learn about possible attributes.
        // Hover to view descriptions of existing attributes.
        // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
        "version": "0.2.0",
        "configurations": [
            {
                "name": "OpenIBL Train Debug",
                "type": "python",
                "request": "launch",
                "program": "/home/leo/anaconda3/envs/openibl/lib/python3.8/site-packages/torch/distributed/launch.py",
                "console": "integratedTerminal",
                "args": [
                    "--nproc_per_node=1 ",
                    "--master_port=6010",
                    "--use_env",
                    "examples/netvlad_img.py",
                    "--launcher=pytorch",
                    "--logs-dir=/home/leo/usman_ws/models/openibl/debug", 
                    "--data-dir=/home/leo/usman_ws/codes/OpenIBL/examples/data/",
                    "--init-dir=/home/leo/usman_ws/datasets/openibl-init",
                    "--esp-encoder=/home/leo/usman_ws/datasets/espnet-encoder/espnet_p_2_q_8.pth",
                    "--method=graphvlad",
                ],
            }
        ]
    }
    ```
- GraphVLAD Test
    ```json
    {
        // Use IntelliSense to learn about possible attributes.
        // Hover to view descriptions of existing attributes.
        // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
        "version": "0.2.0",
        "configurations": [
            {
                "name": "Python: Module",
                "type": "python",
                "request": "launch",
                "program": "/home/leo/anaconda3/lib/python3.1/site-packages/torch/distributed/launch.py",
                "console": "integratedTerminal",
                "args": [
                    "--nproc_per_node=1 ",
                    "--master_port=6010",
                    "--use_env",
                    "examples/test.py",
                    "--vlad",
                    "--reduction",
                    "--resume=/home/leo/usman_ws/models/openibl/official/pitts30k-vgg16/conv5-triplet-lr0.0001-tuple1-07-Nov/checkpoint0.pth.tar",
                ],
            }
        ]
    }
    ```

- Extra

    ```json
    {
        "version": "0.2.0",
        "configurations": [
        {
            "name": "Python: Train Baseline Dist",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/examples/netvlad_img.py",
            "console": "integratedTerminal",
            "env": {
            "PYTHON": "python",
            "GPUS": "1",
            "NUMCLUSTER": "16",
            "LAYERS": "conv5",
            "LR": "0.001",
            "INIT_DIR": "/home/leo/usman_ws/datasets/openibl-init",
            "ESP_ENCODER": "/home/leo/usman_ws/datasets/espnet-encoder/espnet_p_2_q_8.pth",
            "DATASET_DIR": "/home/leo/usman_ws/codes/OpenIBL/examples/data/"
            },
            "args": [
            "--nproc_per_node=1 ",
            "--master_port=6010",
            "--use_env",
            "-d", "${input:dataset}",
            "--scale", "${input:scale}",
            "-a", "${input:arch}",
            "--layers", "conv5",
            "--vlad",
            "--syncbn",
            "--sync-gather",
            "--width", "640",
            "--height", "480",
            "--tuple-size", "1",
            "-j", "1",
            "--neg-num", "10",
            "--test-batch-size", "32",
            "--margin", "0.1",
            "--lr", "0.001",
            "--weight-decay", "0.001",
            "--loss-type", "${input:loss}",
            "--eval-step", "1",
            "--epochs", "5",
            "--step-size", "5",
            "--cache-size", "1000",
            "--logs-dir", "/home/leo/usman_ws/models/openibl/vgg16-graphvlad-triplet-pitts30k-lr0.001-tuple1-11-Jun/",
            "--data-dir", "/home/leo/usman_ws/codes/OpenIBL/examples/data/",
            "--init-dir", "/home/leo/usman_ws/datasets/openibl-init",
            "--esp-encoder", "/home/leo/usman_ws/datasets/espnet-encoder/espnet_p_2_q_8.pth",
            "--method", "${input:method}"
            ],
            "python": "${env:PYTHON}"
        }
        ],
        "inputs": [
        {
            "id": "method",
            "type": "promptString",
            "description": "Enter the method",
            "default": "graphvlad"
        },
        {
            "id": "loss",
            "type": "promptString",
            "description": "Enter the loss type",
            "default": "triplet"
        },
        {
            "id": "arch",
            "type": "promptString",
            "description": "Enter the architecture",
            "default": "vgg16"
        },
        {
            "id": "dataset",
            "type": "promptString",
            "description": "Enter the dataset",
            "default": "pitts"
        },
        {
            "id": "scale",
            "type": "promptString",
            "description": "Enter the scale",
            "default": "30k"
        }
        ]
    }
    

    // {
    //     // Use IntelliSense to learn about possible attributes.
    //     // Hover to view descriptions of existing attributes.
    //     // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    //     "version": "0.2.0",
    //     "configurations": [

    //         {
    //             "name": "Python: Module",
    //             "type": "python",
    //             "request": "launch",
    //             "program": "/home/leo/anaconda3/lib/python3.1/site-packages/torch/distributed/launch.py",
    //             "console": "integratedTerminal",
    //             "args": [
    //                 "--nproc_per_node=1 ",
    //                 "--master_port=6010",
    //                 "--use_env",
    //                 "examples/test.py",
    //                 "--vlad",
    //                 "--reduction",
    //                 "--resume=/home/leo/usman_ws/models/openibl/official/pitts30k-vgg16/conv5-triplet-lr0.0001-tuple1-07-Nov/checkpoint0.pth.tar",
    //             ],
    //         }
    //     ]
    // }
    //Train
    // {
    //     // Use IntelliSense to learn about possible attributes.
    //     // Hover to view descriptions of existing attributes.
    //     // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    //     "version": "0.2.0",
    //     "configurations": [
    //         {
    //             "name": "Debugging Train Baseline Dist",
    //             "type": "python",
    //             "request": "launch",
    //             "program": "/home/leo/anaconda3/envs/openibl/lib/python3.8/site-packages/torch/distributed/launch.py",
    //             "console": "integratedTerminal",
    //             "args": [
    //                 "-m=torch.distributed.launch",
    //                 "--nproc_per_node=1 ",
    //                 "--master_port=6010",
    //                 "--use_env",
    //                 "examples/netvlad_img.py",
    //                 "-d=pitts",
    //                 "--scale=30k",
    //                 "-a ${ARCH}",
    //                 "--layers ${LAYERS}",
    //                 "--vlad",
    //                 "--syncbn",
    //                 "--sync-gather",
    //                 "--width 640",
    //                 "--height 480",
    //                 "--tuple-size=1",
    //                 "-j=1",
    //                 "--neg-num=10",
    //                 "--test-batch-size=32",
    //                 "--margin=0.1",
    //                 "--lr=${LR}",
    //                 "--weight-decay=0.001",
    //                 "--loss-type=${LOSS}",          
    //                 "--eval-step=1",
    //                 "--epochs=5",
    //                 "--step-size=5",
    //                 "--cache-size=1000",
    //                 "--logs-dir=/home/leo/usman_ws/models/openibl/debug/",
    //                 "--data-dir=/home/leo/usman_ws/codes/OpenIBL/examples/data/",
    //                 "--init-dir=/home/leo/usman_ws/datasets/openibl-init",
    //                 "--esp-encoder=/home/leo/usman_ws/datasets/espnet-encoder/espnet_p_2_q_8.pth",
    //                 "--method=graphvlad",
    //             ],
    //             "stopOnEntry": true,
    //             "debugOptions": [
    //                 "WaitOnAbnormalExit",
    //                 "WaitOnNormalExit",
    //                 "RedirectOutput"
    //             ]
    //         }
    //     ]
    // }


    "-d=pitts",
    "--launcher=pytorch",
    "--scale=30k",
    "-a=vgg16",
    "--vlad",
    "--syncbn",
    "--sync-gather",
    "--width=640",
    "--height=480",
    "--tuple-size=1",
    "-j=1",
    "--neg-num=10",
    "--test-batch-size=32",
    "--margin=0.1",
    "--lr=0.001",
    "--weight-decay=0.001",
    "--loss-type=triplet",          
    "--eval-step=1",
    "--epochs=5",
    "--step-size=5",
    "--cache-size=1000",
    "--logs-dir=/home/leo/usman_ws/models/openibl/debug/",
    "--data-dir=/home/leo/usman_ws/codes/OpenIBL/examples/data/",
    "--init-dir=/home/leo/usman_ws/datasets/openibl-init",
    "--esp-encoder=/home/leo/usman_ws/datasets/espnet-encoder/espnet_p_2_q_8.pth",
    "--method=graphvlad",
    ```


- SFRS
    ```json
    {
        // Use IntelliSense to learn about possible attributes.
        // Hover to view descriptions of existing attributes.
        // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
        "version": "0.2.0",
        "configurations": [
            {
                "name": "OpenIBL Train Debug",
                "type": "python",
                "request": "launch",
                "program": "/home/leo/anaconda3/envs/openibl/lib/python3.8/site-packages/torch/distributed/launch.py",
                "console": "integratedTerminal",
                "args": [
                    "--nproc_per_node=1 ",
                    "--master_port=6010",
                    "--use_env",
                    "examples/netvlad_img_sfrs.py",
                    "--launcher=pytorch",
                    "--init-dir=/home/leo/usman_ws/datasets/openibl-init",
                    "--esp-encoder=/home/leo/usman_ws/datasets/espnet-encoder/espnet_p_2_q_8.pth"
                ],
            }
        ]
    }


    ```

### Hipergator
**Interactive:**
```sh
srun --partition=gpu --gpus-per-node=a100:4 --cpus-per-task=24 --pty bash
module load conda/24.1.2 intel/2019.1.144 openmpi/4.0.0
conda activate openibl
```
Train file: `./scripts/debug_train_sfrs_dist_slurm.sh`
```sh
./scripts/debug_train_sfrs_dist_slurm.sh graphvlad sare_ind vgg16 pitts 30k
```
## Install ⚙️

### Configure 🪛

### Dataset 🖼️

Follow Install instruction [link](https://github.com/yxgeee/OpenIBL/blob/master/docs/INSTALL.md)

```shell
cd examples && mkdir data
```
Download the raw datasets and then unzip them under the directory like
```shell
examples/data
├── pitts
│   ├── raw
│   │   ├── pitts250k_test.mat
│   │   ├── pitts250k_train.mat
│   │   ├── pitts250k_val.mat
│   │   ├── pitts30k_test.mat
│   │   ├── pitts30k_train.mat
│   │   ├── pitts30k_val.mat
│   └── └── Pittsburgh/images/
│   └── └── Pittsburgh/queries/
└── tokyo
    ├── raw
    │   ├── tokyo247/images
    │   ├── tokyo247/query    
    │   ├── tokyo247.mat
    │   ├── tokyoTM/
    │   ├── tokyoTM_train.mat
    └── └── tokyoTM_val.mat
```
### Copy weights from PC to Hipergator
rsync -ah --progress -e 'ssh -p 2222' ~/usman_ws/datasets/fast_scnn m.maqboolbhutta@hpg.rc.ufl.edu:/home/m.maqboolbhutta/usman_ws/datasets/

## Copy to cluster
### PC 2 Cluster
rsync -ah --progress -e 'ssh -p 2222' /home/leo/usman_ws/datasets/2015netVLAD/Pittsburgh250k/compressed/ m.maqboolbhutta@hpg.rc.ufl.edu:/home/m.maqboolbhutta/usman_ws/datasets/Pittsburgh250k/images/
##### Extract
find . -name '*.tar' -execdir tar -xvf '{}' \;

#### Copy queries and extract
rsync -ah --progress -e 'ssh -p 2222' /home/leo/usman_ws/datasets/2015netVLAD/Pittsburgh250k/queries/queries_real.zip m.maqboolbhutta@hpg.rc.ufl.edu:/home/m.maqboolbhutta/usman_ws/datasets/Pittsburgh250k/queries/
unzip queries_real.zip

## Configure like
cd /blue/hmedeiros/m.maqboolbhutta/codes/OpenIBL/examples/data/pitts/OpenIBL/examples/data/pitts/raw/

### Copy datasets-specs
rsync -ah --progress -e 'ssh -p 2222' /home/leo/usman_ws/datasets/2015netVLAD/datasets-specs/pitts*.mat m.maqboolbhutta@hpg.rc.ufl.edu:/blue/hmedeiros/m.maqboolbhutta/codes/OpenIBL/examples/data/pitts/OpenIBL/examples/data/pitts/raw/

## Now create symbolic links for the images and queries folders
### Being in tha raw folder
cd /home/leo/usman_ws/codes/OpenIBL/examples/data/pitts/raw
ln -s /blue/hmedeiros/m.maqboolbhutta/datasets/Pittsburgh250k/ Pittsburgh
ln -s /home/leo/usman_ws/datasets/2015netVLAD/Pittsburgh250k/ Pittsburgh 


### Pittsburg now should have two folders
images
queries



#### Tokyo Dataset
cd /blue/hmedeiros/m.maqboolbhutta/datasets/tokyo247
##### Download all the images
wget -r -A.tar https://data.ciirc.cvut.cz/public/projects/2015netVLAD/Tokyo247/database_gsv_vga/
## move the folders to tokyo247 and untar all the files
find . -name '*.tar' -execdir tar -xvf '{}' \;


##### Download all the queries
rsync -ah --progress -e 'ssh -p 2222' /home/leo/usman_ws/datasets/2015netVLAD/247_Tokyo_GSV_Perspective/247query_v3.zip m.maqboolbhutta@hpg.rc.ufl.edu:/blue/hmedeiros/m.maqboolbhutta/datasets/tokyo247/

### Copy datasets-specs
rsync -ah --progress -e 'ssh -p 2222' /home/leo/usman_ws/datasets/2015netVLAD/datasets-specs/tokyo*.mat m.maqboolbhutta@hpg.rc.ufl.edu:/blue/hmedeiros/m.maqboolbhutta/codes/OpenIBL/examples/data/tokyo/raw/

#### Create symbolic links
cd /blue/hmedeiros/m.maqboolbhutta/codes/OpenIBL/examples/data/tokyo/raw
ln -s /blue/hmedeiros/m.maqboolbhutta/datasets/tokyo247/ tokyo247
cd /home/leo/usman_ws/codes/OpenIBL/examples/data/tokyo/raw
ln -s /home/leo/usman_ws/datasets/2015netVLAD/tokyo247/ tokyo247
```