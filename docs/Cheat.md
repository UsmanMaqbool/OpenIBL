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



## Training and Testing 🚀
### GraphVLAD
#### Vanilla
#### SFRS




##### PC

```sh
### Data: 
## Training
./scripts/train_sfrs_dist.sh graphvlad sare_ind vgg16 pitts 30k
## Testing

```
##### Slurm

```sh
# 0620-v9-graphvlad-sort-gal 
## Training
sbatch --j netvlad-triplet scripts/train_sfrs_slurm_all.sh graphvlad sare_ind vgg16 pitts 30k
sbatch --j sfrs-graphvlad-dev3-1355-2jul scripts/train_sfrs_slurm_all.sh graphvlad sare_joint vgg16 pitts 30k

### Testin
sbatch --j netvlad-triplet scripts/test_slurm_all.sh graphvlad vgg16 pitts 30k /home/m.maqboolbhutta/usman_ws/models/openibl/vgg16-graphvlad-sare_ind-pitts30k-lr0.01-tuple4-19-Jun/

```
**Interactive:**
```sh
srun --partition=gpu --gres=a100:4 --time=2:00:00 --nodes=1 --ntasks=1 --cpus-per-task=24 --constraint=a100 --mem-per-cpu=8GB --distribution=cyclic:cyclic --pty bash -i
module load conda/24.1.2 intel/2019.1.144 openmpi/4.0.0
conda activate openibl

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