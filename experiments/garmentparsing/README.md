# Introduce

This project is the semantic segmentation examination of garment pointcloud generated form StyleXD dataset. 

The examination include 3 stage:

1. **Data Preparation**：We preprocess the pointcloud of StyleXD dataset to the s3dis format , and then use the same way to process preprocessed data same as the pointcept process the s3dis dataset.
2. **Train**：We train PointTransformer V3 with those data.
3. **Test**：After train finished, we run test code to get the inference result, and you can visualize it then.

------



# Instillation

To set up the required environment, follow these steps:

Clone this repository, and open a terminal at the root dir of this projrct.

Create a conda environment:

```bash
conda create -n pointcept python=3.8 -y
conda activate pointcept

# We use CUDA 11.8 and PyTorch 2.1.0 for our development of PTv3
conda install cudatoolkit==11.8.0
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 \
              --index-url https://download.pytorch.org/whl/cu118

conda install ninja -y
conda install h5py pyyaml -c anaconda -y
conda install sharedarray tensorboard tensorboardx yapf addict einops scipy plyfile termcolor timm -c conda-forge -y
conda install pytorch-cluster pytorch-scatter pytorch-sparse -c pyg -y
pip install torch-geometric

cd libs/pointops
python setup.py install
cd ../..

# spconv (SparseUNet)
# refer https://github.com/traveller59/spconv
pip install spconv-cu118  # choose version match your local cuda version

# Open3D (visualization, optional)
pip install open3d

pip install flash-attn --no-build-isolation
```



---



# Getting Started


## Data Preparation
Download StyleXD dataset from kaggle, and unzip the archive to `PATH_TO_STYLEXD_DATASET`.

Then, run `preporcess_stylexd.py` to turn our dataset into pointcept format:

```bash
python exp/StyleXD/dataset_process/preporcess_stylexd.py \
    --dataset_root=PATH_TO_STYLEXD_DATASET \
    --output_root=PATH_TO_PREPROCESSED_DATASET
```

For convenience, you can soft link the dataset to the project root path:

```bash
ln -s PATH_TO_PREPROCESSED_DATASET /path/to/the/project/pointcept/data/StyleXD
```

Finally, the project structure is:

```bash
pointcept
├── data
│   ├── StyleXD
│   │   ├── Area_1								# Area used to split dataset
│   │   │   ├── Garment00000
│   │   │   │   ├── coord.npy					# Points Coordination
│   │   │   │   ├── color.npy					# Color (not important)
│   │   │   │   ├── segment.npy					# Semantic Segmentation
│   │   │   │   ├── instance.npy				# Instance Segmentation
│   │   │   │   ├── normal.npy					# Points Normal
│   │   │   ├── ...
│   │   ├── ...
│   │   ├── Area_6
├── ...
```

## Other
Register [WandB](https://wandb.ai/site), and get your API key.

---

## Training
```bash
export PYTHONPATH=/path/to/the/project
python train.py \
--config-file=onfig.py \
--num-gpus=1 \
--options \
	save_path=exp/stylexd_semseg  \
	test_only=false
```


## Testing
```bash
export PYTHONPATH=/path/to/the/project
python test.py \
--config-file=config.py \
--num-gpus=1 \
--options \
	save_path=exp/stylexd_semseg \
	weight=exp/stylexd_semseg/model/model_best.pth \
	batch_size=10 \
	test_only=true
```

Default save path of test results is `exp/stylexd_semseg/results` (you may change `test_save_path` in `config.py`).

Visualizing the test result:

```bash
export PYTHONPATH=<PATH_TO_THIS_PROJECT_ROOT>
python vis.py \
--pred_data_dir=exp/stylexd_semseg \
--original_data_dir=PATH_TO_THE_STYLEXD_DATASET
```


# Experimental Results

| Predict                                    | Ground True                                | Predict                                    | Ground True                                |
| ------------------------------------------ | ------------------------------------------ | ------------------------------------------ | ------------------------------------------ |
| ![garment1_1](docs/garment/garment1_1.png) | ![garment1_2](docs/garment/garment1_2.png) | ![garment2_1](docs/garment/garment2_1.png) | ![garment2_2](docs/garment/garment2_2.png) |
| ![garment6_1](docs/garment/garment6_1.png) | ![garment6_2](docs/garment/garment6_2.png) | ![garment4_1](docs/garment/garment4_1.png) | ![garment4_2](docs/garment/garment4_2.png) |
| ![garment7_1](docs/garment/garment7_1.png) | ![garment7_2](docs/garment/garment7_2.png) | ![garment8_1](docs/garment/garment8_1.png) | ![garment8_2](docs/garment/garment8_2.png) |

Num of each class panel:

![panel_num](docs/statistics/panel_num.png)

Prediction results of each type of panel point:

![panel_num](docs/statistics/pred_results.png)

# Acknowledgement
This project is build upon [Pointcept](https://github.com/Pointcept/Pointcept).