# Attributes-Guided and Pure-Visual Attention Alignment for Few-Shot Recognition

修改[train.py](https://github.com/e96031413/AGAM/blob/master/models/agam/train.py#L97)中，第97行的num-workers改成8，訓練速度會更快一點，請依照CPU性能調整。

## Requirements

The code runs correctly with

* Python 3.6
* PyTorch 1.2
* Torchvision 0.4

```shell
pip install ordered_set
```

## Custom CUB attribute script

在datasets的assets/cub/attributes資料夾底下各別執行以下兩個程式後，可取得自定義的attribute size(例如把CUB的312變成剩下156、50、6)

[create_class_attribute_labels_continuous_file.py](https://github.com/e96031413/AGAM/blob/master/datasets/assets/cub/attributes/create_class_attribute_labels_continuous_file.py)

[create_image_attribute_labels_file.py](https://github.com/e96031413/AGAM/blob/master/datasets/assets/cub/attributes/create_image_attribute_labels_file.py)

## Train CUB with the custom attribute script

* 每次用不同attribute size時都要記得修改以下內容：

1. 修改[AGAM/global_utils.py](https://github.com/e96031413/AGAM/blob/master/global_utils.py#L100)的第100行，把312依據自己的設定改成(156、50、6)

2. 根據[create_class_attribute_labels_continuous_file.py](https://github.com/e96031413/AGAM/blob/master/datasets/assets/cub/attributes/create_class_attribute_labels_continuous_file.py)
所產生的檔案名稱，修改[AGAM/torchmeta/datasets/semantic.py](https://github.com/e96031413/AGAM/blob/master/torchmeta/datasets/semantic.py#L127)第127行的class_attribute_filename_labels內容

3. 根據[create_image_attribute_labels_file.py](https://github.com/e96031413/AGAM/blob/master/datasets/assets/cub/attributes/create_image_attribute_labels_file.py)所產生的檔案名稱，修改[AGAM/torchmeta/datasets/semantic.py](https://github.com/e96031413/AGAM/blob/master/torchmeta/datasets/semantic.py#L129)第129行的image_attribute_filename_labels內容

4. 除此之外，也要修改[AGAM/torchmeta/datasets/semantic.py](https://github.com/e96031413/AGAM/blob/master/torchmeta/datasets/semantic.py#L131)第131行的attributes_dim內容

範例(156為例)：
```python
class CUBClassDataset(ClassDataset):
    folder = 'cub'

    # Google Drive ID from http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz
    gdrive_id = '1hbzc_P1FuxMkcabkgn9ZKinBwW683j45'
    tgz_filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'
    image_folder = 'CUB_200_2011/images'

    filename = '{0}_data.hdf5'
    filename_labels = '{0}_labels.json'

    assets_dir = 'assets'
    text_dir = 'text_c10'
    attribute_dir = 'attributes'
    class_attribute_filename_labels = 'class_attribute_labels_continuous_156.txt'  #改成156
    image_id_name_filename = 'images.txt'
    image_attribute_filename_labels = 'image_attribute_labels_156.txt'   #改成156
    classes_filename = 'classes.txt'
    attributes_dim = 156                     #312改成156
```

## How to run

```bash
# clone project
git clone https://github.com/bighuang624/AGAM.git
cd AGAM/models/agam_protonet

# download data and run on multiple GPUs with special settings
python train.py --train-data [train_data] --test-data [test_data] --backbone [backbone] --num-shots [num_shots] --train-tasks [train_tasks] --semantic-type [semantic_type] --multi-gpu --download


# Example: run on CUB dataset, Conv-4 backbone, 1 shot, single GPU (First time training with --download to get the dataset)
python train.py --train-data cub --test-data cub --backbone conv4 --num-shots 1 --train-tasks 50000 --semantic-type class_attributes --download
# Example: run on SUN dataset, ResNet-12 backbone, 5 shot, multiple GPUs (First time training with --download to get the dataset)
python train.py --train-data sun --test-data sun --backbone resnet12 --num-shots 5 --train-tasks 40000  --semantic-type image_attributes --multi-gpu --download

# If you have downloaded the dataset, use the command below:

# Example: run on CUB dataset, Conv-4 backbone, 1 shot, single GPU
python train.py --train-data cub --test-data cub --backbone conv4 --num-shots 1 --train-tasks 50000 --semantic-type class_attributes
# Example: run on SUN dataset, ResNet-12 backbone, 5 shot, multiple GPUs
python train.py --train-data sun --test-data sun --backbone resnet12 --num-shots 5 --train-tasks 40000  --semantic-type image_attributes --multi-gpu
```

### Data Preparation

You can download datasets automatically by adding `--download` when running the program. However, here we give steps to manually download datasets to prevent problems such as poor network connection:

**CUB**:

1. Create the dir `AGAM/datasets/cub`;
2. Download `CUB_200_2011.tgz` from [here](https://drive.google.com/file/d/1hbzc_P1FuxMkcabkgn9ZKinBwW683j45/view), and put the archive into `AGAM/datasets/cub`;
3. Running the program with `--download`.

**SUN**:

1. Create the dir `AGAM/datasets/sun`;
2. Download the archive of images from [here](http://cs.brown.edu/~gmpatter/Attributes/SUNAttributeDB_Images.tar.gz), and put the archive into `AGAM/datasets/sun`;
3. Download the archive of attributes from [here](http://cs.brown.edu/~gmpatter/Attributes/SUNAttributeDB.tar.gz), and put the archive into `AGAM/datasets/sun`;
4. Running the program with `--download`.

## Citation

If our code is helpful for your research, please cite our paper:

```
@inproceedings{Huang2021AGAM,
  author = {Siteng Huang and Min Zhang and Yachen Kang and Donglin Wang},
  title = {Attributes-Guided and Pure-Visual Attention Alignment for Few-Shot Recognition},
  booktitle = {Proceedings of the 35th AAAI Conference on Artificial Intelligence (AAAI 2021)},
  month = {February},
  year = {2021}
}
```

## Acknowledgement

Our code references the following projects:

* [Torchmeta](https://github.com/tristandeleu/pytorch-meta)
* [FEAT](https://github.com/Sha-Lab/FEAT)
