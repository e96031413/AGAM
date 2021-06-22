# Attributes-Guided and Pure-Visual Attention Alignment for Few-Shot Recognition

程式碼本身包含training和testing

修改[train.py](https://github.com/e96031413/AGAM/blob/master/models/agam/train.py#L97)中，第97行的num-workers改成8，訓練速度會更快一點，請依照CPU性能調整。

不使用attributes-guided的程式碼[agam/model.py](https://github.com/e96031413/AGAM/blob/master/models/agam/model.py#L143)、[train_wo_attribute.py](https://github.com/e96031413/AGAM/blob/master/models/agam/train_wo_attribute.py)、[使用attribute與不使用attribute的train.py差異](https://www.diffchecker.com/sv6TWcM6)

在train_wo_attribute.py當中:
- Line 17 新增```from model import ProtoNetAGAM, ProtoNetAGAMwoAttr```
- Line 47 新增```torch.backends.cudnn.enabled = False```，避免```cuDNN error: CUDNN_STATUS_NOT_SUPPORTED. This error may appear if you passed```
- Line 150 新增```model = ProtoNetAGAMwoAttr(args.backbone, args.semantic_size, args.out_channels)```

**training的部份：**

- Line 239 改成```support_embeddings, ca_weights, sa_weights = model(support_inputs, semantics=support_semantics, output_weights=True)```，原本的code會有semantic的feature，這邊只剩下image的feature
- Line 240 原本還有```addition_loss = get_addition_loss(ca_weights, sca_weights, sa_weights, ssa_weights, args)```，但是因為不使用attribute後，addition loss中的semantic weight也沒有幫助，所以就移除了
- Line 248 變成```del ca_weights, sa_weights```，原本有semantic的weight，這邊也移除

**val的部份：**

- Line 271 改成```support_embeddings, ca_weights, sa_weights = model(support_inputs, semantics=support_semantics, output_weights=True)```，原本的code會有semantic的feature，這邊只剩下image的feature
- Line 272 原本還有```addition_loss = get_addition_loss(ca_weights, sca_weights, sa_weights, ssa_weights, args)```，但是因為不使用attribute後，addition loss中的semantic weight也沒有幫助，所以就移除了
- Line 280 變成```del ca_weights, sa_weights```，原本有semantic的weight，這邊也移除

**test的部份:**

- Line 317 改成```support_embeddings, ca_weights, sa_weights = model(support_inputs, semantics=support_semantics, output_weights=True)```，原本的code會有semantic的feature，這邊只剩下image的feature
- Line 318 原本還有```addition_loss = get_addition_loss(ca_weights, sca_weights, sa_weights, ssa_weights, args)```，但是因為不使用attribute後，addition loss中的semantic weight也沒有幫助，所以就移除了
- Line 326 變成```del ca_weights, sa_weights```，原本有semantic的weight，這邊也移除

參考[Question on training without attribute in 4.4. Ablation Study on Using Attributes #3](https://github.com/bighuang624/AGAM/issues/3)

參考[[訓練報錯]cuDNN error: CUDNN_STATUS_NOT_SUPPORTED.](https://blog.csdn.net/qq_22764813/article/details/108419648)


## Requirements

The code runs correctly with

* Python 3.6
* PyTorch 1.2
* Torchvision 0.4

```shell
pip install torch==1.2.0 torchvision==0.4.0

pip install ordered_set
pip install tqdm
pip install pandas
pip install h5py
pip install scipy
```

## Custom CUB attribute script

在datasets的assets/cub/attributes資料夾底下的ipynb程式後，可取得自定義的attribute size(例如把CUB的312變成剩下156、50、6)

class_attribute_labels_continuous.txt和image_attribute_labels.txt兩種

由於有3種不同的attribute size，因此會有(2x3=6個檔案)、預先建立好的檔案在[semantic_attributes_with_fewer_att_size.zip](https://github.com/e96031413/AGAM/blob/master/datasets/assets/cub/attributes/semantic_attributes_with_fewer_att_size.zip)

[create_dataset_with_fewer_semantic_attribute.ipynb](https://github.com/e96031413/AGAM/blob/master/datasets/assets/cub/attributes/create_dataset_with_fewer_semantic_attribute.ipynb)

這個notebook提供了兩種模式：(1)指定前N個attribute(例如前6個attribute)。(2)指定隨機6個attribute(隨機刪除attribute只保留6個attribute)

## Train CUB with the custom attribute script

* 每次用不同attribute size時都要記得修改以下內容：

1. 修改[AGAM/global_utils.py](https://github.com/e96031413/AGAM/blob/master/global_utils.py#L100)的第100行，把312依據自己的設定改成(156、50、6)

* 範例(156為例)：

```python
def get_semantic_size(args):

    semantic_size_list = []

    for semantic_type in args.semantic_type:

        if semantic_type == 'class_attributes':
            if args.train_data == 'cub':
                semantic_size_list.append(156)    #改成156
```

2. 根據[create_class_attribute_labels_continuous_file.py](https://github.com/e96031413/AGAM/blob/master/datasets/assets/cub/attributes/create_class_attribute_labels_continuous_file.py)
所產生的檔案名稱，修改[AGAM/torchmeta/datasets/semantic.py](https://github.com/e96031413/AGAM/blob/master/torchmeta/datasets/semantic.py#L127)第127行的class_attribute_filename_labels內容

3. 根據[create_image_attribute_labels_file.py](https://github.com/e96031413/AGAM/blob/master/datasets/assets/cub/attributes/create_image_attribute_labels_file.py)所產生的檔案名稱，修改[AGAM/torchmeta/datasets/semantic.py](https://github.com/e96031413/AGAM/blob/master/torchmeta/datasets/semantic.py#L129)第129行的image_attribute_filename_labels內容

4. 除此之外，也要修改[AGAM/torchmeta/datasets/semantic.py](https://github.com/e96031413/AGAM/blob/master/torchmeta/datasets/semantic.py#L131)第131行的attributes_dim內容

* 範例(156為例)：
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
cd AGAM/models/agam

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

# 進行無attribute的訓練
python train_wo_attribute.py --train-data cub --test-data cub --backbone conv4 --num-shots 1 --train-tasks 50000 --semantic-type class_attributes
```

### 從中斷的model繼續進行訓練
```
python train.py --train-data cub --test-data cub --backbone conv4 --num-shots 1 --train-tasks 50000 --semantic-type class_attributes --resume --resume-folder checkpoint資料夾路徑

# 例如：
python train.py --train-data cub --test-data cub --backbone conv4 --num-shots 1 --train-tasks 50000 --semantic-type class_attributes --resume --resume-folder no_attribute_cub_cub_protonet_agam_conv4_2021-05-27-17-59-38
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
