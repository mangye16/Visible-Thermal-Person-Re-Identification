## Visible Thermal Person Re-Identification (Cross-modlaity Person Re-ID)

Demo code for [Hierarchical Discriminative Learning for Visible Thermal Person Re-Identification](http://www.comp.hkbu.edu.hk/~mangye/files/aaai18_vtreid.pdf) in AAAI 2018 and [Visible Thermal Person Re-Identification via Dual-Constrained Top-Ranking](http://www.comp.hkbu.edu.hk/~mangye/files/ijcai18_vtreid.pdf) in IJCAI 2018 .

The framework of our AAAI 18 paper: Two-Stage Framework (Feature Learning + Metric Learning)
![](http://www.comp.hkbu.edu.hk/~mangye/files/aaai18_framework.jpg)


The framework of our IJCAI 18 paper: End-to-End Learning
![](http://www.comp.hkbu.edu.hk/~mangye/files/ijcai18_framework.jpg)

### 1. Prepare the dataset.

- The RegDB dataset can be downloaded from this [website](http://dm.dongguk.edu/link.html) by submitting a copyright form.

(Named: "Dongguk Body-based Person Recognition Database (DBPerson-Recog-DB1) on their website.")

### 2. Two-stream CNN network feature learning (TONE) in AAAI 2018

All the code is in the folder 'TONE/' written in Python. This demo code has been tested on Python 2.7 and Tensorflow v0.11. 

- a. Prepare the dataset and the train/test lists as shown in `TONE/dataset.py`. The list format is `image_path label`.

- b. Download the pre-trained alexnet model and modify the `TONE/model.py`. [Alexnet](http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/)

- c. run `python TONE/tone_train.py` to train the network. 

- d. run `python TONE/tone_eval.py` to evaluate the learned features and extract features for later metric learning. 

(You can also modify the scripts to get different layers of features or evaluate different parameters. A trained model of one split can be downloaded on [BaiduYun](https://pan.baidu.com/s/1kVaMkPx) and [GoogleDrive](https://drive.google.com/open?id=1v2-Cry-9O5ZhHySLpMbsr-BJfe6Zxhe5))


### 3. Hierarchical Cross-modality Metric Learning (HCML) in AAAI 18
All the code is in the folder 'HCML/' written in Matlab. This demo code has been tested on Matlab 2017a.

- a. Make sure the format of the extracted features is correct. Demo Features on [BaiduYun](https://pan.baidu.com/s/1kVaMkPx) and [GoogleDrive](https://drive.google.com/open?id=1v2-Cry-9O5ZhHySLpMbsr-BJfe6Zxhe5)

- b. run `HCML/demo_hcml.m` to evaluate the cross-modal metric learning results.


### 4. Bi-directional Dual-Constrained Top-Ranking (BDTR) in IJCAI 2018 

All the code is in the folder 'BDTR/' written in Python. This demo code has been tested on Python 2.7 and Tensorflow v0.11. `(Better Performance)`

- a. Prepare the dataset and the train/test lists as shown in `BDTR/dataset2.py`. The list format is `image_path label`.

- b. Download the pre-trained alexnet model and modify the `BDTR/model.py`. [Alexnet](http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/)

- c. run `python BDTR/bdtr_train.py` to train the network. 

>>(We could adjust the weights to achieve much better performance by assigning larger weights of the ranking loss for the RegDB dataset.)

- d. run `python BDTR/bdtr_eval.py` to evaluate the learned network. 

 Demo results on RegDB dataset of one split *

|Methods | Rank@1 | Rank@5 | Rank@20 |mAP |
| --------   | -----  | ---- | ----  | ----  |
|#TONE  | 15.7% | 28.2% | 45.6% | 18.0% |
|#HCML | 23.4% | 36.8% | 57.5% | 23.9% |
|#BDTR | 34.3% | 49.0% | 67.9% | 33.1% |

(* Note that it may have some fluctuations due to randomly generated training/testing splits. Above results are obtained with the demo code in one split. We randomly conduct the experiments 10 trials to get the average performance in the paper. Better performance could be achieved by adjusting the weights.)



### Citation
Please kindly cite this paper in your publications if it helps your research:
```
@inproceedings{aaai18vtreid,
  title={Hierarchical Discriminative Learning for Visible Thermal Person Re-Identification},
  author={Ye, Mang and Lan, Xiangyuan and Li, Jiawei and Yuen, Pong C.},
  booktitle={AAAI},
  year={2018},
}

@inproceedings{ijcai18vtreid,
  title={Visible Thermal Person Re-Identification via Dual-Constrained Top-Ranking},
  author={Ye, Mang and Wang, Zheng and Lan, Xiangyuan and Yuen, Pong C.},
  booktitle={IJCAI},
  year={2018},
}
```
If you also use the RegDB dataset, please kindly cite:

```
@article{sensors17,
  title={Person Recognition System Based on a Combination of Body Images from Visible Light and Thermal Cameras},
  author={Nguyen, Dat Tien and Hong, Hyung Gil and Kim, Ki Wan and Park, Kang Ryoung},
  journal={Sensors},
  volume={17},
  number={3},
  pages={605},
  year={2017},
}
```
Contact: mangye@comp.hkbu.edu.hk
