## Hierarchical Discriminative Learning for Visible Thermal Person Re-Identification

Demo code for [Hierarchical Discriminative Learning for Visible Thermal Person Re-Identification](http://www.comp.hkbu.edu.hk/~mangye/files/aaai18.pdf) in AAAI 2018.

![](http://www.comp.hkbu.edu.hk/~mangye/files/aaai18_framework.jpg)

### 1. Prepare the dataset.

> The RegDB dataset can be downloaded from this [website](http://dm.dongguk.edu/link.html) by submitting a copyright form.

### 2. Two-stream CNN network feature learning (TONE)

This demo code has been tested on Python 2.7 and Tensorflow v0.11.

> a. Prepare the dataset and the train/test lists as shown in "TONE/dataset.py".

> b. Download the pre-trained alexnet model and modify the "TONE/model.py". [Alexnet](http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/)

> c. run "TONE/tone_train.py" to train the network. 

> d. run "TONE/tone_eval.py" to evaluate the learned features and extract features for later metric learning. 

(You can also modify the scripts to get different layers of features or evaluate different parameters. A trained model of one split can be downloaded on [BaiduYun](https://pan.baidu.com/s/1kVaMkPx) and [GoogleDrive](https://drive.google.com/open?id=1v2-Cry-9O5ZhHySLpMbsr-BJfe6Zxhe5))


### 3. Hierarchical Cross-modality Metric Learning (HCML)
This demo code has been tested on Matlab 2017a.

> a. Make sure the format of the extracted features is correct. Demo Features on [BaiduYun](https://pan.baidu.com/s/1kVaMkPx) and [GoogleDrive](https://drive.google.com/open?id=1v2-Cry-9O5ZhHySLpMbsr-BJfe6Zxhe5)

> b. run "HCML/demo_hcml" to evaluate the cross-modal metric learning results.


- Demo results on RegDB dataset of one split *

|Methods | Rank@1 | Rank@5 | Rank@20 |mAP |
| --------   | -----  | ---- | ----  | ----  |
|#TONE  | 15.7% | 28.2% | 45.6% | 18.0% |
|#HCML | 23.4% | 36.8% | 57.5% | 23.9% |

(* Note that it may have some fluctuations due to randomly generated training/testing splits. We randomly conduct the experiments 10 trials to get the average performance. Details are shown in the paper.)


### Citation
Please kindly cite this paper in your publications if it helps your research:
```
@inproceedings{aaai18vtreid,
  title={Hierarchical Discriminative Learning for Visible Thermal Person Re-Identification},
  author={Ye, Mang and Lan, Xiangyuan and Li, Jiawei and Yuen, Pong C.},
  booktitle={AAAI},
  year={2018},
}
```

Contact: mangye@comp.hkbu.edu.hk
