### This project is under construction... 
# Description
Here is the Pytorch implementation of the paper:

ADGAN++: Controllable Image Synthesis with Attribute-Decomposed GAN

Guo Pu*, Yifang Men*, Yiming Mao, Yuning Jiang, Wei-ying Ma, Zhouhui Lian
In: IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2022. arXiv preprint (comming soon)

<img src="https://github.com/TrickyGo/ADGAN-plus-plus/blob/main/figure1.png" width="550" height="700">

# Settings
Download https://github.com/vacancy/Synchronized-BatchNorm-PyTorch and put it in models/networks/

Download Pretrained VGG-19 in https://drive.google.com/drive/folders/1GmLecVFnisZYeZ83Sfw0zU9eYu3Hyb5u?usp=sharing

# Quick demo
Download our pretrained model on ADE20K dataset in https://drive.google.com/drive/folders/1GmLecVFnisZYeZ83Sfw0zU9eYu3Hyb5u?usp=sharing

Then run:
```
python transfer_stage1.py \
--how_many 8 \
--name stage1 \
--label_dir demo_images/labels \
--image_dir demo_images/images \
--label_nc 151
```

```
python transfer_stage2.py \
--how_many 8 \
--name stage2 \
--stage1_dir results/stage1/test_latest/images/transfered_image_on_test \
--label_dir demo_images/labels \
--image_dir demo_images/images \
--label_nc 151
```

The results should be as same as demonstrated in the above figure.

# Acknowledgments
This code is based on ADGAN(https://github.com/menyifang/ADGAN), SEAN(https://github.com/ZPdesu/SEAN) and SPADE(https://github.com/NVlabs/SPADE). We thank the authors for sharing their wonderful works!
