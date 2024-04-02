# CSGR-Bio

This repository contains the official implementation of __"Cross-Subject Generalizable Representation Learning with Class-Subject Dual Labels for Biosignals"__ algorithm based on PyTorch.

We propose an inter-subject similar loss to learn representations robust to inter-subject variability in biosignals. This loss promotes subject invariance, improves the generalizability of the representation, and allows better representations to be learned even with fewer training subjects. 
The proposed framework consists of two complementary loss functions: (1) [supervised contrastive loss](https://proceedings.neurips.cc/paper_files/paper/2020/file/d89a66c7c80a29b1bdbab0f2a1a94af8-Paper.pdf) and (2) inter-subject similar loss.

## Environment setup
The code is available in Pytorch 2.1.0. See requirments.txt for all prerequisites, and you can also install them using the following command.

```
pip install -r requirements.txt
```

## Running
During training, users receive continuous logging feedback in the terminal. After training, users can check the log file, tensorboard output and saved models under the generated 'Results' folder.

(1) To train the model in a supervised learning approach with cross-entropy loss, try the following command:
```
python Run.py --task 'single model' --dataset mit_bih_afib --epoch 100 --batch-size 128
```

(2) To train the model with only SupCon, try the following command:
After the pre-training is performed, the downstream task is performed as well.
```
python Run.py --task 'rep_frozen' --rep 'supcon' --dataset mit_bih_afib --epoch 100 --batch-size 128
```

(3) To train the model with our propose method, try the following command:
After the pre-training is performed, the downstream task is performed as well.
```
python Run.py --task 'rep_frozen' --rep 'supcon+protosim' --dataset mit_bih_afib --epoch 100 --batch-size 128
```
