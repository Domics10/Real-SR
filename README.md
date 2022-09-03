# Changes and configuration made for Tencent/Real-SR to work on HC18 on Docker or Colab

## Forked version of https://github.com/Tencent/Real-SR, please refer to the official repository. This repository is only made for research pourpose regarding Ultra-sound image Super Resolution.

# INDEX

0. Path preparation and dependencies using Google Colab
1. Dataset preparation
2. Training Phase
3. Inference Phase
4. Execution using Docker

---

## 0. **Path preparation and dependencies using Google Colab**
Install the following libraries required for script execution::
```bash
!pip install numpy opencv-python lmdb pyyaml
!pip install tb-nightly future
```
Then download the code from the following github: 
```bash
!git clone https://github.com/Domics10/Real-SR
```
## 1. **Dataset preparation**

Download dataset HC18 at :https://zenodo.org/record/1327317#.YpSdTu7P2iM however we already provided a clean copy of the dataset.

Then will be necessary to download the github for the generation of the kernels:

```bash
!cd preprocess && git clone https://github.com/sefibk/KernelGAN
```
Let's proceed to the kernel generation:
```bash
!cd ./preprocess/KernelGAN && CUDA_VISIBLE_DEVICES=0 python3 train.py --X4 --input-dir ../HC18/train_set
```

### Training dataset generation
Run the following code to create the HR and LR images for the selected dataset:
```bash
!python3 ./preprocess/create_kernel_dataset.py --dataset hc18 --artifacts clean --kernel_path ./preprocess/KernelGAN/results
```
After to inject noise in the generated LR images:
```bash
!python3 ./preprocess/collect_noise.py --dataset hc18 --artifacts clean
```
If missing, download the file : **RRDB_PSNR_x4** downloaded from: https://drive.google.com/drive/u/0/folders/17VYV_SoZZesU6mbxz2dMAIccSSlqLecY in the pretrained_models folder.
## 2. **Training Phase**

To run the training:
```bash
!CUDA_VISIBLE_DEVICES=0 python3 train.py -opt options/dped/train_kernel_noise.yml
```

### 2.0 Resume training
To resume an interrupted training, make the following changes; go to the codes / options / deped / train_kernel_noise.yml folder and edit "resume_state" as for example :
```yml
resume_state: ../experiments/hc18_kernel_noise/training_state/5000.state
```
Then run the training command.

## 3. **Inference Phase**
You can edit the file **test_dped.yml** in the folder options/dped

After the training go in the folder experiments/hc18_kernel_noise/models: move the file **current_step_G.pth**, where "current_step" è is the last step of the training phase (5000, 10000, 20000, etc.), then move the file in pretrained_models and change the **current_step_G.pth** in the file codes/option/dped/test_dped.yml, as for example: 
```yml
pretrained_model_G: ./pretrained_model/current_step_G.pth
```

Inference results will be saved in the folder results/hc18_results .

To run the test, execute the following line:
```py
CUDA_VISIBLE_DEVICES=1,0 python3 test.py -opt options/dped/test_dped.yml
```
## 4. **Esecuzione con Docker**

To use Docker and entrypoint.sh, simply run the following commands:
```bash
docker build -t image_name .
```
Then run the following script to launch the container execution:
```bash
docker run --rm --gpus all --env MODE=mode -it -v volume --name container_name image_name
```
mode can be:
* kc: kernel creation
* ka: kernel application to images
* na: collecting noise
* tr: Training phase
* te: Testing phase

Depending on the value of MODE, the volumes may change, these are working examples:
* kc: /mnt/disk1/vrai/CVDL2022/Tencent/Real-SR/codes/preprocess/KernelGAN/results:/home/Tencent/Real-SR/codes/preprocess/KernelGAN/results
* ka: /mnt/disk1/vrai/CVDL2022/Tencent/Real-SR/datasets:/home/Tencent/Real-SR/datasets
* na: /mnt/disk1/vrai/CVDL2022/Tencent/Real-SR/datasets:/home/Tencent/Real-SR/datasets
* tr: /mnt/disk1/vrai/CVDL2022/Tencent/Real-SR/experiments:/home/Tencent/Real-SR/experiments
* te: /mnt/disk1/vrai/CVDL2022/Tencent/Real-SR/codes/results:/home/Tencent/Real-SR/codes/results
