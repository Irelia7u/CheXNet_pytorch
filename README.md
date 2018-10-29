# CheXNet_pytorch
This is an implementation of [**CheXNet**](https://arxiv.org/abs/1711.05225) based on pytorch.
- If there is any errors in my code, please contact me, thanks!
## Dataset
- You can download the chest x-ray images from [**NIH**](https://nihcc.app.box.com/v/ChestXray-NIHCC)
## Train
- fine_tune_densenet.py  
## Test
- test_densenet.py
## Comparsion
- For a fair comparison, we follow the publicly available data split list.
|     Pathology      | Orignial CheXNet | Our Implemented |
| :----------------: | :--------------: | :-------------: |
|    Atelectasis     |      0.8094      |      0.7756     |
|    Cardiomegaly    |      0.9248      |      0.8873     |
|      Effusion      |      0.8638      |      0.8311     |
|    Infiltration    |      0.7345      |      0.6980     |
|        Mass        |      0.8676      |      0.8219     |
|       Nodule       |      0.7802      |      0.7640     |
|     Pneumonia      |      0.7680      |      0.7166     |
|    Pneumothorax    |      0.8887      |      0.8517     |
|   Consolidation    |      0.7901      |      0.7519     |
|       Edema        |      0.8878      |      0.8469     |
|     Emphysema      |      0.9371      |      0.9042     |
|      Fibrosis      |      0.8047      |      0.8280     |
| Pleural Thickening |      0.8062      |      0.7693     |
|       Hernia       |      0.9164      |      0.9314     |
|         Avg        |      0.8414      |      0.8127     |
## Environment
- python 3
- pytorch 0.4.0
