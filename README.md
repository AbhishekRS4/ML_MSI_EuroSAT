# A repo with Ray Torch distributed ML experiments on EuroSAT multi-spectral imagery dataset


## Experiments
* The goal of the project is to perform supervised learning for expanding skills - classification experiments for studying the effects of using Kolmogorov-Arnold Networks (KAN) on the LULC classification model performance and to train models using Ray Torch distributed training on streaming data from huggingface hub without downloading the data to the disk.


## Instruction to run the code
* For running torch based training with the data downloaded to the disk, use the script [src/run_torch_trainer.py](src/run_torch_trainer.py)
* For running ray torch based distributed training with the streaming data directly from huggingface hub, use the script [src/run_ray_torch_trainer.py](src/run_ray_torch_trainer.py)
* For running ray torch based evaluation with the streaming data directly from huggingface hub, use the script [src/run_ray_torch_eval.py](src/run_ray_torch_eval.py)


## Performance Results

| Model         | Bands                        | Val Acc.  | Val F1    | Test Acc. | Test F1   |
| ------------- | ---------------------------- | --------- | --------- | --------- | --------- |
| ResNet        | B, G, R, NIR                 |  0.9744   |  0.9745   |  0.9781   |  0.9782   |
| ResKANet      | B, G, R, NIR                 |  0.9761   |  0.9761   |  0.9798   |  0.9798   |
| SE_ResNet     | B, G, R, NIR                 |  0.9750   |  0.9750   |  0.9780   |  0.9780   |
| SE_ResKANet   | B, G, R, NIR                 |  0.9776   |  0.9776   |  0.9796   |  0.9796   | 

* From the results, it is pretty clear that using KAN layer produces better results than normal dense layer for classification


## Remarks
* The goal is not to get the best performance but upskill to learn Ray Torch distributed training for a tiny streaming dataset and also experiment with KAN and compare the performance with normal dense layer