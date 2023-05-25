# GRACE
Grace: Graph Self-Distillation and Completion to Mitigate Degree-Related Biases
![Grace: Graph Self-Distillation and Completion to Mitigate Degree-Related Biases](system_model.png "Model Architecture")

## Installation
1. conda create -n grace python=3.9.11
2. conda activate grace
3. conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
4. conda install pyg -c pyg
5. pip3 install -r requirements.txt

## How to run
```bash
### First stage
python self_distillation.py --dataset cora --gamma 1.0 --seed 1

### Second Stage
python graph_completion.py --dataset cora --seed 1

### Inference Stage
python evaluate.py --dataset cora --k 9 --K 11 --seed 1

```
Note: the random seed controls the dataset split and should be same across different stages.
