<div align="center">

# Compositional Few-Shot Testing (CFST)

![Language](https://img.shields.io/badge/language-Python-brightgreen)
[[<img src="" />Poster](https://neurips.cc/media/PosterPDFs/NeurIPS%202023/73708.png?t=1701659113.7268684) ]
[[OpenReview](https://openreview.net/forum?id=38bZuqQOhC)]
![License](https://img.shields.io/badge/license-CCBY4.0-yellow)

</div>

**The Official Repository for dataset CGQA, COBJ, and 
NeurIPS2023 Paper "Does Continual Learning Meet Compositionality? New Benchmarks and An Evaluation Framework" ([link](https://neurips.cc/virtual/2023/poster/73708)).** 

This project provides two benchmarks and one evaluation protocol 
focusing on evaluating the compositionality of a continual learner.

Dataset Description: [link](https://liaoweiduo.notion.site/db205c8f05954a17a5c5246ca77e4074?v=a20095dc6c5c40de855d076ed7714b3a&pvs=25).

---
## Requirements 

* **Standard**: 
In order to use our datasets and dataloader, the minimal requirement is [PyTorch](https://pytorch.org/).

* **Avalanche**: 
For the convenience to test on popular CL methods, we use [Avalanche](https://avalanche.continualai.org/getting-started/how-to-install)==0.3.1. 
If you want to use Avalanche=0.4.0, please upgrade PyTorch to 2.1.1 as recommended by the official guide.  

* **Docker**: 
We also provide our docker image 
[here](https://hub.docker.com/repository/docker/liaoweiduo/avalanche/general).

* **Additional Requirements** (Recommended for ViT data augmentation): 
`einops`, `pandas`, `timm`. They can be easily installed by `Pip`. 

---
## Installation

The minimal requirement to use our dataset and dataloader is PyTorch. 
Below we will show a full installation for reproducing experimental results in our paper using Avalanche. 

### Conda

```bash
conda create --name CFST python=3.7
conda activate CSFT
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install avalanche-lib[all]
pip install einops
pip install pandas
pip install timm
```

### Docker

```bash
docker run --name CSFT [--runtime=nvidia OR --gpus all] -itd -v [PATH_FROM]:[PATH_TO] liaoweiduo/avalanche:1.0
docker attach CSFT
pip install einops
pip install pandas
pip install timm
```

> Note: If you encounter import issue when running `from PIL import Image`, uninstall and reinstall it may solve this issue. 
> 
> ```bash
> pip uninstall Pillow
> pip install Pillow
> ```

### CGQA and COBJ

Our benchmarks can be accessed in icloud: [CGQA](https://www.icloud.com/iclouddrive/0986u6Zo6Maa5hgntK1jHvUmA#CGQA), [COBJ](https://www.icloud.com/iclouddrive/0485XJRF0e2nbeuWgOu3nDYMQ#COBJ) (also in [OneDrive](https://portland-my.sharepoint.com/personal/weiduliao2-c_my_cityu_edu_hk/_layouts/15/onedrive.aspx?ga=1&id=%2Fpersonal%2Fweiduliao2%2Dc%5Fmy%5Fcityu%5Fedu%5Fhk%2FDocuments%2FSharing%2FCFST&view=0) and [hugging face](https://huggingface.co/datasets/jiangmingchen/CGQA_and_COBJ)). 
Extract them into two folder named `CGQA` and `COBJ` under `CFST` folder.
JSON file includes meta-information (e.g., concept combinations, concept positions, and concept features) and is needed for the dataloader. 
The folder structure is as follows: 
```Markdown
\CFST
├─CGQA
├ └─GQA_100
├   ├─continual
├   ├ ├─test: [IMAGES]; test.json
├   ├ ├─train: [IMAGES]; train.json
├   ├ └─val: [IMAGES]; val.json
├   └─fewshot
├     ├─non_comp: [IMAGES]; non_comp_fewshot.json     (noc)
├     ├─non_novel: [IMAGES]; non_novel_fewshot.json   (non)
├     ├─pro: [IMAGES]; pro_fewshot.json
├     ├─sub: [IMAGES]; sub_fewshot.json
├     └─sys: [IMAGES]; sys_fewshot.json
└─COBJ 
  └─annotations
    ├─continual
    ├ ├─test: [IMAGES]
    ├ ├─train: [IMAGES]
    ├ └─val: [IMAGES]
    ├─fewshot
    ├ ├─non_comp: [IMAGES]
    ├ ├─non_novel: [IMAGES]
    ├ ├─pro: [IMAGES]
    ├ ├─sub: [IMAGES]
    ├ └─sys: [IMAGES]
    ├─O365_continual_test_crop.json
    ├─O365_continual_train_crop.json
    ├─O365_continual_val_crop.json
    ├─O365_noc_fewshot_crop.json
    ├─O365_non_fewshot_crop.json
    ├─O365_pro_fewshot_crop.json
    └─O365_sys_fewshot_crop.json
```

### Constructing For Your Own Benchmark

TODO: JIANG

need meta-information about 

---
## Benchmark Usage

`datasets` folder contains the python file used to load dataset. 

The following two python files DO NOT NEED avalanche to be installed. 
- `cgqa_general.py` and `cobj_general.py`. 

The following two python files NEED avalanche to be installed. 
- `cgqa.py` and `cobj.py`

Each file contains 2 functions `continual_training_benchmark` and `fewshot_testing_benchmark` 
which will return the specific datasets (or avalanche_benchmarks) and the corresponding label information. 
For `cgqa_general.py` and `cobj_general.py`, each image instance contains two items (image tensor, label).
For `cgqa.py` and `cobj.py`, each image instance contains three items (image tensor, label, task_id). 

### Load ten 10-way continual training tasks
The number of labels for continual training is 100 for CGQA (30 for COBJ). 
These labels are randomly placed into ten continual tasks (each with 10 labels) controlled by `seed`. 
In each task, a class mapping is used to map the original label to a relative label.
Relative labels are either starting from 0 to 9 in each task (`return_task_id=True`) or 
from 0 to 99 for all tasks (`return_task_id=False`),
since `return_task_id` indicates the task-IL setting or class-IL setting. 

```python
import datasets.cgqa_general as cgqa

print(f"Load continual training benchmark.")
benchmark = cgqa.continual_training_benchmark(n_experiences=10, seed=1234, return_task_id=True, 
                                              dataset_root='PATH TO CFST FOLDER (e.g., ~/datasets/CFST)')

# Benchmark statistics
print(f"Relative labels in each experience: \n{benchmark.classes_in_exp}")
print(f"Original labels in each experience: \n{benchmark.original_classes_in_exp}")
print(f"String label for each original label: \n{benchmark.label_info[2]}")

# Load samples in train_datasets/val_datasets/test_datasets
print(f'Benchmark contains {len(benchmark.train_datasets)} experiences.')
# same for benchmark.val_datasets and benchmark.test_datasets
print(f'exp:0 contains {len(benchmark.train_datasets[0])} images.')
print(f'Each instance contains two items (image tensor, label): tensor {benchmark.train_datasets[0][0][0].shape}, label {benchmark.train_datasets[0][0][1]}.')

"""OUTPUTS:
Load continual training benchmark.
Relative labels in each experience (10 experiences, each contains 10 labels): 
[[0 1 2 3 4 5 6 7 8 9]
 [0 1 2 3 4 5 6 7 8 9]
 ...
 [0 1 2 3 4 5 6 7 8 9]
 [0 1 2 3 4 5 6 7 8 9]]
Original labels in each experience (10 experiences, each contains 10 labels): 
[[40 35 81 61 98 68 85 27 39 42]
 [33 59 63 94 56 87 96  1 71 82]
 ...
 [50 99 73 80 69 58 90 89 43 30]
 [26 23 49 15 24 76 53 38 83 47]]
String label for each original label: 
{0: ('bench', 'building'), 1: ('bench', 'chair'), 2: ('bench', 'door'),...

Benchmark contains 10 experiences.
exp:0 contains 10000 images.
Each instance contains two items (image tensor, label): tensor torch.Size([3, 128, 128]), label 8.

"""
```

### Load 300 independent few-shot tasks for specific mode (sys, pro,...)
As for compositional few-shot testings, we prepare 100 labels for each mode. 
In each task, images are randomly sampled according to the setting 
(`n_way`, `n_shot`, `n_val`, `n_query`, 
indicates the number of labels, the number of support samples for each label, 
the number of validation samples for each label, and the number of query samples for each label, respectively). 
All these tasks are independent and the relative labels are starting from `0` to `n_way-1`. 
```python
import datasets.cgqa_general as cgqa

print(f"Load few-shot testing benchmark.")
benchmark = cgqa.fewshot_testing_benchmark(n_experiences=300, mode='sys', seed=1234,
                                           dataset_root='PATH TO CFST FOLDER (e.g., ~/datasets/CFST)')

# Load samples in train_datasets/val_datasets/test_datasets
print(f'Benchmark contains {len(benchmark.train_datasets)} experiences.')
# same for benchmark.val_datasets and benchmark.test_datasets
print(f'exp:0 support set contains {len(benchmark.train_datasets[0])} images, '
      f'validate set contains {len(benchmark.val_datasets[0])} images, '
      f'query set contains {len(benchmark.test_subsets[0])} images.')
print(f'Each instance contains two items (image tensor, label): tensor {benchmark.train_datasets[0][0][0].shape}, label {benchmark.train_datasets[0][0][1]}.')

"""OUTPUTS:
Load few-shot testing benchmark.

Benchmark contains 300 experiences.
exp:0 support set contains 100 images, validate set contains 50 images, query set contains 100 images.
Each instance contains two items (image tensor, label): tensor torch.Size([3, 128, 128]), label 0.

"""
```

### Train a continual learner (Avalanche)
We train a naive (Finetune) continual learner from scratch for task-IL setting as an example. 

```python
import torch
from torch.nn import CrossEntropyLoss
import avalanche as avl
from avalanche.training import Naive
from avalanche.training.plugins import EvaluationPlugin, EarlyStoppingPlugin, LRSchedulerPlugin
from avalanche.evaluation import metrics as metrics

from datasets import cgqa
from models.resnet import get_resnet

# Load continual training benchmark in task-IL setting
benchmark = cgqa.continual_training_benchmark(n_experiences=10, seed=1234, return_task_id=True, 
                                              dataset_root='PATH TO CFST FOLDER (e.g., ~/datasets/CFST)')

# Define resnet18 with multiple heads for task-IL setting
model = get_resnet(multi_head=True, initial_out_features=10)

# Define evaluation plugin
wandb_logger = avl.logging.WandBLogger(project_name='CGQA', run_name='task-il')
wandb_logger.wandb.watch(model)
loggers = [
    avl.logging.InteractiveLogger(), 
    wandb_logger
]

metrics_list = [
    metrics.accuracy_metrics(epoch=True, experience=True, stream=True),
    metrics.loss_metrics(epoch=True, experience=True, stream=True),
    metrics.forgetting_metrics(experience=True, stream=True),
    metrics.forward_transfer_metrics(experience=True, stream=True),
    metrics.timing_metrics(epoch=True)]

evaluation_plugin = EvaluationPlugin(
    *metrics_list,
    # benchmark=benchmark,
    loggers=loggers)

# Define naive strategy
optimizer = torch.optim.Adam(model.parameters(), lr=8e-3)
epochs = 100
plugins = [
    LRSchedulerPlugin(scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, 1e-6)), 
    EarlyStoppingPlugin(patience=5, val_stream_name='val_stream')
]
strategy = Naive(model, optimizer, CrossEntropyLoss(),
                 train_mb_size=100, train_epochs=epochs, eval_mb_size=50, device=torch.device("cuda:0"),
                 plugins=plugins, evaluator=evaluation_plugin, eval_every=1, peval_mode="epoch")

# Start training 
results = []
for exp_idx, (experience, val_task) in enumerate(zip(benchmark.train_stream, benchmark.val_stream)):
    print("Start of experience ", experience.current_experience)
    print("Current Classes: ", experience.classes_in_this_experience)
    print("Current Classes (str): ", [
        benchmark.label_info[2][cls_idx]
        for cls_idx in benchmark.original_classes_in_exp[experience.current_experience]
    ])

    strategy.train(experience, eval_streams=[val_task])
    print("Training experience completed")

    print("Computing accuracy on the whole test set.")
    result = strategy.eval(benchmark.test_stream)
    results.append(result)

```

### Test the trained continual learner on compositional few-shot tasks (Avalanche)
After obtain this trained model, the next step is to evaluate its feature extractor on 300 few-shot tasks for each mode. 

```python
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
import avalanche as avl
from avalanche.training import Naive
from avalanche.training.plugins import EvaluationPlugin, LRSchedulerPlugin
from avalanche.evaluation import metrics as metrics

from datasets import cgqa
from models.resnet import MTResNet18, get_resnet

# Back up the feature extractor from the trained model. 
model : MTResNet18  # Previously learned from the last section
back_up_state_dict_feature_extractor = model.resnet.state_dict()

n_experiences = 300
for mode in ['sys', 'pro', 'sub', 'non', 'noc']: 
    print(f'mode: {mode}')
    # Load continual training benchmark in task-IL setting
    benchmark = cgqa.fewshot_testing_benchmark(n_experiences=n_experiences, seed=1234, mode=mode, 
                                               dataset_root='PATH TO CFST FOLDER (e.g., ~/datasets/CFST)')
    
    # Define evaluation plugin
    loggers = [avl.logging.InteractiveLogger()]
    
    metrics_list = [
        metrics.accuracy_metrics(epoch=True, experience=True, stream=True),
        metrics.loss_metrics(epoch=True, experience=True, stream=True)
    ]
    
    evaluation_plugin = EvaluationPlugin(
        *metrics_list,
        # benchmark=benchmark,
        loggers=loggers)

    # Start training 
    results, accs = [], []
    for exp_idx, experience in enumerate(benchmark.train_stream):
        print("Start of experience ", experience.current_experience)
        print("Current Classes: ", experience.classes_in_this_experience)
        print("Current Classes (str): ", [
            benchmark.label_info[2][cls_idx]
            for cls_idx in benchmark.original_classes_in_exp[experience.current_experience]
        ])
        
        # Reuse model's feature extractor
        model = get_resnet(multi_head=False, initial_out_features=10)   
        # We do not need the previously learned classifier
        model.resnet.load_state_dict(back_up_state_dict_feature_extractor)
        # Freeze the parameters of the feature extractor
        for param in model.resnet.parameters(): 
            param.requires_grad = False
        
        # Define naive strategy
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        epochs = 20
        plugins = [
            LRSchedulerPlugin(scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, 1e-6))
        ]
        strategy = Naive(model, optimizer, CrossEntropyLoss(),
                         train_mb_size=100, train_epochs=epochs, device=torch.device("cuda:0"),
                         plugins=plugins, evaluator=evaluation_plugin)
    
    
        strategy.train(experience)
        print("Training experience completed")
    
        print("Computing accuracy.")
        result = strategy.eval(benchmark.test_stream[experience.current_experience])
        results.append(result)
    
        task_id_str = '%03d' % experience.current_experience    # 0  -> 000
        print(f"Top1_Acc_Stream/eval_phase/test_stream/Task{task_id_str}: ",
              results[-1][f'Top1_Acc_Stream/eval_phase/test_stream/Task{task_id_str}'])
        accs.append(results[-1][f'Top1_Acc_Stream/eval_phase/test_stream/Task{task_id_str}'])

    print('###################################')
    print('accs:', accs)

    avg_test_acc = np.mean(accs)
    std_test_acc = np.std(accs)
    ci95_test_acc = 1.96 * (std_test_acc/np.sqrt(n_experiences))
    print(f'Top1_Acc_Stream/eval_phase/test_stream: '
          f'{avg_test_acc*100:.2f}% +- {ci95_test_acc * 100:.2f}%)')
```

---

## Cite

If you used our benchmark for research purpose, please remember to cite our reference paper published at the 
[Benchmark @ NeurIPS 2023](TODO) track: 
["Does Continual Learning Meet Compositionality? New Benchmarks and An Evaluation Framework"](TODO).
This will help inspire researchers to prioritize compositionality during the development of continual learners:

```
@inproceedings{liao2023does,
    title={Does Continual Learning Meet Compositionality? New Benchmarks and An Evaluation Framework},
    author={Weiduo Liao and Ying Wei and Mingchen Jiang and Qingfu Zhang and Hisao Ishibuchi},
    booktitle={Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
    year={2023},
}
```