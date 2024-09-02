

# SimCLR Implementation

Paper link: https://arxiv.org/pdf/2002.05709.pdf 

This repository contains an implementation of SimCLR (Simple Framework for Contrastive Learning of Visual Representations) done as a part of my Deep Learning Project. SimCLR is a self-supervised learning framework that uses contrastive learning to learn useful visual representations without labeled data.


## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/harini-si/SimCLR.git
    cd SimCLR
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To train the SimCLR model, run:

```bash
cd src
python main_imagenet.py  --epochs 100 --batch_size 256
```

To evaluate, run :

```bash
cd model
python evaluator.py  
```
