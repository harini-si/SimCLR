import logging
import argparse
from datasets import load_dataset
#setup argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, help="The name of the dataset to use")
parser.add_argument("--batch_size", type=int, default=32, help="The batch size to use")
parser.add_argument("--num_workers", type=int, default=4, help="The number of workers to use")
parser.add_argument("--learning_rate", type=float, default=0.001, help="The learning rate to use")
parser.add_argument("--epochs", type=int, default=10, help="The number of epochs to train for")

args = parser.parse_args()
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.info("Loading data")
    #load data
    dataset = load_dataset("zh-plus/tiny-imagenet")
    train_data= dataset['train']
    val_data= dataset['val']
    test_data= dataset['test']
    #train