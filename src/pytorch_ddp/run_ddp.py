import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor

from neural_network import NeuralNetwork
from operations import train_loop, test_loop

# For DDP, refer to https://pytorch.org/tutorials/intermediate/ddp_tutorial.html

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def partition_dataset(dataset, partition_count, partition_idx):
    assert partition_idx < partition_count and partition_idx >= 0

    data_size = len(dataset)
    partition_size = data_size // partition_count
    start_idx = partition_idx * partition_size
    if start_idx + partition_size > data_size:
        stop_idx = data_size
    else:
        stop_idx = start_idx + partition_size

    return Subset(dataset, range(start_idx, stop_idx))


def run(rank, world_size, download_only = False, data_root='/tmp/data', test_only = False, load_file_path = None, save_file_path = None, epochs = 5):
    #######################################
    # Load Data
    #

    training_data = datasets.FashionMNIST(
        root=data_root,
        train=True,
        download=True,
        transform=ToTensor()
    )

    test_data = datasets.FashionMNIST(
        root=data_root,
        train=False,
        download=True,
        transform=ToTensor()
    )

    if download_only:
        return

    training_data = partition_dataset(training_data, world_size, rank)
    train_dataloader = DataLoader(training_data, batch_size=64)
    test_dataloader = DataLoader(test_data, batch_size=64)

    #######################################
    # Build Model
    #

    # Create model and move it to GPU with id rank
    device_id = rank % torch.cuda.device_count()
    nn_model = NeuralNetwork().to(device_id)
    ddp_model = DDP(nn_model, device_ids=[device_id])

    if load_file_path:
        # Configure map_location properly
        map_location = { 'cuda:%d' % 0: 'cuda:%d' % device_id }
        ddp_model.load_state_dict(torch.load(load_file_path, map_location=map_location))
        ddp_model.eval()

    print(ddp_model)

    #######################################
    # Train and Test Model
    #

    loss_fn = nn.CrossEntropyLoss()

    if test_only:
        test_loop(test_dataloader, ddp_model, loss_fn, device_id)
        return

    learning_rate = 1e-3
    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=learning_rate)
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, ddp_model, loss_fn, optimizer, device_id)
        test_loop(test_dataloader, ddp_model, loss_fn, device_id)
    print("Done!")

    #######################################
    # Save and Load Model
    #

    if save_file_path and rank == 0:
        # All processes should see same parameters as they all start from same
        # random parameters and gradients are synchronized in backward passes.
        # Therefore, saving it in one process is sufficient.
        torch.save(ddp_model.state_dict(), save_file_path)

        # Also save the internal model
        file, _ = os.path.splitext(save_file_path)
        torch.save(ddp_model.module, file + ".model.pt")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", dest="download_only", action="store_true", help="download training data and exit without training")
    parser.add_argument("-r", dest="data_root", default="/tmp/data", help="root directory for training and test data")
    parser.add_argument("-t", dest="test_only", action="store_true", help="test a model without training")
    parser.add_argument("-l", dest="load_file_path", help="load trained model from the file path")
    parser.add_argument("-s", dest="save_file_path", help="save trained model to the file path")
    parser.add_argument("-e", dest="epochs", type=int, default=5, help="epochs to train a model")
    parser.add_argument("-w", dest="world_size", type=int, help="total number of parellel processes on all nodes")
    args = parser.parse_args()

    if not args.download_only:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        print(f"Start running DDP on rank {rank}.")
    else:
        rank = None

    run(rank, args.world_size,
        download_only=args.download_only,
        test_only=args.test_only,
        load_file_path=args.load_file_path,
        save_file_path=args.save_file_path,
        epochs=args.epochs)

    if not args.download_only:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()