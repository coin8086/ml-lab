import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor

from neural_network import NeuralNetwork
from operations import train_loop, test_loop

def run(download_only = False, data_root='/tmp/data', test_only = False, load_file_path = None, save_file_path = None, epochs = 5):
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

    train_dataloader = DataLoader(training_data, batch_size=64)
    test_dataloader = DataLoader(test_data, batch_size=64)

    #######################################
    # Build Model
    #

    device = ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    if load_file_path:
        # Specify map_location so that GPU trained model can be loaded to (cpu) device
        model = torch.load(load_file_path, map_location=device)
        model.eval()
    else:
        model = NeuralNetwork().to(device)

    print(model)

    #######################################
    # Train and Test Model
    #

    loss_fn = nn.CrossEntropyLoss()

    if test_only:
        test_loop(test_dataloader, model, loss_fn, device)
        return

    learning_rate = 1e-3
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer, device)
        test_loop(test_dataloader, model, loss_fn, device)

    print("Done!")

    #######################################
    # Save Model
    #

    if save_file_path:
        torch.save(model, save_file_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", dest="download_only", action="store_true", help="download training data and exit without training")
    parser.add_argument("-r", dest="data_root", default="/tmp/data", help="root directory for training and test data")
    parser.add_argument("-t", dest="test_only", action="store_true", help="test a model without training")
    parser.add_argument("-l", dest="load_file_path", help="load trained model from the file path")
    parser.add_argument("-s", dest="save_file_path", help="save trained model to the file path")
    parser.add_argument("-e", dest="epochs", type=int, default=5, help="epochs to train a model")
    args = parser.parse_args()

    run(
        download_only=args.download_only,
        data_root=args.data_root,
        test_only=args.test_only,
        load_file_path=args.load_file_path,
        save_file_path=args.save_file_path,
        epochs=args.epochs)

if __name__ == '__main__':
    main()