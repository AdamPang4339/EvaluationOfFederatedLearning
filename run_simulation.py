from typing import List, Union, Tuple, Dict, Optional
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from datasets.utils.logging import disable_progress_bar
from torch.utils.data import DataLoader
import math

from collections import OrderedDict

import flwr
from flwr.server.client_proxy import ClientProxy
from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import Metrics, Context, EvaluateRes, FitRes, Scalar, parameters_to_ndarrays
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
from flwr.simulation import run_simulation
from flwr_datasets import FederatedDataset
from flwr.common.typing import Parameters

import argparse
import sys

DEVICE = torch.device("cpu")
NUM_CLIENTS = 10
BATCH_SIZE = 64
disable_progress_bar()
with_poison = False
folder_path = ""


class Net(nn.Module):
    """
    Neural network class
    """
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 62)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# MetricsWriter class in charge of writing all metrics results to a CSV file
class MetricsWriter:
    def __init__(self, filename: str):
        self.filename = filename
        self.file_path = os.path.join(folder_path, self.filename)

        os.makedirs(folder_path, exist_ok=True)

        if not os.path.exists(self.file_path):
            self.metrics = pd.DataFrame(columns=["client_id", "loss", "accuracy"])
        else:
            self.metrics = pd.read_csv(self.file_path)

    def write_per_client(self, client_id: int, loss: float, accuracy: float, trust: float, 
                         reputation: float, round: int):
        new_row = pd.DataFrame({"client_id": client_id, "loss": loss, "accuracy": accuracy, 
                                "trust": trust, "reputation": reputation, "round": round}, index=[0])
        self.metrics = pd.concat([self.metrics, new_row], ignore_index=True)
        self.metrics.to_csv(self.file_path, index=False)

    def write_aggregated(self, round: int, loss: float, accuracy: float):
        new_row = pd.DataFrame({"round": round, "agg_loss": loss, "agg_accuracy": accuracy}, index=[0])
        self.metrics = pd.concat([self.metrics, new_row], ignore_index=True)
        self.metrics.to_csv(self.file_path, index=False)

class FlowerClient(NumPyClient):
    """ This is a class for a Federated Learning Client.

    It defines the methods that a client can use for local training
    and evaluation of the local model. Each instnace of the class represents
    a single client.
    """
    def __init__(self, net, trainloader, valloader, partition_id):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.partition_id = partition_id

    def get_parameters(self, config):
        """ Get the parameters of the model """
        return get_parameters(self.net)

    def fit(self, parameters, config):
        """ Get parameters from server, train model, return to server.

        This method recieves the model parameters from the server and then
        trains the model on the local data. After that, the updated model
        parameters are returned to the server.
        """
        set_parameters(self.net, parameters)
        train(self.net, self.trainloader, epochs=1)
        return get_parameters(self.net), len(self.trainloader), {"client_id": self.partition_id}

    def evaluate(self, parameters, config):
        """ Get parameters from server, evaluate model, return to server.

        This method recieves the model parameters from the server and then
        evaluates the model on the local data. After that, the evaluation
        results are returned to the server.
        """
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader)

        # writer = MetricsWriter(filename="metrics_per_client.csv")
        # writer.write_per_client(client_id=self.partition_id, loss=loss, accuracy=accuracy)

        return float(loss), len(self.valloader), {"client_id": self.partition_id, "accuracy": float(accuracy), "loss": float(loss)}
    
class AggregateCustomMetricStrategy(FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trusts = {}
        self.reputations = {}

        # This is a hyperparameter that controls the influence previous
        #   reputations have on the current one
        #   Write-up doesn't specify what to set this to so I chose 0.5
        self.alpha = 0.5

        # This is the trust threshold that a client must surpass to remain
        #   in the training pool
        self.trust_threshold = 0.15


    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[List[np.ndarray], Dict[str, Scalar]]:
        """Aggregate fit parameters and calculate L2 distance-based metrics."""

        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        aggregated_parameters_ndarray = parameters_to_ndarrays(aggregated_parameters)
        aggregated_parameters_flat = np.concatenate([arr.flatten() for arr in aggregated_parameters_ndarray])

        # Dictionary to store client normalized distances
        client_distances = {}

        # runs through the clients and calculates the normalized L2 distance
        #   from the center of the major ML model's parameters' cluster to all
        #   the models
        for _, fit_res in results:
            client_parameters_ndarray = parameters_to_ndarrays(fit_res.parameters)
            client_parameters_flat = np.concatenate([arr.flatten() for arr in client_parameters_ndarray])

            l2_distance = 0.0
            for aggregated_param, client_param in zip(aggregated_parameters_flat, client_parameters_flat):
                l2_distance += np.sum((aggregated_param - client_param)**2)

            normalized_l2_distance = np.log(1 + np.sqrt(l2_distance))

            client_distances[fit_res.metrics["client_id"]] = normalized_l2_distance
        
        # # Calculate the mean, max, and min of the distances
        # mean_distance = np.mean(list(client_distances.values()))
        # max_distance = np.max(list(client_distances.values()))
        # min_distance = np.min(list(client_distances.values()))

        # # Normalize the distances to be between 0 and 1
        # for client_proxy, distance in client_distances.items():
        #     print(f"UNNORMALIZED DISTANCE: {distance}")
        #     normalized_distance = (distance - mean_distance) / (max_distance - min_distance)
        #     normalized_distance = np.clip(normalized_distance, 0, 1)  # Ensure it stays within 0 and 1
        #     client_distances[client_proxy] = normalized_distance

        # Calculate mean and standard deviation for Z-score normalization
        mean_distance = np.mean(list(client_distances.values()))
        std_distance = np.std(list(client_distances.values()))

        # Apply Z-score normalization
        z_scores = {client_id: (distance - mean_distance) / std_distance for client_id, distance in client_distances.items()}

        # Find min and max Z-scores to rescale to [0, 1]
        z_min = min(z_scores.values())
        z_max = max(z_scores.values())

        # Rescale Z-scores to [0, 1]
        for client_id, z_score in z_scores.items():
            rescaled_distance = (z_score - z_min) / (z_max - z_min)
            client_distances[client_id] = np.clip(rescaled_distance, 0, 1)

        # set reputations
        # if it's the first round, we don't have previous reputations
        #   so we start with 1 - distance
        if server_round == 1:
            for client_id, distance in client_distances.items():
                self.reputations[client_id] = 1 - distance
                # self.reputations[client_proxy] = 0
        
        # otherwise, we use the equations provided in the write up
        else:
            current_rep = 0
            # do the other thing
            for client_id, distance in client_distances.items():
                if distance < self.alpha:
                    left_side = self.reputations[client_id] + distance
                    right_side = self.reputations[client_id] / server_round
                    current_rep = 1 - (left_side - right_side)

                    self.reputations[client_id] = max(0, min(1, current_rep))
                else:
                    left_side = self.reputations[client_id] + distance
                    right_side = math.pow(math.e, -(1 - (distance*(self.reputations[client_id] / server_round))))
                    current_rep = 1 - (left_side - right_side)

                    self.reputations[client_id] = max(0, min(1, current_rep))


        # set trusts
        # trusts are calculated using the equation provided in the write up
        #   and they are based uponm the current reputations and distances
        for client_id, rep in self.reputations.items():
            d = client_distances[client_id]
            print("############################")
            print(f"Client: {client_id}")
            print(f"DISTANCE d: {d:.8f}   \t REP r: {rep}   ###############################")
            first_root = math.sqrt(math.pow(rep, 2) + math.pow(d, 2))
            second_root = math.sqrt(math.pow(1 - rep, 2) + math.pow((1 - d), 2))
            trust = (first_root - second_root)
            print(f"PRE-CLAMP TRUST: {trust:.10f}")
            # convert any trust values greater than 1 or less than 0
            #   to 1 or 0 respectively
            epsilon = 1e-8  # A very small value to prevent trust from being 0
            trust = (1 - max(0, min(1, trust))) * ((1 - d) + epsilon)
            # Reclamp in the case of epsilon bringing trust beyond 1
            trust = max(0, min(1, trust))
            
            print(f"TRUST: {trust:.10f}")
            print()


            self.trusts[client_id] = trust

        return aggregated_parameters, {"trusts": self.trusts}


    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation accuracy using weighted average."""

        if not results:
            return None, {}

        # Call aggregate_evaluate from base class (FedAvg) to aggregate loss and metrics
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(
            server_round, results, failures
        )

        # now we make a sperate dictionary to store only results
        #   that surpass the threshold
        filtered_results = {}
        for _, evaluate_res in results:
            if self.trusts[evaluate_res.metrics["client_id"]] >= self.trust_threshold:
                filtered_results[evaluate_res.metrics["client_id"]] = evaluate_res

        # Weigh accuracy of each client by number of examples used
        accuracies = [r.metrics["accuracy"] * r.num_examples for r in filtered_results.values()]
        examples = [r.num_examples for r in filtered_results.values()]

        # Aggregate accuracy
        print(examples)
        aggregated_accuracy = sum(accuracies) / sum(examples)

        # Write aggregated accuracy to our metrics CSV file
        writer = MetricsWriter(filename="metrics.csv")
        writer.write_aggregated(round=server_round, loss=aggregated_loss, accuracy=aggregated_accuracy)

        # For each of our client's results, write the client id, trust, reputation, loss and accuracy to our metrics CSV file
        for _, r in results:
            writer.write_per_client(client_id=r.metrics["client_id"], loss=r.metrics["loss"], 
                                    accuracy=r.metrics["accuracy"], trust=self.trusts[r.metrics["client_id"]], 
                                    reputation=self.reputations[r.metrics["client_id"]], round=server_round)


        # Return aggregated loss and metrics (i.e., aggregated accuracy)
        return aggregated_loss, {"accuracy": aggregated_accuracy}

def set_parameters(net, parameters: List[np.ndarray]):
    """ Used to update the local model with parameters received from the server """
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

def get_parameters(net) -> List[np.ndarray]:
    """ Used to get the parameters from the local model """
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def load_unpoisoned_datasets(partition_id: int):
    '''
    Function loads different parts of the FEMNIST dataset depending on the partition
    ID specified as a parameter. We have 10 clients in our simulation, so we will be using
    10 partition IDs.
    '''
    # Split FEMNIST dataset into 10 partitions
    fds = FederatedDataset(
        dataset="flwrlabs/femnist",
        partitioners={"train": NUM_CLIENTS}
    )
    # femnist["train"].features["character"].int2str(value)
    partition = fds.load_partition(partition_id)

    # Divide data in each partition: 80% train, 20% test
    # Note: We use a set seed to make sure the results are reproducible
    partition_train_test = partition.train_test_split(test_size=0.2, seed=21)

    # Normalize and convert images to PyTorch tensors for more stable training
    pytorch_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5), (0.5))]
    )

    def apply_transforms(batch):
        '''
        Applies transformations of pytorch_transforms on every image in place
        '''
        batch["image"] = [pytorch_transforms(img) for img in batch["image"]]
        return batch

    # Create train/val for each partition and wrap it into DataLoader
    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(
        partition_train_test["train"], batch_size=BATCH_SIZE, shuffle=True
  )

    # Generate DataLoaders for the validation and test data batches
    # Generate testset as the whole dataset
    valloader = DataLoader(partition_train_test["test"], batch_size=BATCH_SIZE)
    testset = fds.load_split("train").with_transform(apply_transforms)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE)

    return trainloader, valloader, testloader

def load_poisoned_datasets(partition_id: int):
    '''
    Function loads different parts of the FEMNIST dataset depending on the partition
    ID specified as a parameter. We have 10 clients in our simulation, so we will be using
    10 partition IDs.
    '''
    # Split FEMNIST dataset into 10 partitions
    fds = FederatedDataset(
        dataset="flwrlabs/femnist",
        partitioners={"train": NUM_CLIENTS}
    )
    partition = fds.load_partition(partition_id)

    # Divide data in each partition: 80% train, 20% test
    # Note: We use a set seed to make sure the results are reproducible
    partition_train_test = partition.train_test_split(test_size=0.2, seed=21)

    # Normalize and convert images to PyTorch tensors for more stable training
    pytorch_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5), (0.5))]
    )

    def apply_transforms(batch):
        '''
        Applies transformations of pytorch_transforms on every image in place
        '''
        batch["image"] = [pytorch_transforms(img) for img in batch["image"]]

        if (partition_id == 0):
            for i in range(len(batch["character"]) // 2):
                batch["character"][i] = (batch["character"][i] + 1) % 62

        return batch

  # Create train/val for each partition and wrap it into DataLoader
    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(
        partition_train_test["train"], batch_size=BATCH_SIZE, shuffle=True
    )

    # Generate DataLoaders for the validation and test data batches
    # Generate testset as the whole dataset
    valloader = DataLoader(partition_train_test["test"], batch_size=BATCH_SIZE)
    testset = fds.load_split("train").with_transform(apply_transforms)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE)

    return trainloader, valloader, testloader

def train(net, trainloader, epochs: int, verbose=False):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())
    net.train()
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for batch in trainloader:
            images, labels = batch["image"].to(DEVICE), batch["character"].to(DEVICE)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # Metrics
            epoch_loss += loss
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total
        if verbose:
            print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")

def test(net, testloader):
    """Evaluate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for batch in testloader:
            images, labels = batch["image"].to(DEVICE), batch["character"].to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy

def client_fn(context: Context) -> Client:
    """Create a Flower client representing a single organization.

    This method will return an instance of a particular client to call
    fit or evaluate. The client is then discarded after use. This frees up
    memory usage since all clients and the server are being hosted on the
    same machine.
    """

    # Load model
    net = Net().to(DEVICE)

    # Load data (CIFAR-10)
    # Note: each client gets a different trainloader/valloader, so each client
    # will train and evaluate on their own unique data partition
    # Read the node_config to fetch data partition associated to this node
    partition_id = context.node_config["partition-id"]

    if (with_poison):
        trainloader, valloader, _ = load_poisoned_datasets(partition_id=partition_id)
    else:
        trainloader, valloader, _ = load_unpoisoned_datasets(partition_id=partition_id)

    # Create a single Flower client representing a single organization
    # FlowerClient is a subclass of NumPyClient, so we need to call .to_client()
    # to convert it to a subclass of `flwr.client.Client`
    return FlowerClient(net, trainloader, valloader, partition_id).to_client()

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}

def server_fn(context: Context) -> ServerAppComponents:
    """Construct components that set the ServerApp behaviour.

    Uses an instance of ServerConfig and the Federated Learning strategy
    to create a ServerAppComponents object containing all of the settings
    that define the behavior of the ServerApp.
    """

    # Create FedAvg strategy
    # This is the Federated Learning strategy that details the approach
    #   to the federated learning. This one uses the built-in
    #   Federated Averaging (FedAvg) with some customizations.

    strategy = AggregateCustomMetricStrategy(
        fraction_fit=1.0,  # Sample 100% of available clients for training
        fraction_evaluate=1.0,  # Sample 100% of available clients for evaluation
        min_fit_clients=10,  # Never sample fewer than 10 clients for training
        min_evaluate_clients=10,  # Never sample fewer than 10 clients for evaluation
        min_available_clients=10,  # Wait until all 10 clients are available
    )

    # Configure the server for 5 rounds of training
    config = ServerConfig(num_rounds=5)

    return ServerAppComponents(strategy=strategy, config=config)

def visualize_results(file_name):
    # Read the CSV file
    file_path = os.path.join(folder_path, file_name)

    data = pd.read_csv(file_path)

    # Fill in round number for
    data['round'] = data['round'].fillna(method='ffill')

    # Drop rows where 'client_id' is NaN, to avoid plotting aggregated data as individual clients
    client_data = data.dropna(subset=['client_id'])

    # Create a figure with subplots
    fig, axis = plt.subplots(3, 2, figsize=(12, 10))
    fig.suptitle('Federated Learning Metrics', fontsize=16)

    # 1. History of loss per client over rounds
    for client_id in client_data['client_id'].unique():
        client_df = client_data[client_data['client_id'] == client_id]
        axis[0, 0].plot(client_df['round'], client_df['loss'], marker='o', label=f'Client {client_id}')

    axis[0, 0].set_title('Loss per Client Over Rounds')
    axis[0, 0].set_xlabel('Rounds')
    axis[0, 0].set_ylabel('Loss')
    axis[0, 0].legend()
    axis[0, 0].grid()

    # 2. History of average loss among clients over rounds
    average_loss = data.groupby('round')['agg_loss'].mean()
    axis[0, 1].plot(average_loss.index, average_loss.values, marker='o', color='orange')
    axis[0, 1].set_title('Average Loss Among Clients Over Rounds')
    axis[0, 1].set_xlabel('Rounds')
    axis[0, 1].set_ylabel('Average Loss')
    axis[0, 1].grid()

    # 3. History of evaluation accuracy per client over rounds
    for client_id in client_data['client_id'].unique():
        client_df = client_data[client_data['client_id'] == client_id]
        axis[1, 0].plot(client_df['round'], client_df['accuracy'], marker='o', label=f'Client {client_id}')

    axis[1, 0].set_title('Evaluation Accuracy per Client Over Rounds')
    axis[1, 0].set_xlabel('Rounds')
    axis[1, 0].set_ylabel('Accuracy')
    axis[1, 0].legend()
    axis[1, 0].grid()

    # 4. History of average evaluation accuracy among clients over rounds
    average_accuracy = data.groupby('round')['agg_accuracy'].mean()
    axis[1, 1].plot(average_accuracy.index, average_accuracy.values, marker='o', color='green')
    axis[1, 1].set_title('Average Evaluation Accuracy Among Clients Over Rounds')
    axis[1, 1].set_xlabel('Rounds')
    axis[1, 1].set_ylabel('Average Accuracy')
    axis[1, 1].grid()

    # 5. History of trust per client over rounds
    for client_id in client_data['client_id'].unique():
        client_df = client_data[client_data['client_id'] == client_id]
        axis[2, 0].plot(client_df['round'], client_df['trust'], marker='o', label=f'Client {client_id}')

    axis[2, 0].set_title('Trust per Client Over Rounds')
    axis[2, 0].set_xlabel('Rounds')
    axis[2, 0].set_ylabel('Accuracy')
    axis[2, 0].legend()
    axis[2, 0].grid()

    # 6. History of reputation per client over rounds
    for client_id in client_data['client_id'].unique():
        client_df = client_data[client_data['client_id'] == client_id]
        axis[2, 1].plot(client_df['round'], client_df['reputation'], marker='o', label=f'Client {client_id}')

    axis[2, 1].set_title('Reputation per Client Over Rounds')
    axis[2, 1].set_xlabel('Rounds')
    axis[2, 1].set_ylabel('Accuracy')
    axis[2, 1].legend()
    axis[2, 1].grid()

    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.savefig(os.path.splitext(file_path)[0])
    print(f"[+] Metrics saved to {file_path}.")

def main(arg):
    global with_poison
    global folder_path
    base_path = os.path.dirname(__file__)

    if arg == "p" or arg == "P":
        print("Running simulation with data poisoning on Client 0.")
        with_poison = True
        folder_path = os.path.join(base_path, "results", "attack")
    elif arg == "u" or arg == "U":
        print("Running simulation withOUT data poisoning.")
        with_poison = False
        folder_path = os.path.join(base_path, "results", "no_attack")
    else:
        print("Error: Invalid option. Please use 'p' or 'u' as the argument. ")
        sys.exit(1)       

    print("[+] Processing FEMNIST dataset...")
    if (with_poison):
        trainloader, valloader, testloader = load_poisoned_datasets(partition_id=0)
    else:
        trainloader, valloader, testloader = load_unpoisoned_datasets(partition_id=0)
    print("[+] FEMNIST dataset partitioned and processed.")

    net = Net().to(DEVICE)

    print("[+] Training global model...")
    for epoch in range(3):
        train(net, trainloader, 2)
        loss, accuracy = test(net, valloader)
        print(f"Epoch {epoch+1}: validation loss {loss}, accuracy {accuracy}")

    loss, accuracy = test(net, testloader)
    print(f"Final test set performance:\n\tloss {loss}\n\taccuracy {accuracy}\n")

    print("[+] Loading Federated Learning model components...")
    # Create the ClientApp
    client = ClientApp(client_fn=client_fn)


    # Create the ServerApp
    server = ServerApp(server_fn=server_fn)
    backend_config = {"client_resources": {"num_cpus": 1, "num_gpus": 0.0}}

    print("[+] Running simulation...")
    run_simulation(
        server_app=server,
        client_app=client,
        num_supernodes=NUM_CLIENTS,
        backend_config=backend_config,
    )

    print("[+] Metrics generated. Generating plots...")
    visualize_results(file_name="metrics.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("poison_flag", help="Specify whether you want a poisoned run or unpoisoned run.")
    args = parser.parse_args()
    
    main(args.poison_flag)