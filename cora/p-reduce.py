import argparse
import os
import warnings
from random import uniform
from time import sleep, time

import dgl
import torch
import torch.distributed as dist
import torch.nn.functional as F

from helper import get_flattened_params, update_model_params_from_flattened_tensor
from model import SAGE, GCN

warnings.filterwarnings("ignore")

device = torch.device('cpu')


def sync_initial_weights(model):
    flattened_tensor = get_flattened_params(model=model)
    dist.broadcast(tensor=flattened_tensor, src=0)
    update_model_params_from_flattened_tensor(model=model, flattened_tensor=flattened_tensor)


def master_run(model):
    _, flattened_params_tensor_shape = get_flattened_params(model=model, get_flattened_shape=True)

    while True:
        aggregated_params = torch.zeros(flattened_params_tensor_shape)
        buffer = torch.zeros(flattened_params_tensor_shape[0] + 1)
        process_rank_list = []

        for _ in range(P):
            dist.recv(tensor=buffer)
            aggregated_params += buffer[:-1]
            process_rank_list.append(int(buffer[-1].item()))

        aggregated_params /= P

        # print(f"Master process has updates from ranks = {process_rank_list}")

        for process in process_rank_list:
            dist.send(tensor=aggregated_params, dst=process)


def slave_run(model, subgraph):
    logging = [[], [], []]
    process_type = rank_to_processtype_dict[WORLD_RANK]
    _, flattened_params_tensor_shape = get_flattened_params(model=model, get_flattened_shape=True)

    if WORLD_RANK == 1:
        flattened_tensor = get_flattened_params(model=model)
        for i in range(2, 9):
            dist.send(tensor=flattened_tensor, dst=i)
    else:
        flattened_tensor = torch.zeros(flattened_params_tensor_shape)
        dist.recv(tensor=flattened_tensor, src=1)
        update_model_params_from_flattened_tensor(model=model, flattened_tensor=flattened_tensor)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print(f"Process {WORLD_RANK} starting training.")

    subgraph_test_pct = 0.3
    subgraph_num_data_points = subgraph.number_of_nodes()
    train_mask = torch.zeros(subgraph_num_data_points)
    subgraph_train_indices = torch.randperm(subgraph_num_data_points)[
                             :int(subgraph_num_data_points * (1 - subgraph_test_pct))]
    train_mask[subgraph_train_indices] = 1
    test_mask = torch.ones(subgraph_num_data_points) - train_mask
    train_mask = train_mask.bool()
    test_mask = test_mask.bool()

    features = subgraph.ndata['feat']
    labels = subgraph.ndata['label']

    training_st = val_st = time()

    for e in range(MAX_EPOCHS):
        logits = model(subgraph, features)
        pred = logits.argmax(1)
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])
        train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
        optimizer.zero_grad()
        if rank_to_processtype_dict[WORLD_RANK] == "fast process":
            sleep(uniform(0, 0.5))
        if rank_to_processtype_dict[WORLD_RANK] == "medium process":
            sleep(uniform(0.4, 0.45))
        if rank_to_processtype_dict[WORLD_RANK] == "slow process":
            sleep(uniform(0.8, 0.85))
        loss.backward()
        flattened_params = get_flattened_params(model=model)
        send_buffer = torch.concat((flattened_params, torch.Tensor([WORLD_RANK])))
        dist.send(tensor=send_buffer, dst=master_rank[0])
        dist.recv(tensor=flattened_params, src=master_rank[0])
        update_model_params_from_flattened_tensor(model=model, flattened_tensor=flattened_params)

        optimizer.step()

        if time() - val_st >= 45:
            val_time = time()
            global_logits = model(graph, graph.ndata['feat'])
            global_pred = global_logits.argmax(1)
            global_loss = F.cross_entropy(global_logits[global_train_mask], global_labels[global_train_mask])
            global_test_acc = (global_pred[global_test_mask] == global_labels[global_test_mask]).float().mean()
            val_time = time() - val_time
            logging[0].append((time() - training_st - val_time).__round__(2))
            logging[1].append(global_test_acc.item().__round__(4))
            logging[2].append(global_loss.item().__round__(4))
            if rank_to_processtype_dict[WORLD_RANK] == "slow process":
                print(f"-------------Time elapsed - {time() - training_st - val_time :.3f} sec | global training loss - {global_loss:.3f} | test acc - {global_test_acc:.3f}---------------")
                print(logging)
            val_st = time()


def init_processes(backend):
    print(f"Initializing process - {WORLD_RANK}")
    dist.init_process_group(backend, rank=WORLD_RANK, world_size=WORLD_SIZE)
    model = GCN(graph.ndata['feat'].shape[1], 16, NUM_CLASSES)

    if rank_to_processtype_dict[WORLD_RANK] == "master process":
        master_run(model)
    else:
        # Range Partitioning
        # NODES_PER_PARTITION = math.ceil(GLOBAL_GRAPH_NUM_NODES / WORLD_SIZE)
        # SUBGRAPH_NODE_RANGE = GLOBAL_GRAPH_NODE_RANGE[WORLD_RANK * NODES_PER_PARTITION: (WORLD_RANK + 1) * NODES_PER_PARTITION]]

        # Hash Partitioning
        NODES_IN_SUBGRAPH = [i for i in range(WORLD_RANK, GLOBAL_GRAPH_NUM_NODES, WORLD_SIZE)]
        SUBGRAPH_NODE_RANGE = GLOBAL_GRAPH_NODE_RANGE[NODES_IN_SUBGRAPH]
        subgraph = dgl.node_subgraph(graph, nodes=SUBGRAPH_NODE_RANGE)
        subgraph = dgl.add_self_loop(subgraph)
        slave_run(model, subgraph)

    dist.barrier()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--P", type=int, default=3)
    args = parser.parse_args()
    MAX_EPOCHS = args.epochs
    LR = args.lr
    P = args.P

    LOG_INTERVAL = 2

    # Environment variables set by torch.distributed.launch
    LOCAL_RANK = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
    WORLD_SIZE = int(os.environ['OMPI_COMM_WORLD_SIZE'])
    WORLD_RANK = int(os.environ['OMPI_COMM_WORLD_RANK'])

    # 0. setup
    print(f"Rank - {WORLD_RANK} -> Begin File")
    # --------------------- CREATING GLOBAL DATA SET -------------------------------
    data = dgl.data.CoraFullDataset(raw_dir='../data/')
    graph = data[0]

    # torch.manual_seed(42)
    test_pct = 0.3
    num_data_points = graph.number_of_nodes()
    global_train_mask = torch.zeros(num_data_points)
    train_indices = torch.randperm(num_data_points)[:int(num_data_points * (1 - test_pct))]
    global_train_mask[train_indices] = 1
    global_test_mask = torch.ones(num_data_points) - global_train_mask
    global_train_mask = global_train_mask.bool()
    global_test_mask = global_test_mask.bool()
    features = graph.ndata['feat']
    global_labels = graph.ndata["label"]

    NUM_CLASSES = data.num_classes
    GLOBAL_GRAPH_NUM_NODES = graph.number_of_nodes()
    GLOBAL_GRAPH_NODE_RANGE = graph.nodes()
    # ------------------ DONE -------------------------------------------------------
    # making lists of processes
    master_rank = [0]
    slow_procs_ranks = [i for i in range(1, 6)]
    med_procs_ranks = [i for i in range(6, 11)]
    fast_procs_ranks = [i for i in range(11, 17)]

    # process-type to rank list mapping
    processtype_to_rank_dict = {"master process": master_rank, "slow process": slow_procs_ranks,
                                "medium process": med_procs_ranks,
                                "fast process": fast_procs_ranks}
    
    # number of groups
    NUM_GROUPS = len(processtype_to_rank_dict.keys())

    # rank to process type mapping
    rank_to_processtype_dict = {}
    for i in range(WORLD_SIZE):
        rank_to_processtype_dict[i] = "unassigned"
        for j in processtype_to_rank_dict.keys():
            if i in processtype_to_rank_dict[j]:
                rank_to_processtype_dict[i] = j
    # number of processes in current process group
    NUM_PROCS_IN_COMM_GROUP = len(processtype_to_rank_dict[rank_to_processtype_dict[WORLD_RANK]])

    init_processes(backend="gloo")