import argparse
import os
import warnings
from random import uniform
from time import sleep, time

import dgl
import torch
import torch.distributed as dist
import torch.nn.functional as F

from helper import get_flattened_parameters, set_model_params_from_flattened_tensor, \
    switch_bw_tensor_and_byte
from model import GCN

warnings.filterwarnings("ignore")

device = torch.device('cpu')


def average_gradients(model, comm_group):

    total_params = None
    for param_idx, param in enumerate(model.parameters()):
        if not param_idx:
            total_params = param.grad.data.flatten()
        else:
            total_params = torch.cat((total_params, param.grad.data.flatten()))

    dist.all_reduce(total_params, op=dist.ReduceOp.SUM, group=comm_group)
    total_params /= NUM_PROCS_IN_COMM_GROUP

    running_idx = 0
    for param_idx, param in enumerate(model.parameters()):
        param.grad.data = total_params[running_idx: running_idx + param.grad.data.numel()].reshape(param.grad.data.shape)
        running_idx += param.grad.data.numel()


def sync_initial_weights(model):
    flattened_tensor = get_flattened_parameters(model=model)
    dist.broadcast(tensor=flattened_tensor, src=0)
    set_model_params_from_flattened_tensor(model= model, flattened_tensor= flattened_tensor)


def update_model_params_from_fileStore(file_store, model, flattened_tensor_shape):

    temp = torch.zeros(flattened_tensor_shape)
    for key in processtype_to_rank_dict.keys():
        temp += switch_bw_tensor_and_byte(tensor_or_Byte=file_store.get(key), tensor_to_Byte=False)

    temp = temp / NUM_GROUPS

    set_model_params_from_flattened_tensor(model=model, flattened_tensor=temp)



def run(file_store, comm_group):
    logging = [[], [], []]
    is_group_leader = False
    process_type = rank_to_processtype_dict[WORLD_RANK]
    if WORLD_RANK == processtype_to_rank_dict[rank_to_processtype_dict[WORLD_RANK]][0]:
        is_group_leader = True

    # Range Partitioning
    # NODES_PER_PARTITION = math.ceil(GLOBAL_GRAPH_NUM_NODES / WORLD_SIZE)
    # SUBGRAPH_NODE_RANGE = GLOBAL_GRAPH_NODE_RANGE[WORLD_RANK * NODES_PER_PARTITION: (WORLD_RANK + 1) * NODES_PER_PARTITION]]

    # Hash Partitioning
    NODES_IN_SUBGRAPH = [i for i in range(WORLD_RANK, GLOBAL_GRAPH_NUM_NODES, WORLD_SIZE)]
    SUBGRAPH_NODE_RANGE = GLOBAL_GRAPH_NODE_RANGE[NODES_IN_SUBGRAPH]
    subgraph = dgl.node_subgraph(graph, nodes=SUBGRAPH_NODE_RANGE)
    subgraph = dgl.add_self_loop(subgraph)

    features = subgraph.ndata['feat']
    labels = subgraph.ndata['label']
    train_mask = subgraph.ndata['train_mask']
    test_mask = subgraph.ndata['test_mask']

    model = GCN(subgraph.ndata['feat'].shape[1], 16, NUM_CLASSES)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    flattened_init_params, flattened_param_tensor_shape = get_flattened_parameters(model, True)

    sync_initial_weights(model=model)

    # initializing fileStore
    if is_group_leader:
        file_store.set(process_type, switch_bw_tensor_and_byte(tensor_or_Byte=get_flattened_parameters(model=model),
                                                               tensor_to_Byte=True))

    dist.barrier()

    print(f"Process {WORLD_RANK} starting training.")
    if WORLD_RANK == 0:
        print("RECORDING THE SLOWEST PROCESS")

    training_st = val_st = time()
    avg_epoch_time, avg_comm_time = 0, 0
    for e in range(MAX_EPOCHS):
        epoch_st = time()
        comm_time = 0
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
        comm_st = time()
        average_gradients(model=model, comm_group=comm_group)
        comm_time += time() - comm_st
        optimizer.step()

        if time() - val_st >= 45:
            comm_st = time()
            if is_group_leader:
                file_store.set(process_type, switch_bw_tensor_and_byte(get_flattened_parameters(model), True))
            update_model_params_from_fileStore(file_store= file_store, model= model, flattened_tensor_shape= flattened_param_tensor_shape)
            comm_time += time() - comm_st
            # if WORLD_RANK in [0, 3, 5]:
            #     print(f"Rank - {WORLD_RANK} :: Epoch Time - {(avg_epoch_time + time() - epoch_st) / e:.3f} :: Comm Time - {(avg_comm_time + comm_time) / e:.3f}")
            val_time = time()
            global_logits = model(graph, graph.ndata['feat'])
            global_pred = global_logits.argmax(1)
            global_loss = F.cross_entropy(global_logits[global_train_mask], global_labels[global_train_mask])
            global_test_acc = (global_pred[global_test_mask] == global_labels[global_test_mask]).float().mean()
            val_time = time() - val_time

            logging[0].append((time() - training_st - val_time).__round__(2))
            logging[1].append(global_test_acc.item().__round__(4))
            logging[2].append(global_loss.item().__round__(4))

            if WORLD_RANK == 0:
                print(
                    f"-------------Time elapsed - {time() - training_st - val_time :.3f} sec | global training loss - {global_loss:.3f} | test acc - {global_test_acc:.3f}---------------")
                # print(logging)
            val_st = time()
        avg_epoch_time += time() - epoch_st
        avg_comm_time += comm_time


def init_processes(backend):
    print(f"Initializing process - {WORLD_RANK}")
    dist.init_process_group(backend, rank=WORLD_RANK, world_size=WORLD_SIZE)

    # making the comm group list
    comm_group_dict = {}    # ps rank is key, worker ranks are values
    for i in range(NUM_GROUPS):
        comm_group_dict[list(processtype_to_rank_dict.keys())[i]] = dist.new_group(ranks = processtype_to_rank_dict[list(processtype_to_rank_dict.keys())[i]])

    # making the filestore
    file_store = dist.FileStore("temp_fs", NUM_GROUPS)
    # run(in_tensor=in_tensor, comm_group=comm_group_dict[rank_to_processtype_dict[WORLD_RANK]], file_store= file_store)
    run(file_store= file_store, comm_group= comm_group_dict[rank_to_processtype_dict[WORLD_RANK]])

    dist.barrier()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()
    MAX_EPOCHS = args.epochs
    LR = args.lr

    LOG_INTERVAL = 2

    # Environment variables set by torch.distributed.launch
    LOCAL_RANK = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
    WORLD_SIZE = int(os.environ['OMPI_COMM_WORLD_SIZE'])
    WORLD_RANK = int(os.environ['OMPI_COMM_WORLD_RANK'])

    # 0. setup
    print(f"Rank - {WORLD_RANK} -> Begin File")
    # --------------------- CREATING GLOBAL DATA SET -------------------------------
    data = dgl.data.RedditDataset()
    graph = data[0]
    global_test_mask = graph.ndata['test_mask']
    global_train_mask = graph.ndata['train_mask']
    global_labels = graph.ndata['label']

    NUM_CLASSES = data.num_classes
    GLOBAL_GRAPH_NUM_NODES = graph.number_of_nodes()
    GLOBAL_GRAPH_NODE_RANGE = graph.nodes()
    # ------------------ DONE -------------------------------------------------------
    # making lists of processes
    slow_procs_ranks = [i for i in range(0, 3)]
    med_procs_ranks = [i for i in range(3, 5)]
    fast_procs_ranks = [i for i in range(5, 8)]

    # process-type to rank list mapping
    processtype_to_rank_dict = {"slow process": slow_procs_ranks, "medium process": med_procs_ranks,
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