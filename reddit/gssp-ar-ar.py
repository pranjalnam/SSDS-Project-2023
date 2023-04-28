import argparse
import os
import warnings
from random import uniform
from time import sleep, time

import dgl
import torch
import torch.distributed as dist
import torch.nn.functional as F

from model import GCN

warnings.filterwarnings("ignore")

device = torch.device('cpu')

def get_param_grad_shape(model):
    total_params = None
    for param_idx, param in enumerate(model.parameters()):
        if not param_idx:
            total_params = param.data.flatten()
        else:
            total_params = torch.cat((total_params, param.data.flatten()))

    return total_params.shape


def flatten_model_params(model, flag_return_flattened_shape=False):
    total_params = None
    for param_idx, param in enumerate(model.parameters()):
        if not param_idx:
            total_params = param.data.flatten()
        else:
            total_params = torch.cat((total_params, param.data.flatten()))
    if flag_return_flattened_shape:
        return total_params, total_params.shape
    else:
        return total_params


def average_params(model, group, num_participants):
    total_params = flatten_model_params(model=model)
    dist.all_reduce(total_params, op=dist.ReduceOp.SUM, group=group)
    total_params /= num_participants
    return total_params


def update_model_params(model, flattened_tensor):
    running_idx = 0
    for param_idx, param in enumerate(model.parameters()):
        param.data = flattened_tensor[running_idx: running_idx + param.data.numel()].reshape(
            param.data.shape)
        running_idx += param.data.numel()


def flatten_model_gradients(model, flag_return_flattened_shape=False):
    total_params = None
    for param_idx, param in enumerate(model.parameters()):
        if not param_idx:
            total_params = param.grad.data.flatten()
        else:
            total_params = torch.cat((total_params, param.grad.data.flatten()))

    if flag_return_flattened_shape:
        return total_params, total_params.shape
    else:
        return total_params


def average_gradients(model, group, num_participants):
    total_params = flatten_model_gradients(model=model)

    dist.all_reduce(total_params, op=dist.ReduceOp.SUM, group=group)
    total_params /= num_participants

    return total_params


def update_model_gradients(model, flattened_tensor):
    running_idx = 0
    for param_idx, param in enumerate(model.parameters()):
        param.grad.data = flattened_tensor[running_idx: running_idx + param.grad.data.numel()].reshape(
            param.grad.data.shape)
        running_idx += param.grad.data.numel()


def get_flattened_parameters(model, get_flattened_shape = False):
    total_params = None
    for param_idx, param in enumerate(model.parameters()):
        if not param_idx:
            total_params = param.data.flatten()
        else:
            total_params = torch.cat((total_params, param.data.flatten()))

    if get_flattened_shape:
        return total_params, total_params.shape
    else:
        return total_params


def set_model_params_from_flattened_tensor(model, flattened_tensor):
    running_idx = 0
    for param_idx, param in enumerate(model.parameters()):
        param.data = flattened_tensor[running_idx: running_idx + param.data.numel()].reshape(param.data.shape)
        running_idx += param.data.numel()


def sync_initial_weights(model):
    flattened_tensor = get_flattened_parameters(model=model)
    dist.broadcast(tensor=flattened_tensor, src=0)
    set_model_params_from_flattened_tensor(model=model, flattened_tensor=flattened_tensor)


def run(comm_group, leaders_comm_group):
    logging = [[], [], []]
    # Check if the current process is a group leader
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
    model = GCN(subgraph.ndata['feat'].shape[1], 16, NUM_CLASSES)
    sync_initial_weights(model=model)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print(f"Process {WORLD_RANK} starting training.")

    features = subgraph.ndata['feat']
    labels = subgraph.ndata['label']
    train_mask = subgraph.ndata['train_mask']
    test_mask = subgraph.ndata['test_mask']
    flattened_grad_tensor_shape = get_param_grad_shape(model=model)

    training_st = val_st = time()
    time_tensor = torch.Tensor([training_st, val_st])
    dist.broadcast(time_tensor, src=0)
    training_st = time_tensor[0].item()
    val_st = time_tensor[0].item()
    avg_epoch_time, avg_comm_time = 0, 0
    for e in range(MAX_EPOCHS):
        epoch_st = time()
        comm_time = 0
        logits = model(subgraph, features)
        pred = logits.argmax(1)
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        if rank_to_processtype_dict[WORLD_RANK] == "fast process":
            sleep(uniform(0, 0.5))
        if rank_to_processtype_dict[WORLD_RANK] == "medium process":
            sleep(uniform(0.4, 0.45))
        if rank_to_processtype_dict[WORLD_RANK] == "slow process":
            sleep(uniform(0.8, 0.85))
        loss.backward()
        comm_st = time()
        flattened_grad_update = average_gradients(model, group=comm_group, num_participants=NUM_PROCS_IN_COMM_GROUP)
        update_model_gradients(model=model, flattened_tensor=flattened_grad_update)
        comm_time += time() - comm_st
        if time() - val_st >= 45 and is_group_leader:
            comm_st = time()
            updated_flattened_params = average_params(model=model, group=leaders_comm_group, num_participants=NUM_GROUPS)
            update_model_params(model=model, flattened_tensor=updated_flattened_params)
            comm_time += time() - comm_st
            # if WORLD_RANK in [0, 3, 5]:
            #     print(f"Rank - {WORLD_RANK} :: Epoch Time - {avg_epoch_time / e:.3f} :: Comm Time - {avg_comm_time / e:.3f}")
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
                print(f"-------------Time elapsed - {time() - training_st - val_time :.3f} sec | global training loss - {global_loss:.3f} | test acc - {global_test_acc:.3f}---------------")
                # print(logging)
            val_st = time()
        optimizer.step()
        avg_epoch_time += time() - epoch_st
        avg_comm_time += comm_time


def init_processes(backend):
    print(f"Initializing process - {WORLD_RANK}")
    dist.init_process_group(backend, rank=WORLD_RANK, world_size=WORLD_SIZE)

    # making the comm group list
    comm_group_dict = {}  # ps rank is key, worker ranks are values
    for i in range(NUM_GROUPS):
        comm_group_dict[list(processtype_to_rank_dict.keys())[i]] = dist.new_group(ranks=processtype_to_rank_dict[list(processtype_to_rank_dict.keys())[i]])

    group_leader_group_list = [processtype_to_rank_dict[key][0] for key in processtype_to_rank_dict.keys()]

    group_leader_comm_group = dist.new_group(ranks=group_leader_group_list)
    # run(in_tensor=in_tensor, comm_group=comm_group_dict[rank_to_processtype_dict[WORLD_RANK]], file_store= file_store)

    run(comm_group=comm_group_dict[rank_to_processtype_dict[WORLD_RANK]], leaders_comm_group=group_leader_comm_group)



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




