import dgl

data = dgl.data.RedditDataset()
graph = data[0]
dgl.distributed.partition_graph(graph, graph_name='reddit', num_parts=8,
                                out_path='reddit-part-8',
                                balance_ntypes=graph.ndata['train_mask'])