import random
from models.detector import *
from dgl.data.utils import load_graphs
import os

class Dataset:
    def __init__(self, name='reddit', prefix='/home/hdou/model/GADBench-master/datasets/'):
        graph = load_graphs(prefix + name)[0][0]
        self.name = name
        self.graph = graph

    def split(self, trial_id=0):
        self.graph.ndata['train_mask'] = self.graph.ndata['train_masks'][:,trial_id]
        self.graph.ndata['val_mask'] = self.graph.ndata['val_masks'][:,trial_id]
        self.graph.ndata['test_mask'] = self.graph.ndata['test_masks'][:,trial_id]
      

model_detector_dict = {
    'MGADN': BaseGNNDetector,
    # 'GCN': BaseGNNDetector,
    # 'GraphSAGE': BaseGNNDetector,
    # 'GAT': BaseGNNDetector,
    # 'GAS': GASDetector,
    # 'PCGNN': PCGNNDetector,
    # 'AMNet': BaseGNNDetector,
    # 'BWGNN': BaseGNNDetector,
    # 'GHRN': GHRNDetector,
}

def save_results(results, file_id):
    if not os.path.exists('results/'):
        os.mkdir('results/')
    if file_id is None:
        file_id = 0
        while os.path.exists('results/{}.xlsx'.format(file_id)):
            file_id += 1
    results.transpose().to_excel('results/{}.xlsx'.format(file_id))
    print('save to file ID: {}'.format(file_id))
    return file_id

