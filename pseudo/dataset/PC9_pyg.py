import os
import os.path as osp
import numpy as np
from tqdm import tqdm
import torch
from sklearn.utils import shuffle
import pandas as pd

from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.data import Data, DataLoader


class PC9_3D(InMemoryDataset):
    
    def __init__(self, root = 'dataset/', transform = None, pre_transform = None, pre_filter = None):

        self.path = 'processed.pkl'
        self.folder = osp.join(root, 'pc9')

        super(PC9_3D, self).__init__(self.folder, transform, pre_transform, pre_filter)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return 'processed.pkl'

    @property
    def processed_file_names(self):
        return 'pc9_pyg.pt'

    def download(self):
        #download_url(self.url, self.raw_dir)
        pass
    
    def process(self):
        
        df = pd.read_pickle(osp.join(self.raw_dir, self.raw_file_names))
        data_list = []

        for idx in tqdm(range(len(df))):
            mol = df.iloc[idx]
            
            R_i = torch.tensor(mol.xyz,dtype=torch.float32)
            z_i = torch.tensor(mol.atoms,dtype=torch.int64)
            y_i = [torch.tensor(mol[name],dtype=torch.float32) for name in ['HOMO-1', 'LUMO+1', 'HOMO', 'LUMO', 'gap', 'E']]
            data = Data(pos=R_i, z=z_i, y=y_i[0], homo_1=y_i[0], lumo_1=y_i[1], homo=y_i[2], lumo=y_i[3], gap=y_i[4], e=y_i[5], unlabelled_status = 1)

            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        
        data, slices = self.collate(data_list)

        print('Saving...')
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self, data_size, train_size, valid_size, seed):
        ids = shuffle(range(data_size), random_state=seed)
        train_idx, val_idx, test_idx = torch.tensor(ids[:train_size]), torch.tensor(ids[train_size:train_size + valid_size]), torch.tensor(ids[train_size + valid_size:])
        split_dict = {'train':train_idx, 'valid':val_idx, 'test':test_idx}
        return split_dict

if __name__ == '__main__':
    dataset = PC9_3D()
    print(dataset)
    print(dataset.data.z.shape)
    print(dataset.data.pos.shape)
    