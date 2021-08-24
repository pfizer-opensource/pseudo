import os
import os.path as osp
import numpy as np
from tqdm import tqdm
import torch
from sklearn.utils import shuffle
import pandas as pd

from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.data import Data, DataLoader

hartree2ev = 27.211386245988

class PC9_QM9_3D(InMemoryDataset):
    
    def __init__(self, root = 'dataset/', transform = None, pre_transform = None, pre_filter = None):

        self.folder = osp.join(root, 'pc9_qm9')

        super(PC9_QM9_3D, self).__init__(self.folder, transform, pre_transform, pre_filter)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return 'qm9_eV.npz'

    @property
    def processed_file_names(self):
        return 'pc9_qm9_pyg.pt'

    def download(self):
        #download_url(self.url, self.raw_dir)
        pass
    
    def process(self):
        
        data_list = []
        ## the qm9 part
        
        data = np.load(osp.join(self.raw_dir, self.raw_file_names))

        R = data['R']
        Z = data['Z']
        N= data['N']
        split = np.cumsum(N)
        R_qm9 = np.split(R, split)
        Z_qm9 = np.split(Z,split)
        target = {}
        for name in ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve','U0', 'U', 'H', 'G', 'Cv']:
            target[name] = np.expand_dims(data[name],axis=-1)

        data_list = []
        for i in tqdm(range(len(N))):
            R_i = torch.tensor(R_qm9[i],dtype=torch.float32)
            z_i = torch.tensor(Z_qm9[i],dtype=torch.int64)
            y_i = [torch.tensor(target[name][i],dtype=torch.float32) for name in ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve','U0', 'U', 'H', 'G', 'Cv']]
            data = Data(pos=R_i, z=z_i, y=y_i[0], y_ood = torch.tensor(np.array([0]), dtype = torch.float32), loss_weight = torch.tensor(np.array([0]), dtype = torch.float32), mu=y_i[0], alpha=y_i[1], homo=y_i[2], lumo=y_i[3], gap=y_i[4], r2=y_i[5], zpve=y_i[6], U0=y_i[7], U=y_i[8], H=y_i[9], G=y_i[10], Cv=y_i[11], E = torch.tensor(np.array([0]), dtype = torch.float32), unlabelled_status = 0, data_category = 0)

            data_list.append(data)
        
        self.qm9_data_size = len(data_list)
        
        df = pd.read_pickle(osp.join(self.raw_dir, 'processed.pkl'))
       
        for idx in tqdm(range(len(df))):
            mol = df.iloc[idx]
            
            R_i = torch.tensor(mol.xyz,dtype=torch.float32)
            z_i = torch.tensor(mol.atoms,dtype=torch.int64)
            y_i = [torch.tensor(np.array([mol[name] * hartree2ev]),dtype=torch.float32) for name in ['HOMO', 'LUMO', 'gap', 'E']]
            data = Data(pos=R_i, 
                        z=z_i, 
                        y=torch.tensor(np.array([0]), dtype = torch.float32),
                        y_ood = y_i[0],
                        mu=torch.tensor(np.array([0]), dtype = torch.float32), 
                        alpha=torch.tensor(np.array([0]), dtype = torch.float32), 
                        homo=y_i[0], 
                        lumo=y_i[1], 
                        gap=y_i[2],
                        r2=torch.tensor(np.array([0]), dtype = torch.float32), 
                        zpve=torch.tensor(np.array([0]), dtype = torch.float32), 
                        U0=torch.tensor(np.array([0]), dtype = torch.float32), 
                        U=torch.tensor(np.array([0]), dtype = torch.float32), 
                        H=torch.tensor(np.array([0]), dtype = torch.float32), 
                        G=torch.tensor(np.array([0]), dtype = torch.float32), 
                        Cv=torch.tensor(np.array([0]), dtype = torch.float32), 
                        E = y_i[3], 
                        unlabelled_status = 1, 
                        data_category = 1,
                       loss_weight = torch.tensor(np.array([0]), dtype = torch.float32))

            data_list.append(data)
            
        self.total_data_size = len(data_list)
        
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        
        data, slices = self.collate(data_list)

        print('Saving...')
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self, train_size, valid_size, seed):
        qm9_data_size = 130831
        total_data_size = 230065
        ids = shuffle(range(qm9_data_size), random_state=seed)
        train_idx, val_idx, test_idx = torch.tensor(ids[:train_size]), torch.tensor(ids[train_size:train_size + valid_size]), torch.tensor(ids[train_size + valid_size:])
        split_dict = {'train':train_idx, 'valid':val_idx, 'test':test_idx, 'unlabeled': torch.tensor(list(range(qm9_data_size, total_data_size)), dtype = torch.int64)}
        return split_dict

if __name__ == '__main__':
    dataset = PC9_QM9_3D()
    print(dataset)
    print(dataset.data.z.shape)
    print(dataset.data.pos.shape)
    