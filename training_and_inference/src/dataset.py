import numpy as np

import torch

from torch.utils.data import Dataset
from torch_geometric.data import Data

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

def feature_preprocessing(col_name, value) -> np.ndarray:
  
    if col_name in ['dom_x', 'dom_y', 'dom_z', 'dom_x_rel', 'dom_y_rel', 'dom_z_rel']:
        value = value / 500
    elif col_name in ['rde']:
        value = (value - 1.25) / 0.25
    elif col_name in ['pmt_area']:
        value = value / 0.05
    elif col_name in ['q1', 'q2', 'q3', 'q4', 'q5', 'Q25', 'Q75', 'Qtotal']:
        mask = value > 0
        value[mask] = np.log10(value[mask])
    elif col_name in ['t1', 't2', 't3','t4', 't5']:
        mask = value > 0
        value[mask] = (value[mask] - 1.0e04) / 3.0e04
    elif col_name in ['T10', 'T50', 'sigmaT']:
        mask = value > 0
        value[mask] = value[mask] / 1.0e04

    return value


class PMTfiedDatasetPyArrow(Dataset):
    def __init__(
            self, 
            truth_paths,
            selection=None,
            transform=feature_preprocessing,
            zenith_threshold=None,
            zenith_condition=None,
    ):
        '''
        Args:
        - truth_paths: List of paths to the truth files
        - selection: List of event numbers to select from the corresponding truth files
        - transform: Function to apply to the features as preprocessing
        '''

        self.truth_paths = truth_paths
        self.selection = selection
        self.transform = transform

        self.zenith_threshold = zenith_threshold
        self.zenith_condition = zenith_condition

        # Metadata variables
        self.event_counts = []
        self.cumulative_event_counts = []
        self.current_file_idx = None
        self.current_truth = None
        self.current_feature_path = None
        self.current_features = None
        total_events = 0

        # Scan the truth files to get the event counts
        for path in self.truth_paths:
            truth = pq.read_table(path)
            if self.selection is not None:
                mask = pc.is_in(truth['event_no'], value_set=pa.array(self.selection))
                truth = truth.filter(mask)
            
            if self.zenith_threshold is not None and self.zenith_condition is not None:
                zenith_values = truth.column('zenith')
                if self.zenith_condition == 'greater':
                    zenith_mask = pc.greater(zenith_values, self.zenith_threshold)
                elif self.zenith_condition == 'less':
                    zenith_mask = pc.less(zenith_values, self.zenith_threshold)
                else:
                    print(f"No zenith filtering applied for condition: {self.zenith_condition}")
                    zenith_mask = None

                if zenith_mask is not None :
                    truth = truth.filter(zenith_mask)   

            n_events = len(truth)
            self.event_counts.append(n_events)
            total_events += n_events
            self.cumulative_event_counts.append(total_events)

        self.total_events = total_events

    def __len__(self):
        return self.total_events

    def __getitem__(self, idx):
        # Find the corresponding file index
        file_idx = np.searchsorted(self.cumulative_event_counts, idx, side='right')
        
        # Define the truth paths
        truth_path = self.truth_paths[file_idx]

        # Define the local event index
        local_idx = idx if file_idx == 0 else idx - self.cumulative_event_counts[file_idx - 1]

        # Load the truth and apply selection
        if file_idx != self.current_file_idx:
            self.current_file_idx = file_idx
            truth_table_from_disk = pq.read_table(truth_path)
            
            if self.selection is not None:
                mask = pc.is_in(truth_table_from_disk['event_no'], value_set=pa.array(self.selection))
                truth_table_from_disk = truth_table_from_disk.filter(mask)
            
            if self.zenith_threshold is not None and self.zenith_condition is not None:
                zenith_values = truth_table_from_disk.column('zenith')
                zenith_mask = None
                if self.zenith_condition == 'greater':
                    zenith_mask = pc.greater(zenith_values, self.zenith_threshold)
                elif self.zenith_condition == 'less':
                    zenith_mask = pc.less(zenith_values, self.zenith_threshold)
                
                if zenith_mask is not None:
                    truth_table_from_disk = truth_table_from_disk.filter(zenith_mask)
            
            
            self.current_truth = truth_table_from_disk
            
        truth = self.current_truth

        # Get the event details
        event_no = torch.tensor(int(truth.column('event_no')[local_idx].as_py()), dtype=torch.long)
        energy_original = torch.tensor(truth.column('energy')[local_idx].as_py(), dtype=torch.float32)

        offset = int(truth.column('offset')[local_idx].as_py())
        n_doms = int(truth.column('N_doms')[local_idx].as_py())
        part_no = int(truth.column('part_no')[local_idx].as_py())
        shard_no = int(truth.column('shard_no')[local_idx].as_py())

        # n_doms_for_division = torch.tensor(n_doms, dtype=torch.float32).clamp(min=1.0) # Use float, clamp for division
        # energy_per_n = energy_original / n_doms_for_division

        energy = torch.log10(energy_original.clamp(min=1e-7))
        # energy = torch.log10(energy_per_n.clamp(min=1e-7)) 

        # Define the feature path based on the truth path
        feature_path = truth_path.replace('truth_{}.parquet'.format(part_no), '' + str(part_no) + '/PMTfied_{}.parquet'.format(shard_no))

        # x from rows (offset-n_doms) to offset
        start_row = offset - n_doms

        # Load the features and apply preprocessing
        if feature_path != self.current_feature_path:
            self.current_feature_path = feature_path
            self.current_features = pq.read_table(feature_path)

        features = self.current_features

        x = features.slice(start_row, n_doms)
        x = x.drop_columns(['event_no', 'original_event_no'])
        num_columns = x.num_columns

        x_tensor = torch.full((n_doms, num_columns), fill_value=torch.nan, dtype=torch.float32)

        for i, col_name in enumerate(x.column_names):
            value = x.column(i).to_numpy()
            value = value.copy()
            value = self.transform(col_name, value)
            # convert to torch tensor
            value_tensor = torch.from_numpy(value)
            x_tensor[:, i] = value_tensor

        return Data(x=x_tensor, n_doms=n_doms, event_no=event_no, feature_path=feature_path, energy_original=energy_original, energy=energy)
    