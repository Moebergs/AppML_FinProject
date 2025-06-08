import torch
from torch.utils.data import DataLoader
import yaml

import numpy as np
import pandas as pd

from src.dataset import PMTfiedDatasetPyArrow

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

reconstruction_target = config['input_data']['reconstruction_target']

def pad_or_truncate(event, max_seq_length=config['input_data']['seq_dim'], total_charge_index=int(16)):
    """
    Pad or truncate an event to the given max sequence length, and create an attention mask.
    
    Args:
    - event: Tensor of shape (seq_length, feature_dim) where seq_length can vary.
    - max_seq_length: Maximum sequence length to pad/truncate to.
    
    Returns:
    - Padded or truncated event of shape (max_seq_length, feature_dim).
    - Attention mask of shape (max_seq_length) where 1 indicates a valid token and 0 indicates padding.
    """
    seq_length = event.size(0)
    
    # Truncate if the sequence is too long
    if seq_length > max_seq_length:
        # sort the event by total charge
        event = event[event[:, total_charge_index].argsort(descending=True)]
        truncated_event = event[:max_seq_length]
        return truncated_event, max_seq_length
    

    # Pad if the sequence is too short
    elif seq_length < max_seq_length:
        padding = torch.zeros((max_seq_length - seq_length, event.size(1)))
        padded_event = torch.cat([event, padding], dim=0)
        return padded_event,  seq_length
    
    # No need to pad or truncate if it's already the correct length
    return event, seq_length

def custom_collate_fn(batch, max_seq_length=config['input_data']['seq_dim']):
    """
    Custom collate function to pad or truncate each event in the batch.
    
    Args:
    - batch: List of (event, label) tuples where event has a variable length [seq_length, 7].
    - max_seq_length: The fixed length to pad/truncate each event to (default is 512).
    
    Returns:
    - A batch of padded/truncated events with shape [batch_size, max_seq_length, 7].
    - Corresponding labels.
    """
    # Separate events and labels
    events = [item.x for item in batch]  # Each event has shape [seq_length, 7]
    
    padded_events, event_lengths = zip(*[pad_or_truncate(event, max_seq_length) for event in events])

    batch_events = torch.stack(padded_events)
    event_lengths = torch.tensor(event_lengths)

    # Extract labels and convert to tensors
    label_name = reconstruction_target

    # Extract labels and convert to tensors (3D vectors)
    vectors_target = [item[label_name] for item in batch]
    vectors_original = [item.energy_original for item in batch]

    # Stack labels in case of multi-dimensional output
    labels = torch.stack(vectors_target)
    original_energy = torch.stack(vectors_original)
    original_n_doms = torch.tensor([item.n_doms for item in batch])

    zenith_batch = torch.stack([item.zenith for item in batch]).float()

    # set to float32
    batch_events = batch_events.float()
    labels = labels.float()
    original_energy = original_energy.float()
    
    return batch_events, labels, event_lengths, original_energy, original_n_doms, zenith_batch


def make_dataloader_PMT(
        root_dir,
        dataset_id,
        training_parts,
        validation_parts,
        batch_size=config['training_params']['batch_size'],
        num_workers=config['input_data']['num_workers'],
        zenith_threshold=config['input_data']['zenith_threshold'],
        zenith_condition=config['input_data']['zenith_condition']
):
    """
    Create a DataLoader for the PMTfied dataset. Takes data from a single dataset ID.
    
    Args:
    - root_dir: Root directory of the dataset.
    - dataset_id: ID of the dataset.
    - training_parts: List of training parts.
    - validation_parts: List of validation parts.
    
    Returns:
    - train_loader: DataLoader for training data.
    - val_loader: DataLoader for validation data.
    """

    # unpack the training and validation parts such that you can either input a list of lists or a single list
    if isinstance(training_parts[0], list):
        training_parts = np.array(training_parts[0])
    else:
        training_parts = np.array(training_parts)
    if isinstance(validation_parts[0], list):
        validation_parts = np.array(validation_parts[0])
    else:
        validation_parts = np.array(validation_parts)

    train_paths = []
    val_paths = []
    for part in training_parts:
        train_paths.append(f"{root_dir}/{dataset_id}/truth_{part}.parquet")
    for part in validation_parts:
        val_paths.append(f"{root_dir}/{dataset_id}/truth_{part}.parquet")
        


    train_set = PMTfiedDatasetPyArrow(
    truth_paths=train_paths, zenith_threshold=zenith_threshold, zenith_condition=zenith_condition
    )

    val_set = PMTfiedDatasetPyArrow(
    truth_paths=val_paths, zenith_threshold=zenith_threshold, zenith_condition=zenith_condition
    )

    train_loader = DataLoader(
        dataset=train_set,
        collate_fn= custom_collate_fn,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True,
        pin_memory=True,
    )

    val_loader = DataLoader(
        dataset=val_set,
        collate_fn= custom_collate_fn,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True,
        pin_memory=True,
    )

    return train_loader, val_loader


def make_dataloader_PMT_inference(
        root_dir,
        dataset_id,
        inference_parts,
        batch_size=config['training_params']['batch_size'],
        num_workers=config['input_data']['num_workers'],
        zenith_threshold=config['input_data']['zenith_threshold'],
        zenith_condition=config['input_data']['zenith_condition']
):
    """
    Create a DataLoader for the PMTfied dataset for inference. Takes data from a single dataset ID.
    
    Args:
    - root_dir: Root directory of the dataset.
    - dataset_id: ID of the dataset.
    - inference_parts: List of dataset parts for inference.

    Returns:
    - inference_loader: DataLoader for inference data.
    """

    # unpack the training and validation parts such that you can either input a list of lists or a single list
    if isinstance(inference_parts[0], list):
        inference_parts = np.array(inference_parts[0])
    else:
        inference_parts = np.array(inference_parts)


    inference_paths = []
    for part in inference_parts:
        inference_paths.append(f"{root_dir}/{dataset_id[0]}/truth_{part}.parquet")

    print(f"Loading inference data from {inference_paths}")
    event_no_array = np.array([])
    for path in inference_paths:
        pd_truth = pd.read_parquet(path)
        event_no = pd_truth['event_no'].values
        # append to the event_no array
        event_no_array = np.append(event_no_array, event_no)

    # take first 500_000 events
    if len(event_no_array) > 500_000:
        print(f"WARNING: Taking first 500_000 events from {len(event_no_array)} events because of memory constraints")
        event_no_array = event_no_array[:500_000]
        
    inference_set = PMTfiedDatasetPyArrow(
    truth_paths=inference_paths,
    selection=event_no_array,
    zenith_threshold=zenith_threshold,
    zenith_condition=zenith_condition
    )

    inference_loader = DataLoader(
        dataset=inference_set,
        collate_fn= custom_collate_fn,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True,
        pin_memory=True,
    )

    return inference_loader, event_no_array