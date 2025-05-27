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
    vectors = [item[label_name] for item in batch]

    # Stack labels in case of multi-dimensional output
    labels = torch.stack(vectors)

    # set to float32
    batch_events = batch_events.float()
    labels = labels.float()
    
    return batch_events, labels, event_lengths

def pad_or_truncate_pulse(event, max_seq_length=config['input_data']['seq_dim'], charge_index=int(0)):
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
        event = event[event[:, charge_index].argsort(descending=True)]
        truncated_event = event[:max_seq_length]
        return truncated_event, max_seq_length
    

    # Pad if the sequence is too short
    elif seq_length < max_seq_length:
        padding = torch.zeros((max_seq_length - seq_length, event.size(1)))
        padded_event = torch.cat([event, padding], dim=0)
        return padded_event,  seq_length
    
    # No need to pad or truncate if it's already the correct length
    return event, seq_length

def custom_collate_fn_pulse(batch, max_seq_length=config['input_data']['seq_dim']):
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
    
    padded_events, event_lengths = zip(*[pad_or_truncate_pulse(event, max_seq_length) for event in events])

    batch_events = torch.stack(padded_events)
    event_lengths = torch.tensor(event_lengths)

    if reconstruction_target == 'dir3vec': # exception for dir3vec pulse, since graphnet does not have dir3vec in dataset class
        zenith = torch.tensor([item['zenith'] for item in batch])
        azimuth = torch.tensor([item['azimuth'] for item in batch])

        # Calculate a 3D unit-vector from the zenith and azimuth angles
        x_dir = torch.sin(zenith) * torch.cos(azimuth)
        y_dir = torch.sin(zenith) * torch.sin(azimuth)
        z_dir = torch.cos(zenith)

        # Stack to dir3vec tensor
        labels = torch.stack([x_dir, y_dir, z_dir], dim=-1)

    else:
        # Extract labels and convert to tensors
        label_name = reconstruction_target

        # Extract labels and convert to tensors (3D vectors)
        vectors = [item[label_name] for item in batch]

        # Stack labels in case of multi-dimensional output
        labels = torch.stack(vectors)

    # set to float32
    batch_events = batch_events.float()
    labels = labels.float()
    
    return batch_events, labels, event_lengths

def make_dataloader_PMT(
        root_dir,
        dataset_id,
        training_parts,
        validation_parts,
        batch_size=config['training_params']['batch_size'],
        num_workers=config['input_data']['num_workers'],
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
    truth_paths=train_paths,
    )

    val_set = PMTfiedDatasetPyArrow(
    truth_paths=val_paths,
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