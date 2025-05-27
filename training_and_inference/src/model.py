import torch
import torch.nn as nn
import pytorch_lightning as pl
import yaml # Ensure this is imported
import numpy as np

from src.loss import dir_3vec_loss, MSE_loss, VonMisesFisherLoss3D, opening_angle_loss, Simon_loss

# This block loads the config to select the global loss_func.
# Make sure the path to config.yaml is correct for your Colab environment.
# This should be the main config file for the project.
try:
    with open('/content/IceCubeTransformer/config.yaml', 'r') as f: # Absolute path for Colab
        config = yaml.safe_load(f)
except FileNotFoundError:
    # Fallback for local testing if config is in the same directory as model.py (less likely for train.py)
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError("config.yaml not found. Please ensure the path is correct in src/model.py for global loss_func selection.")


# set loss func based on the config file but not the string
if config['training_params']['loss_function'] == 'dir_3vec_loss':
    loss_func = dir_3vec_loss
elif config['training_params']['loss_function'] == 'MSE_loss':
    loss_func = MSE_loss
elif config['training_params']['loss_function'] == 'VonMisesFisherLoss3D':
    loss_func = VonMisesFisherLoss3D
elif config['training_params']['loss_function'] == 'opening_angle_loss':
    loss_func = opening_angle_loss
elif config['training_params']['loss_function'] == 'Simon_loss':
    loss_func = Simon_loss
else:
    raise ValueError(f"Unknown loss function specified in config: {config['training_params']['loss_function']}")

class AttentionHead(nn.Module):
    """ 
    Single attention head class:
    - takes in a sequence of embeddings and returns a sequence of the same length
    """
    def __init__(
            self,
            head_dim: int,
            dropout,
            ):
        super(AttentionHead, self).__init__()
        self.head_dim = head_dim
        self.query = nn.Linear(head_dim, head_dim)
        self.key = nn.Linear(head_dim, head_dim)
        self.value = nn.Linear(head_dim, head_dim)
        self.dropout = nn.Dropout(dropout) # This dropout is defined but not used in original forward

    def forward(self, q, k, v, event_lengths=None, seq_dim_kv=None):
        batch_size, seq_dim_q, _ = q.shape # q can have different seq_dim from k,v in cross-attention
        seq_dim_kv = k.shape[1]

        attention_weights =  torch.matmul(q, k.transpose(-2, -1)) 
        attention_weights = attention_weights / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32, device=q.device))

        if event_lengths is not None:
            # Create a mask of shape (batch_size, seq_dim_q, seq_dim_kv)
            # Mask positions where either query token or key/value token is padding
            mask_q = torch.arange(seq_dim_q, device=q.device).unsqueeze(0).unsqueeze(-1) < event_lengths.unsqueeze(-1).unsqueeze(-1) # (B, L_q, 1)
            mask_kv = torch.arange(seq_dim_kv, device=k.device).unsqueeze(0).unsqueeze(0) < event_lengths.unsqueeze(-1).unsqueeze(-1)   # (B, 1, L_kv)
            attention_mask = mask_q & mask_kv # (B, L_q, L_kv)
            attention_weights = attention_weights.masked_fill(~attention_mask, float('-1e9'))

        attention_weights = torch.softmax(attention_weights, dim=-1)
        # The original code had a dropout here commented out, which is common for attention weights
        # attention_weights = self.dropout(attention_weights) 

        output = torch.matmul(attention_weights, v)
        return output


class MultiAttentionHead(nn.Module):
    """ 
    Multi-head attention class:
    - takes in a sequence of embeddings and returns a sequence of the same length
    - uses multiple attention heads in parallel
    """
    def __init__(
            self, 
            embedding_dim, 
            n_heads,
            dropout,
            ):
        super(MultiAttentionHead, self).__init__()
        assert embedding_dim % n_heads == 0, "embedding_dim must be divisible by n_heads"

        self.embedding_dim = embedding_dim
        self.nheads = n_heads
        self.head_dim = embedding_dim // n_heads
        
        # Using ModuleList for heads as in original
        self.heads = nn.ModuleList([AttentionHead(self.head_dim, dropout) for _ in range(n_heads)])

        self.q_proj = nn.Linear(embedding_dim, embedding_dim)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim)

        self.summarize = nn.Linear(embedding_dim, embedding_dim) # Output projection
        self.dropout = nn.Dropout(dropout) # Dropout on the final output

    def forward(self, x, event_lengths=None): # Assuming x is the input for Q, K, V (self-attention)
        batch_size, seq_dim, _ = x.shape

        q_full = self.q_proj(x)
        k_full = self.k_proj(x)
        v_full = self.v_proj(x)

        # Split into heads: (batch_size, n_heads, seq_len, head_dim)
        q = q_full.view(batch_size, seq_dim, self.nheads, self.head_dim).permute(0, 2, 1, 3)
        k = k_full.view(batch_size, seq_dim, self.nheads, self.head_dim).permute(0, 2, 1, 3)
        v = v_full.view(batch_size, seq_dim, self.nheads, self.head_dim).permute(0, 2, 1, 3)

        # Process each head
        head_outputs_list = []
        for i in range(self.nheads):
            # Pass q_head, k_head, v_head for this specific head
            # q[:, i] is (batch_size, seq_len, head_dim)
            head_output = self.heads[i](q[:, i, :, :], k[:, i, :, :], v[:, i, :, :], event_lengths=event_lengths, seq_dim_kv=seq_dim)
            head_outputs_list.append(head_output)
        
        # Concatenate outputs from all heads along the head_dim dimension
        # Each head_output is (batch_size, seq_len, head_dim)
        # Concatenating results in (batch_size, seq_len, n_heads * head_dim) which is (batch_size, seq_len, embedding_dim)
        multihead_output_concat = torch.cat(head_outputs_list, dim=-1) 
        
        # The original code had 'output = self.dropout(multihead_output)' which is incorrect.
        # Dropout should be on the summarized output. And 'multihead_output' was used instead of 'multihead_output_concat'.
        # Also, the original code's self.summarize was applied to multihead_output (which was an un-concatenated list item).
        # Corrected logic:
        summarized_output = self.summarize(multihead_output_concat)
        output = self.dropout(summarized_output) # Apply dropout to the final combined and projected output
        return output

class FeedForward(nn.Module):
    def __init__(
            self, 
            embedding_dim,
            dropout,
            hidden_mult: int = 4, # Standard practice
            ):
        super(FeedForward, self).__init__()
        self.step = nn.Sequential(
            nn.Linear(embedding_dim, hidden_mult * embedding_dim),
            nn.ReLU(), # Or nn.GELU()
            nn.Linear(hidden_mult * embedding_dim, embedding_dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.step(x)

class DecoderBlock(nn.Module): # This is an Encoder block if only self-attention
    def __init__(
            self, 
            embedding_dim,
            n_heads,
            dropout,
            ):
        super(DecoderBlock, self).__init__()
        self.multihead = MultiAttentionHead(embedding_dim, n_heads, dropout)
        self.feedforward = FeedForward(embedding_dim, dropout)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        # It's common to apply dropout to sublayer outputs before residual connection and norm
        # self.dropout_sublayer = nn.Dropout(dropout) 

    def forward(self, x, event_lengths=None):
        attn_output = self.multihead(x, event_lengths=event_lengths)
        # x = x + self.dropout_sublayer(attn_output) # If using sublayer dropout
        x = x + attn_output # Original: x = x + x_multi
        x = self.norm1(x)

        ff_output = self.feedforward(x)
        # x = x + self.dropout_sublayer(ff_output) # If using sublayer dropout
        x = x + ff_output # Original: x = x + x_ff
        x = self.norm2(x)
        return x
    
class AveragePooling(nn.Module):
    def __init__(self):
        super(AveragePooling, self).__init__()
    
    def forward(self, x, event_lengths=None): # Modified to accept event_lengths for masked pooling
        if event_lengths is None:
            return torch.mean(x, dim=1)
        else:
            # Masked Average Pooling
            batch_size, seq_len, embedding_dim = x.shape
            # Create a mask of shape (batch_size, seq_len)
            mask = torch.arange(seq_len, device=x.device).expand(batch_size, seq_len) < event_lengths.unsqueeze(1)
            # Expand mask to (batch_size, seq_len, embedding_dim) to multiply with x
            mask_expanded = mask.unsqueeze(-1).expand_as(x)
            
            x_masked = x * mask_expanded # Zero out padded tokens
            
            # Sum pooled values and divide by actual lengths (event_lengths)
            # Clamp event_lengths to avoid division by zero for empty sequences (should not happen for valid events)
            summed_x = x_masked.sum(dim=1) # Shape: (batch_size, embedding_dim)
            actual_lengths = event_lengths.unsqueeze(1).float().clamp(min=1e-6) # Shape: (batch_size, 1)
            
            pooled_output = summed_x / actual_lengths
            return pooled_output
    
class MaxPooling(nn.Module): # Not used in original regression_Transformer's forward, but defined
    def __init__(self):
        super(MaxPooling, self).__init__()
    def forward(self, x):
        return torch.max(x, dim=1)[0] # Return only values, not indices
    
class Linear_regression(nn.Module):
    def __init__(
            self, 
            embedding_dim,
            output_dim,
            ):
        super(Linear_regression, self).__init__()
        self.linear = nn.Linear(embedding_dim, output_dim)
    def forward(self, x):
        return self.linear(x)
    
class regression_Transformer(nn.Module):
    def __init__(
            self,
            embedding_dim=96,
            n_layers=6,
            n_heads=6,
            input_dim=7,    # Number of features per pulse/hit from dataset
            seq_dim=256,    # Max sequence length for positional embedding table
            dropout=0.1,
            output_dim=1,   # For energy regression, this is 1
            ):
        super(regression_Transformer, self).__init__()

        self.input_embedding = nn.Linear(input_dim, embedding_dim)
        self.max_seq_len = seq_dim # Store max sequence length for positional embedding
        self.position_embedding = nn.Embedding(self.max_seq_len, embedding_dim)
        self.embedding_dropout = nn.Dropout(dropout) # Dropout after sum of embeddings + PE

        self.layers = nn.ModuleList([DecoderBlock(embedding_dim, n_heads, dropout) for _ in range(n_layers)])
        self.final_layer_norm = nn.LayerNorm(embedding_dim) # Norm before pooling
        
        # The original code's forward used a custom masked sum and division.
        # Using the AveragePooling class that handles masking is cleaner.
        self.pooling_layer = AveragePooling() 
        
        self.linear_regression = Linear_regression(embedding_dim, output_dim)

    def forward(self, x, target=None, event_lengths=None):
        # x shape: (batch_size, current_seq_len, input_dim)
        # event_lengths shape: (batch_size,) tensor of actual lengths
        
        batch_size, current_seq_len, _ = x.shape
        device = x.device

        # Truncate or error if current_seq_len > self.max_seq_len
        if current_seq_len > self.max_seq_len:
            # This case needs careful handling. For now, we'll truncate.
            # Consider if your DataLoader should pad/truncate to self.max_seq_len.
            x = x[:, :self.max_seq_len, :]
            current_seq_len = self.max_seq_len
            if event_lengths is not None:
                event_lengths = event_lengths.clamp(max=self.max_seq_len)


        input_emb = self.input_embedding(x) # (batch_size, current_seq_len, embedding_dim)
        
        # Create positions up to current_seq_len for positional embedding
        positions = torch.arange(current_seq_len, device=device).unsqueeze(0).expand(batch_size, -1) # (B, L)
        pos_emb = self.position_embedding(positions) # (B, L, D_emb)
        
        x = input_emb + pos_emb
        x = self.embedding_dropout(x) # Apply dropout

        for layer in self.layers:
            x = layer(x, event_lengths=event_lengths) # x shape: (B, L, D_emb)

        x = self.final_layer_norm(x) # Apply final layer norm

        # Use the AveragePooling class which handles event_lengths for masking
        pooled_x = self.pooling_layer(x, event_lengths=event_lengths) # (B, D_emb)
        
        y_pred = self.linear_regression(pooled_x) # (B, output_dim)

        loss = None
        if target is not None:
            # Ensure y_pred and target have compatible shapes for the loss function
            # If y_pred is (B, 1) and target is (B), squeeze y_pred for MSE_loss
            # The global loss_func (e.g., MSE_loss from src.loss) might expect this.
            y_pred_for_loss = y_pred
            if y_pred.ndim == 2 and y_pred.shape[1] == 1 and target.ndim == 1:
                y_pred_for_loss = y_pred.squeeze(-1)
            
            loss = loss_func(y_pred_for_loss, target) # Uses global loss_func
            
        return y_pred, loss

#==================================================================================================
# Define the PyTorch Lightning model      
#==================================================================================================
class LitModel(pl.LightningModule):
    def __init__(
            self, 
            model,           # Instance of regression_Transformer
            optimizer,       # Pre-configured optimizer (or tuple of optimizer/scheduler config)
            train_dataset,   # Not typically used directly if DataLoaders passed to trainer.fit
            val_dataset,     # Not typically used directly
            batch_size=16,   # For logging batch size
            ):
        super(LitModel, self).__init__()
        self.model = model
        self.optimizer = optimizer # Store the optimizer/scheduler config passed from train.py
        # self.train_dataset = train_dataset # Not used if train_dataloader() is not defined here
        # self.val_dataset = val_dataset   # Not used if val_dataloader() is not defined here
        self.batch_size = batch_size # For self.log(..., batch_size=self.batch_size)

        # Store the training and validation losses for epoch-end median calculation
        self.train_losses = []
        self.val_losses = []
        # self.train_opening_angles = [] # REMOVED - Not used for energy-only regression
        # self.val_opening_angles = []   # REMOVED - Not used for energy-only regression

    def forward(self, x, event_lengths=None):
        # This forward is for when LitModel itself is called, e.g., during inference by PL
        # It should just pass through to the core model's forward,
        # typically without the target if just for prediction.
        # The regression_Transformer.forward handles target=None.
        y_pred, _ = self.model(x, target=None, event_lengths=event_lengths)
        return y_pred

    def training_step(self, batch, batch_idx):
        # Assuming batch structure from DataLoader: (features, target_E_or_E_over_N, event_lengths)
        # Adapt this if your DataLoader provides a different structure.
        x, target, event_lengths = batch[0], batch[1], batch[2] 
        
        # self.model is your regression_Transformer.
        # Its forward method calculates loss internally using the global loss_func.
        y_pred, loss = self.model(x, target=target, event_lengths=event_lengths) 
        
        if loss is None:
            raise ValueError("Loss is None in training_step. Ensure target is passed to model and loss is computed.")

        self.train_losses.append(loss.item())

        # Log learning rate
        if self.trainer.optimizers and len(self.trainer.optimizers) > 0:
            self.log('learning_rate', self.trainer.optimizers[0].param_groups[0]['lr'], 
                     on_step=True, prog_bar=False, logger=True, sync_dist=True)

        # Simplified debug printing for energy prediction
        if batch_idx % 200 == 0: # Print less frequently
            print("\n--- Training Step Debug ---")
            print(f"Batch Index: {batch_idx}")
            # y_pred from regression_Transformer is likely (batch_size, 1)
            # target (energy or E/N) is likely (batch_size,)
            print(f"y_pred (first 5): {y_pred[:5].detach().squeeze().cpu().numpy()}") 
            print(f"target (first 5): {target[:5].detach().cpu().numpy()}")
            print(f"Loss: {loss.item()}")
            
        self.log('train_loss', loss, prog_bar=True, on_step=True, logger=True, sync_dist=True)
        return loss
    
    def on_train_epoch_end(self):
        if self.train_losses: # Check if list is not empty
            median_train_loss = torch.tensor(self.train_losses).median().item()
            self.log('median_train_loss', median_train_loss, prog_bar=True, on_epoch=True, logger=True, sync_dist=True)
        # self.log('mean_train_opening_angle', ...) # REMOVED
        # self.log('median_train_opening_angle', ...) # REMOVED
        self.train_losses = [] # Reset for next epoch
        # self.train_opening_angles = [] # REMOVED
        
    def validation_step(self, batch, batch_idx):
        x, target, event_lengths = batch[0], batch[1], batch[2] 
        y_pred, loss = self.model(x, target=target, event_lengths=event_lengths)
        
        if loss is not None:
            self.val_losses.append(loss.item())
            self.log('val_loss', loss, prog_bar=True, on_epoch=True, logger=True, sync_dist=True)
        else:
            # This case should ideally not happen if target is always provided for validation
            print(f"Warning: Loss is None in validation_step for batch_idx {batch_idx}")


        # Simplified debug printing for energy prediction
        if batch_idx % 50 == 0: # Print less frequently for validation
            print("\n--- Validation Step Debug ---")
            print(f"Batch Index: {batch_idx}")
            print(f"y_pred (first 5, validation): {y_pred[:5].detach().squeeze().cpu().numpy()}")
            print(f"target (first 5, validation): {target[:5].detach().cpu().numpy()}")
            if loss is not None:
                print(f"Loss (validation): {loss.item()}")
            
            # If predicting E/N and want to see unscaled E for debug:
            # N_doms = event_lengths # N_doms is already event_lengths
            # pred_E_unscaled_debug = y_pred[:5].detach().squeeze() * N_doms[:5].detach().float()
            # target_E_unscaled_debug = target[:5].detach() * N_doms[:5].detach().float()
            # print(f"pred_E_unscaled (debug, first 5): {pred_E_unscaled_debug.cpu().numpy()}")
            # print(f"target_E_unscaled (debug, first 5): {target_E_unscaled_debug.cpu().numpy()}")

        return loss # Return loss for PyTorch Lightning to potentially use

    def on_validation_epoch_end(self):
        if self.val_losses: # Check if list is not empty
            median_val_loss = torch.tensor(self.val_losses).median().item()
            self.log('median_val_loss', median_val_loss, prog_bar=True, on_epoch=True, logger=True, sync_dist=True) # Added sync_dist
        # self.log('mean_val_opening_angle', ...) # REMOVED
        # self.log('median_val_opening_angle', ...) # REMOVED
        self.val_losses = [] # Reset for next epoch
        # self.val_opening_angles = [] # REMOVED

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # Adapt batch unpacking based on what your dataloader provides for prediction.
        # Example: x, event_lengths = batch[0], batch[2] (if target is batch[1] but not used)
        try:
            x, _, event_lengths = batch[0], batch[1], batch[2] 
        except IndexError: # If batch doesn't have 3 elements
            x = batch[0]
            event_lengths = None # Or handle as error if event_lengths are crucial
            if len(batch) > 1 and torch.is_tensor(batch[1]) and batch[1].ndim == 1 and batch[1].numel() == x.shape[0]:
                event_lengths = batch[1] # Heuristic: if second element looks like event_lengths

        # Call the model with target=None to only get predictions
        y_pred, _ = self.model(x, target=None, event_lengths=event_lengths) 
        # The original predict_step returned a dict, which is good practice
        # return {'y_pred': y_pred, 'target': target} # But target might not be in batch here
        return y_pred # Or return y_pred and any other relevant info like event_no if available in batch

    def configure_optimizers(self):
        # This method directly returns the optimizer (and scheduler config if any)
        # that was passed to __init__ from train.py.
        # The original train.py prepares optimizer as a tuple: ([optimizer_obj], [scheduler_config_dict] or [])
        if isinstance(self.optimizer, tuple) and len(self.optimizer) == 2:
            optimizers_list, schedulers_list = self.optimizer
            if not optimizers_list: # Should not happen if train.py is correct
                raise ValueError("Optimizer list is empty in LitModel.configure_optimizers")
            
            if not schedulers_list: # No scheduler
                return optimizers_list[0] 
            else:
                # Ensure scheduler config is a dict as PL expects
                if not isinstance(schedulers_list[0], dict):
                    raise TypeError(f"Scheduler config must be a dict. Got {type(schedulers_list[0])}")
                return {'optimizer': optimizers_list[0], 'lr_scheduler': schedulers_list[0]}
        elif isinstance(self.optimizer, torch.optim.Optimizer): # If only an optimizer object was passed
            return self.optimizer
        else:
            raise TypeError(f"Unexpected type for self.optimizer: {type(self.optimizer)}. Expected Optimizer or (list_of_optimizers, list_of_schedulers_config_dicts)")