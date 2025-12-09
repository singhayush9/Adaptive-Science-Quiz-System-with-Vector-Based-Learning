import os
import re
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
from itertools import product

# ==================== CONFIGURATION ====================
csv_file = r"Preprocessed_QA_Dataset.csv"
embeddings_dir = r"NLP_LLM_Comparison"
models_dir = r"trained_models"
results_dir = r"grid_search_results"

os.makedirs(models_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==================== HYPERPARAMETER GRID ====================
param_grid = {
    'hidden_dim': [128, 256, 512],
    'num_layers': [1, 2, 3],
    'learning_rate': [0.001, 0.0005, 0.0001],
    'batch_size': [16, 32, 64],
    'sequence_length': [5, 10, 20],
    'dropout': [0.1, 0.2, 0.3]
}

# For faster testing, use a smaller grid:
# param_grid = {
#     'hidden_dim': [128, 256],
#     'num_layers': [1, 2],
#     'learning_rate': [0.001, 0.0005],
#     'batch_size': [32],
#     'sequence_length': [10],
#     'dropout': [0.2]
# }

embedding_dim = 768  # SciBERT dimension
num_epochs = 15  # Reduced for grid search

# ==================== DATASET CLASS ====================
class AnswerEmbeddingDataset(Dataset):
    """
    Dataset that creates sequences from embeddings for LSTM/GRU training.
    Each embedding is split into chunks to create sequential data.
    """
    def __init__(self, embeddings, answers, sequence_length=10):
        self.embeddings = embeddings
        self.answers = answers
        self.sequence_length = sequence_length
        self.embedding_dim = embeddings.shape[1]
        
        # Split each embedding into sequences
        self.sequences = []
        self.targets = []
        self.indices = []
        
        for idx, emb in enumerate(embeddings):
            # Reshape embedding into sequence (sequence_length, features_per_step)
            features_per_step = self.embedding_dim // sequence_length
            
            if self.embedding_dim % sequence_length != 0:
                # Pad to make divisible
                pad_size = sequence_length - (self.embedding_dim % sequence_length)
                emb = np.pad(emb, (0, pad_size), mode='constant')
                features_per_step = len(emb) // sequence_length
            
            # Create sequence
            seq = emb.reshape(sequence_length, features_per_step)
            self.sequences.append(seq)
            self.targets.append(emb[:self.embedding_dim])  # Original embedding as target
            self.indices.append(idx)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.sequences[idx]),
            torch.FloatTensor(self.targets[idx]),
            self.indices[idx]
        )

# ==================== MODEL ARCHITECTURES ====================
class LSTMEncoder(nn.Module):
    """LSTM model to encode answer embeddings"""
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.2):
        super(LSTMEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        lstm_out, (hidden, cell) = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        last_hidden = self.dropout(last_hidden)
        output = self.fc(last_hidden)
        return output

class GRUEncoder(nn.Module):
    """GRU model to encode answer embeddings"""
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.2):
        super(GRUEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.gru = nn.GRU(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        gru_out, hidden = self.gru(x)
        last_hidden = gru_out[:, -1, :]
        last_hidden = self.dropout(last_hidden)
        output = self.fc(last_hidden)
        return output

# ==================== TRAINING FUNCTIONS ====================
def train_single_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    
    for sequences, targets, _ in train_loader:
        sequences = sequences.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def validate(model, val_loader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for sequences, targets, _ in val_loader:
            sequences = sequences.to(device)
            targets = targets.to(device)
            
            outputs = model(sequences)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    
    return total_loss / len(val_loader)

def train_with_params(model, train_loader, val_loader, params, num_epochs, device):
    """Train model with specific hyperparameters"""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=2, factor=0.5
    )
    
    best_val_loss = float('inf')
    patience_counter = 0
    early_stopping_patience = 5
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        train_loss = train_single_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"    Early stopping at epoch {epoch+1}")
                break
    
    return best_val_loss, train_losses, val_losses

# ==================== GRID SEARCH ====================
class GridSearchCV:
    """Custom Grid Search for PyTorch models"""
    def __init__(self, model_class, param_grid, embedding_dim, device):
        self.model_class = model_class
        self.param_grid = param_grid
        self.embedding_dim = embedding_dim
        self.device = device
        self.results = []
        self.best_params = None
        self.best_score = float('inf')
        
    def _generate_combinations(self):
        """Generate all parameter combinations"""
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        
        for combination in product(*values):
            yield dict(zip(keys, combination))
    
    def fit(self, train_data, val_data, num_epochs):
        """Run grid search"""
        total_combinations = np.prod([len(v) for v in self.param_grid.values()])
        print(f"\nStarting Grid Search with {total_combinations} combinations...")
        print(f"Model: {self.model_class.__name__}")
        print("="*80)
        
        for i, params in enumerate(self._generate_combinations(), 1):
            print(f"\n[{i}/{total_combinations}] Testing parameters:")
            for key, value in params.items():
                print(f"  {key}: {value}")
            
            try:
                # Calculate features_per_step
                sequence_length = params['sequence_length']
                features_per_step = self.embedding_dim // sequence_length
                if self.embedding_dim % sequence_length != 0:
                    pad_size = sequence_length - (self.embedding_dim % sequence_length)
                    features_per_step = (self.embedding_dim + pad_size) // sequence_length
                
                # Create datasets with current sequence_length
                train_dataset = AnswerEmbeddingDataset(
                    train_data[0], train_data[1], params['sequence_length']
                )
                val_dataset = AnswerEmbeddingDataset(
                    val_data[0], val_data[1], params['sequence_length']
                )
                
                train_loader = DataLoader(
                    train_dataset, 
                    batch_size=params['batch_size'], 
                    shuffle=True
                )
                val_loader = DataLoader(
                    val_dataset, 
                    batch_size=params['batch_size'], 
                    shuffle=False
                )
                
                # Initialize model
                model = self.model_class(
                    input_dim=features_per_step,
                    hidden_dim=params['hidden_dim'],
                    num_layers=params['num_layers'],
                    output_dim=self.embedding_dim,
                    dropout=params['dropout']
                ).to(self.device)
                
                # Train
                val_loss, train_losses, val_losses = train_with_params(
                    model, train_loader, val_loader, params, num_epochs, self.device
                )
                
                print(f"  → Validation Loss: {val_loss:.6f}")
                
                # Store results
                result = {
                    'params': params.copy(),
                    'val_loss': val_loss,
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'features_per_step': features_per_step
                }
                self.results.append(result)
                
                # Update best
                if val_loss < self.best_score:
                    self.best_score = val_loss
                    self.best_params = params.copy()
                    self.best_params['features_per_step'] = features_per_step
                    print(f"  ★ New best validation loss: {val_loss:.6f}")
                
                # Clean up
                del model
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"  ✗ Error with parameters: {e}")
                continue
        
        print("\n" + "="*80)
        print("GRID SEARCH COMPLETE")
        print("="*80)
        print(f"\nBest Validation Loss: {self.best_score:.6f}")
        print("\nBest Parameters:")
        for key, value in self.best_params.items():
            print(f"  {key}: {value}")
        
        return self
    
    def save_results(self, filename):
        """Save grid search results"""
        results_to_save = {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'all_results': [
                {
                    'params': r['params'],
                    'val_loss': r['val_loss'],
                    'features_per_step': r['features_per_step']
                }
                for r in self.results
            ]
        }
        
        with open(filename, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        
        print(f"\n✓ Results saved to: {filename}")
    
    def plot_results(self, filename, top_n=10):
        """Plot top N configurations"""
        sorted_results = sorted(self.results, key=lambda x: x['val_loss'])[:top_n]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Validation loss comparison
        labels = [f"Config {i+1}" for i in range(len(sorted_results))]
        val_losses = [r['val_loss'] for r in sorted_results]
        
        ax1.barh(labels, val_losses)
        ax1.set_xlabel('Validation Loss')
        ax1.set_title(f'Top {top_n} Configurations by Validation Loss')
        ax1.invert_yaxis()
        
        # Plot 2: Best model training history
        best_result = sorted_results[0]
        ax2.plot(best_result['train_losses'], label='Training Loss')
        ax2.plot(best_result['val_losses'], label='Validation Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_title('Best Model Training History')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        
        print(f"✓ Plots saved to: {filename}")

# ==================== LOAD DATA ====================
print("Loading dataset and embeddings...")

# First, let's see what files are actually in the embeddings directory
print(f"\nScanning embeddings directory: {embeddings_dir}")
if os.path.exists(embeddings_dir):
    all_files = os.listdir(embeddings_dir)
    embedding_files = [f for f in all_files if f.endswith('.npy')]
    print(f"Found {len(embedding_files)} .npy files:")
    for f in embedding_files[:10]:  # Show first 10
        print(f"  - {f}")
    if len(embedding_files) > 10:
        print(f"  ... and {len(embedding_files) - 10} more")
else:
    print(f"ERROR: Directory not found: {embeddings_dir}")
    exit(1)

df = pd.read_csv(csv_file)
df.columns = df.columns.str.strip().str.lower()

print(f"\nDataset columns: {list(df.columns)}")

# Detect answer columns
a_pattern = re.compile(r"^answer(\d+)$")
answer_cols = {}
for col in df.columns:
    m = a_pattern.match(col)
    if m:
        answer_cols[int(m.group(1))] = col

print(f"Found {len(answer_cols)} answer columns: {sorted(answer_cols.keys())}")

# Combine all answer embeddings
all_embeddings = []
all_answers = []

# Load embeddings - try multiple naming patterns
for idx, a_col in sorted(answer_cols.items()):
    # Try multiple naming patterns based on what we see in your folder
    possible_names = [
        f"pair_{idx}_original_embeddings.npy",  # Pattern from your screenshot
        f"{a_col}_embeddings.npy",              # Pattern from original code
        f"answer{idx}_embeddings.npy",          # Alternative pattern
        f"pair_{idx}_generated_embeddings.npy"  # Another pattern from screenshot
    ]
    
    embedding_file = None
    for name in possible_names:
        test_path = os.path.join(embeddings_dir, name)
        if os.path.exists(test_path):
            embedding_file = test_path
            print(f"\n✓ Found: {name}")
            break
    
    if embedding_file is None:
        print(f"\n✗ Warning: No embedding file found for {a_col}")
        print(f"  Tried: {possible_names}")
        continue
    
    embeddings = np.load(embedding_file)
    print(f"  Shape: {embeddings.shape}")
    
    answers = df[a_col].fillna("").astype(str).tolist()
    
    valid_count = 0
    for i, (emb, ans) in enumerate(zip(embeddings, answers)):
        if ans.strip():
            all_embeddings.append(emb)
            all_answers.append(ans)
            valid_count += 1
    
    print(f"  Loaded {valid_count} valid embeddings from {a_col}")

all_embeddings = np.array(all_embeddings)
print(f"\nTotal embeddings loaded: {len(all_embeddings)}")
print(f"Embedding shape: {all_embeddings.shape}")

# ==================== SPLIT DATA ====================
train_emb, val_emb, train_ans, val_ans = train_test_split(
    all_embeddings, all_answers, test_size=0.2, random_state=42
)

print(f"\nTrain set: {len(train_emb)} samples")
print(f"Validation set: {len(val_emb)} samples")

train_data = (train_emb, train_ans)
val_data = (val_emb, val_ans)

# ==================== RUN GRID SEARCH ====================
# Grid Search for LSTM
print("\n" + "="*80)
print("LSTM GRID SEARCH")
print("="*80)

lstm_grid_search = GridSearchCV(
    model_class=LSTMEncoder,
    param_grid=param_grid,
    embedding_dim=embedding_dim,
    device=device
)

lstm_grid_search.fit(train_data, val_data, num_epochs)
lstm_grid_search.save_results(os.path.join(results_dir, 'lstm_grid_search_results.json'))
lstm_grid_search.plot_results(os.path.join(results_dir, 'lstm_grid_search_plots.png'))

# Grid Search for GRU
print("\n" + "="*80)
print("GRU GRID SEARCH")
print("="*80)

gru_grid_search = GridSearchCV(
    model_class=GRUEncoder,
    param_grid=param_grid,
    embedding_dim=embedding_dim,
    device=device
)

gru_grid_search.fit(train_data, val_data, num_epochs)
gru_grid_search.save_results(os.path.join(results_dir, 'gru_grid_search_results.json'))
gru_grid_search.plot_results(os.path.join(results_dir, 'gru_grid_search_plots.png'))

# ==================== TRAIN FINAL MODELS WITH BEST PARAMS ====================
print("\n" + "="*80)
print("TRAINING FINAL MODELS WITH BEST HYPERPARAMETERS")
print("="*80)

def train_final_model(model_class, best_params, train_data, val_data, model_name):
    """Train final model with best hyperparameters"""
    print(f"\nTraining final {model_name} model...")
    
    # Create datasets
    train_dataset = AnswerEmbeddingDataset(
        train_data[0], train_data[1], best_params['sequence_length']
    )
    val_dataset = AnswerEmbeddingDataset(
        val_data[0], val_data[1], best_params['sequence_length']
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=best_params['batch_size'], shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=best_params['batch_size'], shuffle=False
    )
    
    # Initialize model
    model = model_class(
        input_dim=best_params['features_per_step'],
        hidden_dim=best_params['hidden_dim'],
        num_layers=best_params['num_layers'],
        output_dim=embedding_dim,
        dropout=best_params['dropout']
    ).to(device)
    
    # Train with more epochs
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=best_params['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(25):  # More epochs for final training
        train_loss = train_single_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(models_dir, f"{model_name}_best.pth"))
        
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/25 - Train: {train_loss:.6f}, Val: {val_loss:.6f}")
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(models_dir, f"{model_name}_final.pth"))
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model_name} Final Training History')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(models_dir, f"{model_name}_final_training.png"))
    plt.close()
    
    return model, best_params

# Train LSTM with best params
lstm_final, lstm_best_params = train_final_model(
    LSTMEncoder, lstm_grid_search.best_params, train_data, val_data, "LSTM"
)

# Train GRU with best params
gru_final, gru_best_params = train_final_model(
    GRUEncoder, gru_grid_search.best_params, train_data, val_data, "GRU"
)

# ==================== SAVE CONFIGURATIONS ====================
final_config = {
    'lstm_best_params': lstm_best_params,
    'gru_best_params': gru_best_params,
    'embedding_dim': embedding_dim
}

with open(os.path.join(models_dir, 'best_model_config.json'), 'w') as f:
    json.dump(final_config, f, indent=2)

print("\n" + "="*80)
print("TRAINING COMPLETE!")
print("="*80)
print(f"\nResults saved in:")
print(f"  - {results_dir}/")
print(f"  - {models_dir}/")
print("\nBest LSTM Configuration:")
for key, value in lstm_best_params.items():
    print(f"  {key}: {value}")
print("\nBest GRU Configuration:")
for key, value in gru_best_params.items():
    print(f"  {key}: {value}")