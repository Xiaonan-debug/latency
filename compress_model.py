# Model Compression and Pruning for DQN Agent
# Compresses the trained model and evaluates performance
# Adapted for the old ql_eye.py agent (Eye_Results/best_dqn_eye_agent.pth)

# Fix OpenMP conflict between PyTorch and scipy on macOS
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import sys
import copy
import gzip
import pickle
from scipy import sparse

# All files are in the same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import from the old ql_eye.py system
from ql_eye import (
    DQN, DQNAgent, PumpScapeEye,
    load_agent, save_agent, device,
    criticalDepletion, depletion, excess, criticalExcess
)

# Import visualization from new system
from dqn_new_system import plot_rl_only_states

# Output directory - use this folder
script_dir = os.path.dirname(os.path.abspath(__file__))
eye_output_dir = script_dir
compress_output_dir = os.path.join(script_dir, "Compressed_Models")
os.makedirs(compress_output_dir, exist_ok=True)

# ============================================================================
# MODEL SIZE UTILITIES
# ============================================================================

def count_parameters(model):
    """Count total number of parameters in the model."""
    return sum(p.numel() for p in model.parameters())

def count_nonzero_parameters(model):
    """Count number of non-zero parameters (after pruning)."""
    total = 0
    nonzero = 0
    for name, param in model.named_parameters():
        total += param.numel()
        nonzero += torch.count_nonzero(param).item()
    return nonzero, total

def get_model_size_bytes(model):
    """Get model size in bytes (parameter memory)."""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    return param_size + buffer_size

def get_file_size_bytes(filepath):
    """Get actual file size on disk."""
    if os.path.exists(filepath):
        return os.path.getsize(filepath)
    return 0

def format_size(size_bytes):
    """Format size in human-readable format."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.2f} MB"

def save_sparse_model(model, filepath):
    """
    Save model using sparse matrix format for pruned weights.
    Only stores non-zero values - dramatically reduces size for sparse models.
    
    This is key: PyTorch tensors store ALL values including zeros.
    Sparse format only stores non-zero values + their indices.
    """
    state_dict = model.state_dict()
    sparse_dict = {}
    
    for key, tensor in state_dict.items():
        arr = tensor.cpu().numpy()
        # Convert 2D weight matrices to sparse CSR format
        if arr.ndim == 2:
            sparse_matrix = sparse.csr_matrix(arr)
            sparse_dict[key] = {
                'format': 'csr',
                'data': sparse_matrix.data.astype(np.float32),
                'indices': sparse_matrix.indices,
                'indptr': sparse_matrix.indptr,
                'shape': sparse_matrix.shape
            }
        else:
            # Keep 1D arrays (biases, batch norm params) as dense
            sparse_dict[key] = {
                'format': 'dense', 
                'data': arr.astype(np.float32)
            }
    
    # Save with gzip compression (zeros in indices also compress well)
    with gzip.open(filepath, 'wb', compresslevel=9) as f:
        pickle.dump(sparse_dict, f)
    
    return get_file_size_bytes(filepath)

def load_sparse_model(filepath, model_template):
    """
    Load a model saved in sparse format.
    
    Args:
        filepath: Path to the sparse model file
        model_template: A model instance with the correct architecture
    
    Returns:
        Model with loaded weights
    """
    with gzip.open(filepath, 'rb') as f:
        sparse_dict = pickle.load(f)
    
    state_dict = {}
    for key, data in sparse_dict.items():
        if data['format'] == 'csr':
            # Reconstruct dense matrix from sparse
            sparse_matrix = sparse.csr_matrix(
                (data['data'], data['indices'], data['indptr']),
                shape=data['shape']
            )
            state_dict[key] = torch.from_numpy(sparse_matrix.toarray())
        else:
            state_dict[key] = torch.from_numpy(data['data'])
    
    model_template.load_state_dict(state_dict)
    return model_template

def print_model_info(model, name="Model"):
    """Print comprehensive model information."""
    total_params = count_parameters(model)
    nonzero_params, _ = count_nonzero_parameters(model)
    model_size = get_model_size_bytes(model)
    sparsity = 1 - (nonzero_params / total_params) if total_params > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"{name} Information")
    print(f"{'='*60}")
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Non-zero parameters:  {nonzero_params:,}")
    print(f"  Sparsity:             {sparsity*100:.2f}%")
    print(f"  Model memory size:    {format_size(model_size)}")
    print(f"{'='*60}\n")
    
    return {
        'total_params': total_params,
        'nonzero_params': nonzero_params,
        'sparsity': sparsity,
        'model_size_bytes': model_size
    }

# ============================================================================
# PRUNING METHODS
# ============================================================================

def clone_model(model):
    """
    Clone a model by creating a new instance and loading state dict.
    Works better than deepcopy for models with pruning masks.
    Adapted for the old ql_eye DQN architecture (4 layers, fc4 is last).
    """
    # Get the model's state dict
    state_dict = model.state_dict()
    
    # Create a new model with the same architecture
    # Old DQN: fc1 -> fc2 -> fc3 -> fc4 (last layer)
    new_model = DQN(
        state_size=model.fc1.in_features,
        action_size=model.fc4.out_features,
        hidden_size=model.fc1.out_features
    ).to(next(model.parameters()).device)
    
    # Load the state dict
    new_model.load_state_dict(state_dict)
    
    return new_model

def apply_magnitude_pruning(model, amount=0.3):
    """
    Apply magnitude-based pruning to linear layers.
    Removes weights with smallest absolute values.
    """
    model_copy = clone_model(model)
    
    for name, module in model_copy.named_modules():
        if isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=amount)
            if module.bias is not None:
                prune.l1_unstructured(module, name='bias', amount=amount)
    
    return model_copy

def apply_structured_pruning(model, amount=0.2):
    """
    Apply structured pruning (removes entire neurons/channels).
    More hardware-friendly than unstructured pruning.
    """
    model_copy = clone_model(model)
    
    for name, module in model_copy.named_modules():
        if isinstance(module, nn.Linear):
            prune.ln_structured(module, name='weight', amount=amount, n=2, dim=0)
    
    return model_copy

def make_pruning_permanent(model):
    """
    Make pruning permanent by removing the pruning reparametrization.
    This actually removes the pruned weights from the model.
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            try:
                prune.remove(module, 'weight')
            except ValueError:
                pass
            try:
                prune.remove(module, 'bias')
            except ValueError:
                pass
    
    return model

def apply_global_pruning(model, amount=0.3):
    """
    Apply global pruning - prunes across all layers based on global importance.
    This tends to preserve more important weights regardless of layer.
    """
    model_copy = clone_model(model)
    
    parameters_to_prune = []
    for name, module in model_copy.named_modules():
        if isinstance(module, nn.Linear):
            parameters_to_prune.append((module, 'weight'))
    
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount
    )
    
    return model_copy

# ============================================================================
# QUANTIZATION METHODS
# ============================================================================

def apply_dynamic_quantization(model):
    """
    Apply dynamic quantization (int8) to linear layers.
    Reduces model size and can speed up inference on CPU.
    """
    model_copy = clone_model(model)
    model_copy.eval()
    model_copy = model_copy.cpu()
    
    quantized_model = torch.quantization.quantize_dynamic(
        model_copy,
        {nn.Linear},
        dtype=torch.qint8
    )
    
    return quantized_model

def apply_half_precision(model):
    """
    Convert model to half precision (float16).
    Reduces memory by 50%.
    """
    model_copy = clone_model(model)
    return model_copy.half()

# ============================================================================
# THRESHOLDS (from ql_eye.py for visualization)
# ============================================================================

def get_eye_thresholds(param_idx):
    """Get threshold lines for plotting based on old ql_eye thresholds."""
    if param_idx < len(criticalDepletion):
        return [
            (criticalDepletion[param_idx], 'red', 'Critical Low'),
            (depletion[param_idx], 'orange', 'Low'),
            (excess[param_idx], 'orange', 'High'),
            (criticalExcess[param_idx], 'red', 'Critical High')
        ]
    return []

# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_compressed_model(model, env, num_episodes=1, use_quantized=False):
    """
    Evaluate a compressed model using the old PumpScapeEye environment.
    """
    model.eval()
    
    if use_quantized:
        eval_device = torch.device('cpu')
    else:
        eval_device = device
        model = model.to(eval_device)
    
    rewards = []
    steps = []
    all_states = []
    all_actions = []
    success_count = 0
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_states = [env.bigState.copy()]
        episode_actions = []
        total_reward = 0
        step_count = 0
        done = False
        
        while not done and step_count < 24:
            with torch.no_grad():
                if use_quantized:
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                else:
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(eval_device)
                
                q_values = model(state_tensor)
                action = q_values.argmax().item()
            
            next_state, reward, done, info = env.step(action)
            
            episode_states.append(env.bigState.copy())
            episode_actions.append(env.decode_action(action))
            
            state = next_state
            total_reward += reward
            step_count += 1
            
            # Check hours survived from bigState directly
            hours_survived = env.bigState[16]
            if hours_survived >= 24:
                success_count += 1
        
        rewards.append(total_reward)
        steps.append(step_count)
        
        if episode == 0:
            all_states = episode_states
            all_actions = episode_actions
    
    hours_survived = env.bigState[16] if env.bigState else 0
    
    results = {
        'avg_reward': np.mean(rewards),
        'avg_steps': np.mean(steps),
        'success_rate': success_count / num_episodes * 100,
        'hours_survived': hours_survived,
        'states': all_states,
        'actions': all_actions
    }
    
    return results

# ============================================================================
# VISUALIZATION (adapted for old ql_eye bigState layout)
# ============================================================================

def plot_eye_rl_only_states(states, save_dir):
    """
    Plot state evolution for the old ql_eye system.
    Old bigState: [0]temp, [3]PFI, [4]pH, [6]pvO2, [9]glucose, [10]insulin
    """
    import matplotlib.pyplot as plt
    
    states_arr = np.array(states)
    
    param_names = ["Temperature", "PFI", "pH", "pvO2", "Glucose", "Insulin"]
    param_indices = [0, 3, 4, 6, 9, 10]
    
    plt.style.use('seaborn-v0_8-darkgrid')
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()
    
    agent_color = '#1E88E5'
    critical_threshold_color = '#D32F2F'
    warning_threshold_color = '#F39C12'
    safe_zone_color = '#E8F8F5'
    warning_zone_color = '#FEF9E7'
    danger_zone_color = '#FDEDEC'
    
    hours = range(len(states))
    
    for i, (param_name, param_idx) in enumerate(zip(param_names, param_indices)):
        if i < len(axes):
            ax = axes[i]
            
            thresholds = get_eye_thresholds(param_idx)
            if thresholds:
                critical_low = thresholds[0][0]
                depl = thresholds[1][0]
                exc = thresholds[2][0]
                critical_high = thresholds[3][0]
                
                data_min = states_arr[:, param_idx].min()
                data_max = states_arr[:, param_idx].max()
                y_min = min(critical_low * 0.85, data_min * 0.9)
                y_max = max(critical_high * 1.15, data_max * 1.1)
                
                ax.axhspan(y_min, critical_low, alpha=0.15, color=danger_zone_color, zorder=0)
                ax.axhspan(critical_high, y_max, alpha=0.15, color=danger_zone_color, zorder=0)
                ax.axhspan(critical_low, depl, alpha=0.12, color=warning_zone_color, zorder=0)
                ax.axhspan(exc, critical_high, alpha=0.12, color=warning_zone_color, zorder=0)
                ax.axhspan(depl, exc, alpha=0.15, color=safe_zone_color, zorder=0)
                
                ax.axhline(y=critical_low, color=critical_threshold_color, linestyle='--', linewidth=2.5, alpha=0.8, zorder=1, label='Critical')
                ax.axhline(y=critical_high, color=critical_threshold_color, linestyle='--', linewidth=2.5, alpha=0.8, zorder=1)
                ax.axhline(y=depl, color=warning_threshold_color, linestyle=':', linewidth=2, alpha=0.7, zorder=1, label='Warning')
                ax.axhline(y=exc, color=warning_threshold_color, linestyle=':', linewidth=2, alpha=0.7, zorder=1)
                
                ax.set_ylim(y_min, y_max)
            
            ax.plot(hours, states_arr[:, param_idx], color='black', linewidth=5, alpha=0.15, zorder=2)
            ax.plot(hours, states_arr[:, param_idx], color=agent_color, linewidth=3.5, 
                   marker='o', markersize=8, markerfacecolor='white', markeredgewidth=2.5,
                   markeredgecolor=agent_color, label='RL Controller', zorder=3, markevery=1)
            
            ax.set_title(f'{param_name}', fontsize=16, fontweight='bold', pad=12)
            ax.set_xlabel('Time (hours)', fontsize=13, fontweight='bold')
            
            ylabel_map = {
                'Temperature': 'Temperature (°C)',
                'PFI': 'Perfusion Flow Index',
                'pH': 'pH',
                'pvO2': 'pvO₂ (mmHg)',
                'Glucose': 'Glucose (mM)',
                'Insulin': 'Insulin (mU)'
            }
            ax.set_ylabel(ylabel_map.get(param_name, 'Value'), fontsize=13, fontweight='bold')
            
            ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.8, color='gray')
            ax.set_axisbelow(True)
            ax.tick_params(labelsize=11, width=1.5, length=6)
            
            if i == 0:
                ax.legend(fontsize=10, loc='upper right', framealpha=0.95, 
                         edgecolor='#555555', fancybox=True, shadow=True)
            
            for spine in ax.spines.values():
                spine.set_edgecolor('#888888')
                spine.set_linewidth(1.5)
            ax.set_facecolor('#fafafa')
    
    fig.suptitle('EYE Scenario - RL Controller State Evolution (Pruned)', 
                fontsize=20, fontweight='bold', y=0.995, color='#2C3E50')
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(os.path.join(save_dir, 'EYE_Pruned_rl_only_states.png'), 
               dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.savefig(os.path.join(save_dir, 'EYE_Pruned_rl_only_states.pdf'), 
               bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.show()
    
    plt.style.use('default')

# ============================================================================
# MAIN COMPRESSION PIPELINE
# ============================================================================

def run_compression_analysis(agent_path, pruning_amounts=[0.1, 0.2, 0.3, 0.5]):
    """
    Run comprehensive compression analysis on the old ql_eye trained model.
    
    Args:
        agent_path: Path to the trained agent (Eye_Results/best_dqn_eye_agent.pth)
        pruning_amounts: List of pruning ratios to test
    """
    print("\n" + "="*70)
    print("MODEL COMPRESSION AND PRUNING ANALYSIS")
    print("="*70)
    
    # Load the original agent using ql_eye's load_agent
    agent = load_agent(agent_path)
    if agent is None:
        print(f"Failed to load agent from {agent_path}")
        return
    
    original_model = agent.policy_net
    
    # Create old environment for evaluation
    env = PumpScapeEye()
    
    # Get original file size
    original_file_size = get_file_size_bytes(agent_path)
    
    # Print original model info
    print("\n" + "-"*70)
    print("ORIGINAL MODEL")
    print("-"*70)
    original_info = print_model_info(original_model, "Original Model")
    print(f"  File size on disk:    {format_size(original_file_size)}")
    
    # Save original in sparse format for fair baseline comparison
    original_sparse_path = os.path.join(compress_output_dir, 'EYE_original_sparse.gz')
    original_sparse_size = save_sparse_model(original_model, original_sparse_path)
    print(f"  Sparse format size:   {format_size(original_sparse_size)}")
    
    # Evaluate original model
    print("\nEvaluating original model...")
    original_results = evaluate_compressed_model(original_model, env, num_episodes=1)
    print(f"  Hours survived: {original_results['hours_survived']}")
    print(f"  Total reward:   {original_results['avg_reward']:.2f}")
    
    # Store results for comparison
    compression_results = [{
        'method': 'Original',
        'amount': 0,
        'total_params': original_info['total_params'],
        'nonzero_params': original_info['nonzero_params'],
        'sparsity': 0,
        'model_size': original_sparse_size,
        'hours_survived': original_results['hours_survived'],
        'reward': original_results['avg_reward'],
        'compression_ratio': 1.0
    }]
    
    # ========================================================================
    # Test different pruning amounts with Global Pruning
    # ========================================================================
    print("\n" + "-"*70)
    print("GLOBAL MAGNITUDE PRUNING")
    print("-"*70)
    
    best_pruned_model = None
    best_pruned_states = None
    best_pruning_amount = 0
    
    for amount in pruning_amounts:
        print(f"\n>>> Testing {amount*100:.0f}% Global Pruning...")
        
        pruned_model = apply_global_pruning(original_model, amount=amount)
        pruned_model = make_pruning_permanent(pruned_model)
        
        info = print_model_info(pruned_model, f"Pruned Model ({amount*100:.0f}%)")
        
        # Save in SPARSE format to get ACTUAL file size reduction
        sparse_path = os.path.join(compress_output_dir, f'EYE_pruned_{int(amount*100)}_sparse.gz')
        actual_file_size = save_sparse_model(pruned_model, sparse_path)
        print(f"  Sparse file size:     {format_size(actual_file_size)}")
        print(f"  Size reduction:       {(1 - actual_file_size/original_sparse_size)*100:.1f}%")
        
        # Evaluate
        results = evaluate_compressed_model(pruned_model, env, num_episodes=1)
        print(f"  Hours survived: {results['hours_survived']}")
        print(f"  Total reward:   {results['avg_reward']:.2f}")
        
        compression_ratio = original_sparse_size / actual_file_size if actual_file_size > 0 else float('inf')
        
        compression_results.append({
            'method': 'Global Pruning',
            'amount': amount,
            'total_params': info['total_params'],
            'nonzero_params': info['nonzero_params'],
            'sparsity': info['sparsity'],
            'model_size': actual_file_size,
            'hours_survived': results['hours_survived'],
            'reward': results['avg_reward'],
            'compression_ratio': compression_ratio
        })
        
        # Track best pruned model (highest pruning with 24h survival)
        if results['hours_survived'] >= 24 and (best_pruned_model is None or amount > best_pruning_amount):
            best_pruned_model = pruned_model
            best_pruned_states = results['states']
            best_pruning_amount = amount
    
    # ========================================================================
    # Test Dynamic Quantization
    # ========================================================================
    print("\n" + "-"*70)
    print("DYNAMIC QUANTIZATION (INT8)")
    print("-"*70)
    
    try:
        quantized_model = apply_dynamic_quantization(original_model)
        
        print("\n>>> Testing Dynamic Quantization...")
        
        quant_path = os.path.join(compress_output_dir, 'EYE_quantized_model.pth')
        torch.save(quantized_model.state_dict(), quant_path)
        quant_file_size = get_file_size_bytes(quant_path)
        
        print(f"  Quantized file size: {format_size(quant_file_size)}")
        print(f"  Compression ratio:   {original_file_size / quant_file_size:.2f}x")
        
        results = evaluate_compressed_model(quantized_model, env, num_episodes=1, use_quantized=True)
        print(f"  Hours survived: {results['hours_survived']}")
        print(f"  Total reward:   {results['avg_reward']:.2f}")
        
        compression_results.append({
            'method': 'Dynamic Quantization',
            'amount': 0,
            'total_params': count_parameters(original_model),
            'nonzero_params': count_parameters(original_model),
            'sparsity': 0,
            'model_size': quant_file_size,
            'hours_survived': results['hours_survived'],
            'reward': results['avg_reward'],
            'compression_ratio': original_file_size / quant_file_size
        })
        
    except Exception as e:
        print(f"  Quantization failed: {e}")
    
    # ========================================================================
    # Test Combined: Pruning + Quantization
    # ========================================================================
    print("\n" + "-"*70)
    print("COMBINED: PRUNING (30%) + QUANTIZATION")
    print("-"*70)
    
    try:
        pruned_30 = apply_global_pruning(original_model, amount=0.3)
        pruned_30 = make_pruning_permanent(pruned_30)
        
        combined_model = apply_dynamic_quantization(pruned_30)
        
        combined_path = os.path.join(compress_output_dir, 'EYE_pruned_quantized_model.pth')
        torch.save(combined_model.state_dict(), combined_path)
        combined_file_size = get_file_size_bytes(combined_path)
        
        print(f"  Combined file size:  {format_size(combined_file_size)}")
        print(f"  Compression ratio:   {original_file_size / combined_file_size:.2f}x")
        
        results = evaluate_compressed_model(combined_model, env, num_episodes=1, use_quantized=True)
        print(f"  Hours survived: {results['hours_survived']}")
        print(f"  Total reward:   {results['avg_reward']:.2f}")
        
        compression_results.append({
            'method': 'Pruning + Quantization',
            'amount': 0.3,
            'total_params': count_parameters(pruned_30),
            'nonzero_params': count_nonzero_parameters(pruned_30)[0],
            'sparsity': 0.3,
            'model_size': combined_file_size,
            'hours_survived': results['hours_survived'],
            'reward': results['avg_reward'],
            'compression_ratio': original_file_size / combined_file_size
        })
        
    except Exception as e:
        print(f"  Combined compression failed: {e}")
    
    # ========================================================================
    # Summary Report
    # ========================================================================
    print("\n" + "="*70)
    print("COMPRESSION SUMMARY REPORT")
    print("="*70)
    print(f"\n{'Method':<30} {'Sparsity':>10} {'Size':>12} {'Compress':>10} {'Hours':>8} {'Reward':>10}")
    print("-"*80)
    
    for r in compression_results:
        method = r['method']
        if r['amount'] > 0 and 'Pruning' in method:
            method = f"{method} ({r['amount']*100:.0f}%)"
        
        sparsity_str = f"{r['sparsity']*100:.1f}%" if r['sparsity'] > 0 else "-"
        size_str = format_size(r['model_size'])
        compress_str = f"{r['compression_ratio']:.2f}x"
        
        print(f"{method:<30} {sparsity_str:>10} {size_str:>12} {compress_str:>10} {r['hours_survived']:>8} {r['reward']:>10.2f}")
    
    print("-"*80)
    
    # ========================================================================
    # Save best compressed model
    # ========================================================================
    if best_pruned_model is not None:
        print(f"\n>>> Saving best pruned model ({best_pruning_amount*100:.0f}% pruning)...")
        
        sparse_save_path = os.path.join(compress_output_dir, 'best_pruned_agent_EYE_sparse.gz')
        sparse_file_size = save_sparse_model(best_pruned_model, sparse_save_path)
        
        save_path = os.path.join(compress_output_dir, 'best_pruned_agent_EYE.pth')
        torch.save(best_pruned_model.state_dict(), save_path)
        regular_file_size = get_file_size_bytes(save_path)
        
        print(f"\n  Original sparse size:    {format_size(original_sparse_size)}")
        print(f"  Pruned sparse size:      {format_size(sparse_file_size)}")
        print(f"  Actual size reduction:   {(1 - sparse_file_size/original_sparse_size)*100:.1f}%")
        print(f"  Compression ratio:       {original_sparse_size/sparse_file_size:.2f}x")
        print(f"\n  (Regular PyTorch save:   {format_size(regular_file_size)} - no reduction)")
        
        # Generate visualization for best pruned model
        if best_pruned_states:
            plot_eye_rl_only_states(best_pruned_states, compress_output_dir)
    
    return compression_results


if __name__ == "__main__":
    print("Running compression analysis for EYE scenario (old ql_eye agent)...")
    
    # Path to the old trained agent
    agent_path = os.path.join(eye_output_dir, 'best_dqn_eye_agent.pth')
    
    if not os.path.exists(agent_path):
        print(f"No trained agent found at {agent_path}!")
        print("Please ensure Eye_Results/best_dqn_eye_agent.pth exists.")
        sys.exit(1)
    
    print(f"Using agent: {agent_path}")
    
    # Run compression analysis
    results = run_compression_analysis(
        agent_path=agent_path,
        pruning_amounts=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    )
    
    print(f"\n{'='*70}")
    print("Analysis complete!")
    print(f"Results saved to: {compress_output_dir}")
    print(f"{'='*70}")
