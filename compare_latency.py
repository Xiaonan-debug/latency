# Compare inference latency between original and pruned DQN models
# Tests: single inference, batched inference, and full episode simulation

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import time
import numpy as np
import torch
import sys

# All files are in the same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ql_eye import DQN, load_agent, PumpScapeEye, device
from compress_model import (
    clone_model, apply_global_pruning, make_pruning_permanent,
    apply_dynamic_quantization, count_parameters, count_nonzero_parameters,
    format_size, get_model_size_bytes, save_sparse_model, get_file_size_bytes
)

# Output - use this folder
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, "results")
os.makedirs(output_dir, exist_ok=True)


def measure_single_inference(model, input_tensor, num_runs=1000, warmup=100):
    """
    Measure single-sample inference latency.
    
    Args:
        model: The model to benchmark
        input_tensor: A single input tensor (1, state_size)
        num_runs: Number of inference runs to average
        warmup: Number of warmup runs (not counted)
    
    Returns:
        Dictionary with latency statistics (in microseconds)
    """
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_tensor)
    
    # Measure
    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = model(input_tensor)
            end = time.perf_counter()
            latencies.append((end - start) * 1e6)  # Convert to microseconds
    
    latencies = np.array(latencies)
    return {
        'mean_us': np.mean(latencies),
        'median_us': np.median(latencies),
        'std_us': np.std(latencies),
        'min_us': np.min(latencies),
        'max_us': np.max(latencies),
        'p95_us': np.percentile(latencies, 95),
        'p99_us': np.percentile(latencies, 99),
    }


def measure_batch_inference(model, state_size, batch_sizes=[1, 8, 32, 64, 128], 
                            num_runs=500, warmup=50):
    """
    Measure inference latency across different batch sizes.
    """
    model.eval()
    results = {}
    
    for bs in batch_sizes:
        input_tensor = torch.randn(bs, state_size).to(next(model.parameters()).device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(input_tensor)
        
        # Measure
        latencies = []
        with torch.no_grad():
            for _ in range(num_runs):
                start = time.perf_counter()
                _ = model(input_tensor)
                end = time.perf_counter()
                latencies.append((end - start) * 1e6)
        
        results[bs] = {
            'mean_us': np.mean(latencies),
            'median_us': np.median(latencies),
            'std_us': np.std(latencies),
        }
    
    return results


def measure_episode_latency(model, env, num_episodes=10):
    """
    Measure total decision time over full 24-hour episodes.
    """
    model.eval()
    eval_device = next(model.parameters()).device
    
    episode_times = []
    episode_steps = []
    per_step_times = []
    
    for ep in range(num_episodes):
        state = env.reset()
        done = False
        step_count = 0
        ep_decision_time = 0
        
        while not done and step_count < 24:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(eval_device)
            
            with torch.no_grad():
                start = time.perf_counter()
                q_values = model(state_tensor)
                action = q_values.argmax().item()
                end = time.perf_counter()
            
            decision_us = (end - start) * 1e6
            ep_decision_time += decision_us
            per_step_times.append(decision_us)
            
            state, reward, done, info = env.step(action)
            step_count += 1
        
        episode_times.append(ep_decision_time)
        episode_steps.append(step_count)
    
    return {
        'avg_episode_time_us': np.mean(episode_times),
        'avg_episode_time_ms': np.mean(episode_times) / 1000,
        'avg_steps': np.mean(episode_steps),
        'avg_per_step_us': np.mean(per_step_times),
        'median_per_step_us': np.median(per_step_times),
        'total_decisions': len(per_step_times),
    }


def run_latency_comparison(agent_path, pruning_amount=0.7):
    """
    Run full latency comparison between original and pruned models.
    
    Args:
        agent_path: Path to the original trained agent
        pruning_amount: The pruning ratio for the pruned model (default 0.7 = 70%)
    """
    print("\n" + "=" * 70)
    print("INFERENCE LATENCY COMPARISON: ORIGINAL vs PRUNED MODEL")
    print("=" * 70)
    
    # ========================================================================
    # Load models
    # ========================================================================
    print("\nLoading original agent...")
    agent = load_agent(agent_path)
    if agent is None:
        print(f"Failed to load agent from {agent_path}")
        return
    
    original_model = agent.policy_net
    original_model.eval()
    
    print(f"Creating {pruning_amount*100:.0f}% pruned model...")
    pruned_model = apply_global_pruning(original_model, amount=pruning_amount)
    pruned_model = make_pruning_permanent(pruned_model)
    pruned_model.eval()
    
    # Model info
    state_size = original_model.fc1.in_features
    orig_total = count_parameters(original_model)
    orig_nonzero, _ = count_nonzero_parameters(original_model)
    pruned_total = count_parameters(pruned_model)
    pruned_nonzero, _ = count_nonzero_parameters(pruned_model)
    
    print(f"\n{'Model':<25} {'Total Params':>15} {'Non-zero':>15} {'Sparsity':>10}")
    print("-" * 65)
    print(f"{'Original':<25} {orig_total:>15,} {orig_nonzero:>15,} {'0.0%':>10}")
    sparsity = (1 - pruned_nonzero / pruned_total) * 100
    print(f"{'Pruned (' + str(int(pruning_amount*100)) + '%)':<25} {pruned_total:>15,} {pruned_nonzero:>15,} {sparsity:>9.1f}%")
    
    # ========================================================================
    # Test 1: Single inference latency
    # ========================================================================
    print("\n" + "-" * 70)
    print("TEST 1: SINGLE INFERENCE LATENCY (1000 runs)")
    print("-" * 70)
    
    eval_device = next(original_model.parameters()).device
    dummy_input = torch.randn(1, state_size).to(eval_device)
    
    print("\nBenchmarking original model...")
    orig_single = measure_single_inference(original_model, dummy_input, num_runs=1000)
    
    print("Benchmarking pruned model...")
    pruned_single = measure_single_inference(pruned_model, dummy_input, num_runs=1000)
    
    print(f"\n{'Metric':<20} {'Original (μs)':>18} {'Pruned (μs)':>18} {'Speedup':>10}")
    print("-" * 66)
    for metric in ['mean_us', 'median_us', 'p95_us', 'p99_us', 'min_us', 'max_us']:
        label = metric.replace('_us', '').replace('_', ' ').title()
        orig_val = orig_single[metric]
        pruned_val = pruned_single[metric]
        speedup = orig_val / pruned_val if pruned_val > 0 else float('inf')
        print(f"{label:<20} {orig_val:>18.1f} {pruned_val:>18.1f} {speedup:>9.2f}x")
    
    # ========================================================================
    # Test 2: Batch inference latency
    # ========================================================================
    print("\n" + "-" * 70)
    print("TEST 2: BATCH INFERENCE LATENCY (500 runs per batch size)")
    print("-" * 70)
    
    batch_sizes = [1, 8, 32, 64, 128]
    
    print("\nBenchmarking original model...")
    orig_batch = measure_batch_inference(original_model, state_size, batch_sizes)
    
    print("Benchmarking pruned model...")
    pruned_batch = measure_batch_inference(pruned_model, state_size, batch_sizes)
    
    print(f"\n{'Batch Size':<12} {'Original Mean (μs)':>20} {'Pruned Mean (μs)':>20} {'Speedup':>10}")
    print("-" * 62)
    for bs in batch_sizes:
        orig_val = orig_batch[bs]['mean_us']
        pruned_val = pruned_batch[bs]['mean_us']
        speedup = orig_val / pruned_val if pruned_val > 0 else float('inf')
        print(f"{bs:<12} {orig_val:>20.1f} {pruned_val:>20.1f} {speedup:>9.2f}x")
    
    # ========================================================================
    # Test 3: Full episode simulation latency
    # ========================================================================
    print("\n" + "-" * 70)
    print("TEST 3: FULL EPISODE SIMULATION (10 episodes each)")
    print("-" * 70)
    
    env = PumpScapeEye()
    
    print("\nBenchmarking original model (full episodes)...")
    orig_episode = measure_episode_latency(original_model, env, num_episodes=10)
    
    print("Benchmarking pruned model (full episodes)...")
    pruned_episode = measure_episode_latency(pruned_model, env, num_episodes=10)
    
    print(f"\n{'Metric':<35} {'Original':>18} {'Pruned':>18} {'Speedup':>10}")
    print("-" * 81)
    
    metrics = [
        ('Avg per-step decision (μs)', 'avg_per_step_us', 'μs'),
        ('Median per-step decision (μs)', 'median_per_step_us', 'μs'),
        ('Avg episode total (ms)', 'avg_episode_time_ms', 'ms'),
        ('Avg steps per episode', 'avg_steps', ''),
        ('Total decisions made', 'total_decisions', ''),
    ]
    
    for label, key, unit in metrics:
        orig_val = orig_episode[key]
        pruned_val = pruned_episode[key]
        if key in ['avg_per_step_us', 'median_per_step_us', 'avg_episode_time_ms']:
            speedup = orig_val / pruned_val if pruned_val > 0 else float('inf')
            print(f"{label:<35} {orig_val:>17.1f} {pruned_val:>17.1f} {speedup:>9.2f}x")
        else:
            print(f"{label:<35} {orig_val:>17.0f} {pruned_val:>17.0f} {'':>10}")
    
    # ========================================================================
    # Test 4: Quantized model latency (CPU only)
    # ========================================================================
    print("\n" + "-" * 70)
    print("TEST 4: QUANTIZED MODEL LATENCY (INT8, CPU)")
    print("-" * 70)
    
    try:
        print("\nCreating quantized model...")
        quantized_model = apply_dynamic_quantization(original_model)
        
        cpu_input = torch.randn(1, state_size)  # CPU tensor for quantized model
        
        # Also move original to CPU for fair comparison
        original_cpu = clone_model(original_model).cpu()
        original_cpu.eval()
        cpu_input_orig = torch.randn(1, state_size)
        
        print("Benchmarking original model (CPU)...")
        orig_cpu_latency = measure_single_inference(original_cpu, cpu_input_orig, num_runs=1000)
        
        print("Benchmarking quantized model (CPU)...")
        quant_latency = measure_single_inference(quantized_model, cpu_input, num_runs=1000)
        
        print(f"\n{'Metric':<20} {'Original CPU (μs)':>20} {'Quantized (μs)':>20} {'Speedup':>10}")
        print("-" * 70)
        for metric in ['mean_us', 'median_us', 'p95_us']:
            label = metric.replace('_us', '').replace('_', ' ').title()
            orig_val = orig_cpu_latency[metric]
            quant_val = quant_latency[metric]
            speedup = orig_val / quant_val if quant_val > 0 else float('inf')
            print(f"{label:<20} {orig_val:>20.1f} {quant_val:>20.1f} {speedup:>9.2f}x")
    
    except Exception as e:
        print(f"  Quantization benchmark failed: {e}")
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n  Original model:  {orig_total:,} parameters, {format_size(get_model_size_bytes(original_model))}")
    print(f"  Pruned model:    {pruned_nonzero:,} non-zero params ({sparsity:.1f}% sparse)")
    print(f"\n  Single inference speedup:  {orig_single['mean_us'] / pruned_single['mean_us']:.2f}x")
    print(f"  Episode decision speedup:  {orig_episode['avg_per_step_us'] / pruned_episode['avg_per_step_us']:.2f}x")
    print(f"\n  Original avg decision time:  {orig_single['mean_us']:.1f} μs")
    print(f"  Pruned avg decision time:    {pruned_single['mean_us']:.1f} μs")
    print(f"  Time saved per decision:     {orig_single['mean_us'] - pruned_single['mean_us']:.1f} μs")
    print("=" * 70)


if __name__ == "__main__":
    # Model is in the same folder
    agent_path = os.path.join(script_dir, 'best_dqn_eye_agent.pth')
    
    if not os.path.exists(agent_path):
        print(f"Agent not found at {agent_path}")
        sys.exit(1)
    
    run_latency_comparison(agent_path, pruning_amount=0.7)
