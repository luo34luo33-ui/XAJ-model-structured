"""
XAJ Model Utility Module
Provides common utility functions for logging, file I/O, and result comparison.
"""

import json
import logging
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path


def setup_logger(name: str, log_file: str = None, level=logging.INFO) -> logging.Logger:
    """Set up a logger with console and optional file output.
    
    Args:
        name: Logger name
        log_file: Optional log file path
        level: Logging level
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def load_json(file_path: str) -> Dict:
    """Load JSON file.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Dictionary containing JSON data
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Dict, file_path: str, indent: int = 4):
    """Save data to JSON file.
    
    Args:
        data: Data to save
        file_path: Output file path
        indent: JSON indentation level
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def calculate_metrics(observed: np.ndarray, simulated: np.ndarray) -> Dict[str, float]:
    """Calculate model performance metrics.
    
    Args:
        observed: Observed values
        simulated: Simulated values
        
    Returns:
        Dictionary containing performance metrics
    """
    # Remove NaN values
    mask = ~(np.isnan(observed) | np.isnan(simulated))
    obs = observed[mask]
    sim = simulated[mask]
    
    if len(obs) == 0:
        return {
            'MAE': np.nan,
            'RMSE': np.nan,
            'NSE': np.nan,
            'PBIAS': np.nan,
            'R2': np.nan
        }
    
    # Mean Absolute Error
    mae = np.mean(np.abs(sim - obs))
    
    # Root Mean Square Error
    rmse = np.sqrt(np.mean((sim - obs) ** 2))
    
    # Nash-Sutcliffe Efficiency
    obs_mean = np.mean(obs)
    nse = 1 - np.sum((obs - sim) ** 2) / np.sum((obs - obs_mean) ** 2)
    
    # Percent Bias
    pbias = 100 * np.sum(obs - sim) / np.sum(obs)
    
    # Coefficient of Determination (R2)
    ss_res = np.sum((obs - sim) ** 2)
    ss_tot = np.sum((obs - obs_mean) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'NSE': nse,
        'PBIAS': pbias,
        'R2': r2
    }


def compare_results(result1: Dict, result2: Dict, tolerance: float = 1e-6) -> Dict:
    """Compare two sets of results for consistency.
    
    Args:
        result1: First result set (original)
        result2: Second result set (new)
        tolerance: Tolerance for considering values equal
        
    Returns:
        Dictionary containing comparison results
    """
    comparison = {
        'is_consistent': True,
        'details': {},
        'max_error': 0.0,
        'mean_error': 0.0
    }
    
    for key in result1.keys():
        if key in result2:
            val1 = result1[key]
            val2 = result2[key]
            
            if isinstance(val1, np.ndarray) and isinstance(val2, np.ndarray):
                # Array comparison
                if val1.shape == val2.shape:
                    abs_error = np.abs(val1 - val2)
                    rel_error = np.abs((val1 - val2) / (np.abs(val1) + 1e-10))
                    
                    max_abs_error = np.max(abs_error)
                    mean_abs_error = np.mean(abs_error)
                    max_rel_error = np.max(rel_error)
                    
                    is_consistent = max_abs_error < tolerance
                    
                    comparison['details'][key] = {
                        'max_abs_error': max_abs_error,
                        'mean_abs_error': mean_abs_error,
                        'max_rel_error': max_rel_error,
                        'is_consistent': is_consistent
                    }
                    
                    comparison['max_error'] = max(comparison['max_error'], max_abs_error)
                    comparison['mean_error'] += mean_abs_error
                    
                    if not is_consistent:
                        comparison['is_consistent'] = False
                else:
                    comparison['details'][key] = {
                        'error': f'Shape mismatch: {val1.shape} vs {val2.shape}',
                        'is_consistent': False
                    }
                    comparison['is_consistent'] = False
            else:
                # Scalar comparison
                abs_error = abs(val1 - val2)
                is_consistent = abs_error < tolerance
                
                comparison['details'][key] = {
                    'value1': val1,
                    'value2': val2,
                    'abs_error': abs_error,
                    'is_consistent': is_consistent
                }
                
                if not is_consistent:
                    comparison['is_consistent'] = False
    
    comparison['mean_error'] /= len(comparison['details']) if comparison['details'] else 1
    
    return comparison


def print_comparison_report(comparison: Dict):
    """Print a formatted comparison report.
    
    Args:
        comparison: Comparison results dictionary
    """
    print("\n" + "="*60)
    print("Comparison Report")
    print("="*60)
    
    if comparison['is_consistent']:
        print("[PASS] Results are completely consistent")
    else:
        print("[FAIL] Results have differences")
    
    print(f"\nMax error: {comparison['max_error']:.2e}")
    print(f"Mean error: {comparison['mean_error']:.2e}")
    
    print("\nDetailed comparison:")
    print("-"*60)
    
    for key, detail in comparison['details'].items():
        if detail.get('is_consistent', False):
            status = "[OK]"
        else:
            status = "[!!]"
        
        if 'max_abs_error' in detail:
            print(f"{status} {key}: max_error={detail['max_abs_error']:.2e}, mean_error={detail['mean_abs_error']:.2e}")
        elif 'abs_error' in detail:
            print(f"{status} {key}: val1={detail['value1']:.6f}, val2={detail['value2']:.6f}, error={detail['abs_error']:.2e}")
    
    print("="*60)


def normalize_parameters(params: np.ndarray, param_ranges: Dict) -> np.ndarray:
    """Normalize parameters to 0-1 range.
    
    Args:
        params: Parameter array
        param_ranges: Dictionary of parameter ranges
        
    Returns:
        Normalized parameter array
    """
    normalized = np.zeros_like(params)
    param_names = list(param_ranges.keys())
    
    for i, name in enumerate(param_names):
        if i < len(params):
            range_info = param_ranges[name]
            min_val = range_info['min']
            max_val = range_info['max']
            normalized[i] = (params[i] - min_val) / (max_val - min_val)
    
    return normalized


def denormalize_parameters(normalized_params: np.ndarray, param_ranges: Dict) -> np.ndarray:
    """Denormalize parameters from 0-1 range to original scale.
    
    Args:
        normalized_params: Normalized parameter array
        param_ranges: Dictionary of parameter ranges
        
    Returns:
        Denormalized parameter array
    """
    denormalized = np.zeros_like(normalized_params)
    param_names = list(param_ranges.keys())
    
    for i, name in enumerate(param_names):
        if i < len(normalized_params):
            range_info = param_ranges[name]
            min_val = range_info['min']
            max_val = range_info['max']
            denormalized[i] = normalized_params[i] * (max_val - min_val) + min_val
    
    return denormalized


def ensure_directory(path: str):
    """Ensure directory exists, create if not.
    
    Args:
        path: Directory path
    """
    Path(path).mkdir(parents=True, exist_ok=True)
