"""
XAJ Model Preprocessing Module
Contains data loading, cleaning, and synthetic data generation functions.
"""

import json
import numpy as np
from typing import Dict, Tuple, Optional
from datetime import datetime, timedelta


def save_json(data: Dict, file_path: str):
    """Save data to JSON file.
    
    Args:
        data: Data dictionary to save
        file_path: Output file path
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_data_from_json(file_path: str) -> Dict:
    """Load hydrological data from JSON file.
    
    Args:
        file_path: Path to JSON data file
        
    Returns:
        Dictionary containing hydrological data
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data


def prepare_input_data(data: Dict) -> np.ndarray:
    """Prepare input data array from JSON data.
    
    Args:
        data: Dictionary containing precipitation and evapotranspiration data
        
    Returns:
        Input data array [time, basin, feature=2]
    """
    # Extract precipitation and evapotranspiration
    precipitation = np.array(data.get('precipitation', []))
    evapotranspiration = np.array(data.get('evapotranspiration', []))
    
    # Get dimensions
    time_steps = len(precipitation)
    n_basin = 1  # Default single basin
    
    # Handle multi-basin case
    if precipitation.ndim > 1:
        n_basin = precipitation.shape[1]
    
    # Create input array
    p_and_e = np.zeros((time_steps, n_basin, 2))
    p_and_e[:, :, 0] = precipitation.reshape(time_steps, n_basin) if precipitation.ndim == 1 else precipitation
    p_and_e[:, :, 1] = evapotranspiration.reshape(time_steps, n_basin) if evapotranspiration.ndim == 1 else evapotranspiration
    
    return p_and_e


def generate_synthetic_data(
    time_steps: int = 365,
    n_basin: int = 1,
    precipitation_pattern: str = 'seasonal',
    evapotranspiration_pattern: str = 'seasonal',
    seed: int = 42
) -> Dict:
    """Generate synthetic hydrological data for testing.
    
    Args:
        time_steps: Number of time steps (days)
        n_basin: Number of basins
        precipitation_pattern: Pattern for precipitation ('uniform', 'seasonal', 'storm')
        evapotranspiration_pattern: Pattern for evapotranspiration ('uniform', 'seasonal')
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary containing synthetic data
    """
    np.random.seed(seed)
    
    # Generate time series
    dates = []
    start_date = datetime(2020, 1, 1)
    for i in range(time_steps):
        dates.append((start_date + timedelta(days=i)).strftime('%Y-%m-%d'))
    
    # Generate precipitation
    if precipitation_pattern == 'uniform':
        # Uniform precipitation with some randomness
        precipitation = np.random.uniform(0, 10, (time_steps, n_basin))
    elif precipitation_pattern == 'seasonal':
        # Seasonal precipitation pattern
        t = np.arange(time_steps)
        seasonal = 5 + 3 * np.sin(2 * np.pi * t / 365)  # Seasonal variation
        noise = np.random.normal(0, 1, (time_steps, n_basin))
        precipitation = np.maximum(0, seasonal.reshape(-1, 1) + noise)
        
        # Add some storm events
        storm_days = np.random.choice(time_steps, size=int(time_steps * 0.1), replace=False)
        precipitation[storm_days, :] += np.random.uniform(20, 50, (len(storm_days), n_basin))
    elif precipitation_pattern == 'storm':
        # Mostly dry with occasional storms
        precipitation = np.zeros((time_steps, n_basin))
        storm_days = np.random.choice(time_steps, size=int(time_steps * 0.05), replace=False)
        precipitation[storm_days, :] = np.random.uniform(30, 80, (len(storm_days), n_basin))
    else:
        raise ValueError(f"Unknown precipitation pattern: {precipitation_pattern}")
    
    # Generate evapotranspiration
    if evapotranspiration_pattern == 'uniform':
        evapotranspiration = np.random.uniform(2, 5, (time_steps, n_basin))
    elif evapotranspiration_pattern == 'seasonal':
        t = np.arange(time_steps)
        seasonal = 3 + 2 * np.sin(2 * np.pi * t / 365)  # Higher in summer
        noise = np.random.normal(0, 0.5, (time_steps, n_basin))
        evapotranspiration = np.maximum(0.5, seasonal.reshape(-1, 1) + noise)
    else:
        raise ValueError(f"Unknown evapotranspiration pattern: {evapotranspiration_pattern}")
    
    return {
        'dates': dates,
        'precipitation': precipitation.tolist(),
        'evapotranspiration': evapotranspiration.tolist(),
        'time_steps': time_steps,
        'n_basin': n_basin,
        'pattern': {
            'precipitation': precipitation_pattern,
            'evapotranspiration': evapotranspiration_pattern
        }
    }


def validate_data(data: Dict) -> Tuple[bool, list]:
    """Validate hydrological data.
    
    Args:
        data: Dictionary containing hydrological data
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    # Check required fields
    required_fields = ['precipitation', 'evapotranspiration']
    for field in required_fields:
        if field not in data:
            errors.append(f"Missing required field: {field}")
    
    # Check data consistency
    if 'precipitation' in data and 'evapotranspiration' in data:
        prec = np.array(data['precipitation'])
        evap = np.array(data['evapotranspiration'])
        
        if prec.shape != evap.shape:
            errors.append("Precipitation and evapotranspiration have different shapes")
        
        # Check for negative values
        if (prec < 0).any():
            errors.append("Precipitation contains negative values")
        
        if (evap < 0).any():
            errors.append("Evapotranspiration contains negative values")
    
    return len(errors) == 0, errors


def split_data(
    data: Dict,
    train_ratio: float = 0.7,
    warmup_ratio: float = 0.1
) -> Tuple[Dict, Dict, Dict]:
    """Split data into warmup, training, and validation sets.
    
    Args:
        data: Dictionary containing hydrological data
        train_ratio: Ratio of data for training (excluding warmup)
        warmup_ratio: Ratio of data for warmup
        
    Returns:
        Tuple of (warmup_data, train_data, validation_data)
    """
    n_total = len(data['precipitation'])
    n_warmup = int(n_total * warmup_ratio)
    n_train = int((n_total - n_warmup) * train_ratio)
    
    warmup_data = {
        'precipitation': data['precipitation'][:n_warmup],
        'evapotranspiration': data['evapotranspiration'][:n_warmup],
        'dates': data.get('dates', [])[:n_warmup]
    }
    
    train_data = {
        'precipitation': data['precipitation'][n_warmup:n_warmup + n_train],
        'evapotranspiration': data['evapotranspiration'][n_warmup:n_warmup + n_train],
        'dates': data.get('dates', [])[n_warmup:n_warmup + n_train]
    }
    
    validation_data = {
        'precipitation': data['precipitation'][n_warmup + n_train:],
        'evapotranspiration': data['evapotranspiration'][n_warmup + n_train:],
        'dates': data.get('dates', [])[n_warmup + n_train:]
    }
    
    return warmup_data, train_data, validation_data


def calculate_statistics(data: Dict) -> Dict:
    """Calculate statistical summary of hydrological data.
    
    Args:
        data: Dictionary containing hydrological data
        
    Returns:
        Dictionary containing statistical summary
    """
    prec = np.array(data['precipitation'])
    evap = np.array(data['evapotranspiration'])
    
    return {
        'precipitation': {
            'mean': np.mean(prec),
            'std': np.std(prec),
            'min': np.min(prec),
            'max': np.max(prec),
            'total': np.sum(prec)
        },
        'evapotranspiration': {
            'mean': np.mean(evap),
            'std': np.std(evap),
            'min': np.min(evap),
            'max': np.max(evap),
            'total': np.sum(evap)
        },
        'time_steps': len(prec)
    }
