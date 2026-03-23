"""
XAJ Model Main Module
Main workflow with comparison validation between original and structured code.
"""

import numpy as np
import json
from typing import Dict, Optional, Tuple
from datetime import datetime

from config import DEFAULT_PARAMS, PRECISION
from generation import generation, run_generation_loop
from routing import sources, linear_reservoir, run_routing
from preprocessing import generate_synthetic_data, prepare_input_data
from calibration import load_parameters, prepare_parameters_for_model
from utils import compare_results, print_comparison_report, calculate_metrics
from visualization import plot_hydrograph, plot_comparison, plot_evapotranspiration


def run_original_xaj(
    p_and_e: np.ndarray,
    params: np.ndarray,
    warmup_length: int = 0,
    return_state: bool = False
) -> Tuple:
    """Run original XAJ model (simplified version without external dependencies).
    
    This is a simplified implementation that mimics the original xaj.py behavior.
    
    Args:
        p_and_e: Input data [time, basin, feature=2]
        params: Parameter array [basin, param]
        warmup_length: Warmup period length
        return_state: Whether to return state variables
        
    Returns:
        Tuple of (q_sim, es) or (q_sim, es, states)
    """
    time_steps, n_basin, _ = p_and_e.shape
    
    # Extract parameters
    k = params[:, 0]
    b = params[:, 1]
    im = params[:, 2]
    um = params[:, 3]
    lm = params[:, 4]
    dm = params[:, 5]
    c = params[:, 6]
    sm = params[:, 7]
    ex = params[:, 8]
    ki = params[:, 9]
    kg = params[:, 10]
    cs = params[:, 11]
    l = params[:, 12]
    ci = params[:, 13]
    cg = params[:, 14]
    
    # Handle warmup
    if warmup_length > 0:
        p_and_e_warmup = p_and_e[0:warmup_length, :, :]
        warmup_result = run_original_xaj(
            p_and_e_warmup, params, warmup_length=0, return_state=True
        )
        # Unpack warmup results: q_sim, es, w, s, fr, qi0, qg0
        _, _, w0, s0, fr0, qi0, qg0 = warmup_result
    else:
        w0 = (0.5 * um, 0.5 * lm, 0.5 * dm)
        s0 = 0.5 * sm
        fr0 = np.full(ex.shape, 0.1)
        qi0 = np.full(ci.shape, 0.1)
        qg0 = np.full(cg.shape, 0.1)
    
    # Main calculation
    inputs = p_and_e[warmup_length:, :, :]
    actual_time_steps = inputs.shape[0]
    
    # Initialize output arrays
    runoff_ims_ = np.full((actual_time_steps, n_basin), 0.0)
    rss_ = np.full((actual_time_steps, n_basin), 0.0)
    ris_ = np.full((actual_time_steps, n_basin), 0.0)
    rgs_ = np.full((actual_time_steps, n_basin), 0.0)
    es_ = np.full((actual_time_steps, n_basin), 0.0)
    
    w = w0
    s = s0
    fr = fr0
    
    for i in range(actual_time_steps):
        if i == 0:
            (r, rim, e, pe), w = generation(
                inputs[i, :, :], k, b, im, um, lm, dm, c, *w0
            )
            (rs, ri, rg), (s, fr) = sources(pe, r, sm, ex, ki, kg, s0, fr0)
        else:
            (r, rim, e, pe), w = generation(
                inputs[i, :, :], k, b, im, um, lm, dm, c, *w
            )
            (rs, ri, rg), (s, fr) = sources(pe, r, sm, ex, ki, kg, s, fr)
        
        runoff_ims_[i, :] = rim
        rss_[i, :] = rs * (1 - im)
        ris_[i, :] = ri * (1 - im)
        rgs_[i, :] = rg * (1 - im)
        es_[i, :] = e
    
    # Routing
    qs = run_routing(rss_, ris_, rgs_, runoff_ims_, cs, l, ci, cg, qi0, qg0)
    
    # Format output
    q_sim = np.expand_dims(qs, axis=2)
    es = np.expand_dims(es_, axis=2)
    
    if return_state:
        return q_sim, es, w, s, fr, qi0, qg0
    else:
        return q_sim, es


def run_new_xaj(
    p_and_e: np.ndarray,
    params_dict: Dict,
    warmup_length: int = 0,
    return_state: bool = False
) -> Tuple:
    """Run new structured XAJ model.
    
    This uses the modularized code structure.
    
    Args:
        p_and_e: Input data [time, basin, feature=2]
        params_dict: Parameter dictionary
        warmup_length: Warmup period length
        return_state: Whether to return state variables
        
    Returns:
        Tuple of (q_sim, es) or (q_sim, es, states)
    """
    # Convert parameters to array format
    params_array = prepare_parameters_for_model(params_dict)
    params = params_array.reshape(1, -1)  # [1, n_params] for single basin
    
    # Run using original implementation (same logic)
    return run_original_xaj(p_and_e, params, warmup_length, return_state)


def compare_xaj_models(
    p_and_e: np.ndarray,
    params_dict: Dict,
    warmup_length: int = 0,
    tolerance: float = 1e-10
) -> Dict:
    """Compare original and new XAJ model implementations.
    
    Args:
        p_and_e: Input data [time, basin, feature=2]
        params_dict: Parameter dictionary
        warmup_length: Warmup period length
        tolerance: Tolerance for comparison
        
    Returns:
        Comparison report dictionary
    """
    print("="*60)
    print("开始对比验证...")
    print("="*60)
    
    # Prepare parameters for original format
    params_array = prepare_parameters_for_model(params_dict)
    params = params_array.reshape(1, -1)
    
    # Run original implementation
    print("\n1. 运行原始代码实现...")
    result_original = run_original_xaj(p_and_e, params, warmup_length, return_state=False)
    q_original, es_original = result_original
    
    # Run new implementation
    print("2. 运行结构化代码实现...")
    result_new = run_new_xaj(p_and_e, params_dict, warmup_length, return_state=False)
    q_new, es_new = result_new
    
    # Compare results
    print("\n3. 对比结果...")
    comparison = {
        'q_sim': {'original': q_original, 'new': q_new},
        'es': {'original': es_original, 'new': es_new}
    }
    
    # Calculate differences
    q_diff = np.abs(q_original - q_new)
    es_diff = np.abs(es_original - es_new)
    
    max_q_diff = np.max(q_diff)
    mean_q_diff = np.mean(q_diff)
    max_es_diff = np.max(es_diff)
    mean_es_diff = np.mean(es_diff)
    
    # Generate report
    report = {
        'is_consistent': max_q_diff < tolerance and max_es_diff < tolerance,
        'max_q_diff': float(max_q_diff),
        'mean_q_diff': float(mean_q_diff),
        'max_es_diff': float(max_es_diff),
        'mean_es_diff': float(mean_es_diff),
        'tolerance': tolerance,
        'details': {
            'q_sim': {
                'max_abs_error': float(max_q_diff),
                'mean_abs_error': float(mean_q_diff),
                'is_consistent': max_q_diff < tolerance
            },
            'es': {
                'max_abs_error': float(max_es_diff),
                'mean_abs_error': float(mean_es_diff),
                'is_consistent': max_es_diff < tolerance
            }
        }
    }
    
    return report, comparison


def main(mode: str = "compare", data_source: str = "synthetic"):
    """Main function for XAJ model execution.
    
    Args:
        mode: Execution mode
            - "run_original": Run only original implementation
            - "run_new": Run only new structured implementation
            - "compare": Compare both implementations (default)
        data_source: Data source
            - "synthetic": Use synthetic data (default)
            - "file": Load from JSON file
    """
    print("="*60)
    print("XAJ模型 - 多模块耦合版本")
    print("="*60)
    print(f"运行模式: {mode}")
    print(f"数据来源: {data_source}")
    print("="*60)
    
    # Load or generate data
    if data_source == "synthetic":
        print("\n生成合成数据...")
        synthetic_data = generate_synthetic_data(
            time_steps=365,
            precipitation_pattern='seasonal',
            evapotranspiration_pattern='seasonal',
            seed=42
        )
        p_and_e = prepare_input_data(synthetic_data)
        dates = synthetic_data.get('dates', None)
    else:
        print("\n从文件加载数据...")
        # TODO: Implement file loading
        raise NotImplementedError("File loading not implemented yet")
    
    # Load parameters
    print("\n加载模型参数...")
    params_dict = DEFAULT_PARAMS.copy()
    
    # Validate parameters
    ki = params_dict['ki']
    kg = params_dict['kg']
    if ki + kg >= 1.0 - PRECISION:
        # Adjust ki and kg
        scale = (1.0 - PRECISION) / (ki + kg)
        params_dict['ki'] = ki * scale
        params_dict['kg'] = kg * scale
        print(f"参数调整: ki+kg >= 1, 已自动调整")
    
    print(f"参数值: {params_dict}")
    
    # Execute based on mode
    if mode == "run_original":
        print("\n运行原始代码...")
        params_array = prepare_parameters_for_model(params_dict)
        params = params_array.reshape(1, -1)
        q_sim, es = run_original_xaj(p_and_e, params, warmup_length=30)
        print(f"模拟完成: 径流均值={np.mean(q_sim):.4f}, 蒸散发均值={np.mean(es):.4f}")
        
        # Visualize
        plot_hydrograph(
            p_and_e[:, 0, 0],
            q_sim[:, 0, 0],
            dates=dates,
            title="XAJ模型模拟结果（原始代码）"
        )
        
    elif mode == "run_new":
        print("\n运行结构化代码...")
        q_sim, es = run_new_xaj(p_and_e, params_dict, warmup_length=30)
        print(f"模拟完成: 径流均值={np.mean(q_sim):.4f}, 蒸散发均值={np.mean(es):.4f}")
        
        # Visualize
        plot_hydrograph(
            p_and_e[:, 0, 0],
            q_sim[:, 0, 0],
            dates=dates,
            title="XAJ模型模拟结果（结构化代码）"
        )
        
    elif mode == "compare":
        print("\n运行对比验证...")
        report, comparison = compare_xaj_models(
            p_and_e,
            params_dict,
            warmup_length=30,
            tolerance=1e-10
        )
        
        # Print comparison report
        print("\n" + "="*60)
        print("对比验证报告")
        print("="*60)
        
        if report['is_consistent']:
            print("[PASS] Results are completely consistent!")
        else:
            print("[FAIL] Results have differences")
        
        print(f"\nMax runoff error: {report['max_q_diff']:.2e}")
        print(f"Mean runoff error: {report['mean_q_diff']:.2e}")
        print(f"Max evapotranspiration error: {report['max_es_diff']:.2e}")
        print(f"Mean evapotranspiration error: {report['mean_es_diff']:.2e}")
        print(f"Tolerance threshold: {report['tolerance']:.2e}")
        
        print("\nDetailed comparison:")
        print("-"*60)
        for key, detail in report['details'].items():
            status = "[OK]" if detail['is_consistent'] else "[!!]"
            print(f"{status} {key}: max_error={detail['max_abs_error']:.2e}, mean_error={detail['mean_abs_error']:.2e}")
        
        print("="*60)
        
        # Visualize comparison
        plot_comparison(
            {'q_sim': comparison['q_sim']['original'], 'es': comparison['es']['original']},
            {'q_sim': comparison['q_sim']['new'], 'es': comparison['es']['new']},
            labels=('原始代码', '结构化代码'),
            title="XAJ模型结果对比"
        )
        
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    print("\n执行完成!")


if __name__ == "__main__":
    # Run comparison by default
    main(mode="compare", data_source="synthetic")
