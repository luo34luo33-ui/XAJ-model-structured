"""
XAJ Model Visualization Module
Contains plotting functions for hydrological process visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, Tuple, List
from datetime import datetime


def setup_chinese_font():
    """Setup Chinese font for matplotlib."""
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False


# Initialize Chinese font
setup_chinese_font()


def plot_hydrograph(
    precipitation: np.ndarray,
    simulated: np.ndarray,
    observed: Optional[np.ndarray] = None,
    dates: Optional[List[str]] = None,
    title: str = "水文过程线",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
):
    """Plot hydrograph with precipitation, simulated and optional observed streamflow.
    
    Args:
        precipitation: Precipitation array [time]
        simulated: Simulated streamflow array [time]
        observed: Optional observed streamflow array [time]
        dates: Optional date strings for x-axis
        title: Plot title
        save_path: Optional path to save figure
        figsize: Figure size
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[1, 3])
    
    # Plot precipitation
    if dates is not None:
        x = range(len(dates))
        ax1.bar(x, precipitation, color='steelblue', alpha=0.7, width=0.8)
        ax1.set_xticks(x[::max(1, len(x)//10)])
        ax1.set_xticklabels([dates[i] for i in x[::max(1, len(x)//10)]], rotation=45)
    else:
        ax1.bar(range(len(precipitation)), precipitation, color='steelblue', alpha=0.7)
    
    ax1.set_ylabel('降水 (mm)')
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.3)
    
    # Plot streamflow
    if dates is not None:
        x = range(len(dates))
        ax2.plot(x, simulated, 'b-', linewidth=1.5, label='模拟径流')
        if observed is not None:
            ax2.plot(x, observed, 'r--', linewidth=1.5, label='实测径流')
        ax2.set_xticks(x[::max(1, len(x)//10)])
        ax2.set_xticklabels([dates[i] for i in x[::max(1, len(x)//10)]], rotation=45)
    else:
        ax2.plot(simulated, 'b-', linewidth=1.5, label='模拟径流')
        if observed is not None:
            ax2.plot(observed, 'r--', linewidth=1.5, label='实测径流')
    
    ax2.set_xlabel('时间')
    ax2.set_ylabel('流量 (m³/s)')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


plot_hydrograph_with_precip = plot_hydrograph


def plot_evapotranspiration(
    potential_ev: np.ndarray,
    actual_ev: np.ndarray,
    dates: Optional[List[str]] = None,
    title: str = "蒸散发过程线",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
):
    """Plot evapotranspiration process.
    
    Args:
        potential_ev: Potential evapotranspiration array
        actual_ev: Actual evapotranspiration array
        dates: Optional date strings
        title: Plot title
        save_path: Optional save path
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if dates is not None:
        x = range(len(dates))
        ax.plot(x, potential_ev, 'r-', linewidth=1.5, label='潜在蒸散发')
        ax.plot(x, actual_ev, 'b-', linewidth=1.5, label='实际蒸散发')
        ax.set_xticks(x[::max(1, len(x)//10)])
        ax.set_xticklabels([dates[i] for i in x[::max(1, len(x)//10)]], rotation=45)
    else:
        ax.plot(potential_ev, 'r-', linewidth=1.5, label='潜在蒸散发')
        ax.plot(actual_ev, 'b-', linewidth=1.5, label='实际蒸散发')
    
    ax.set_xlabel('时间')
    ax.set_ylabel('蒸散发 (mm)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_soil_moisture(
    wu: np.ndarray,
    wl: np.ndarray,
    wd: np.ndarray,
    dates: Optional[List[str]] = None,
    title: str = "土壤含水量变化",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
):
    """Plot soil moisture content for three layers.
    
    Args:
        wu: Upper layer soil moisture
        wl: Lower layer soil moisture
        wd: Deep layer soil moisture
        dates: Optional date strings
        title: Plot title
        save_path: Optional save path
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if dates is not None:
        x = range(len(dates))
        ax.plot(x, wu, 'g-', linewidth=1.5, label='上层')
        ax.plot(x, wl, 'b-', linewidth=1.5, label='下层')
        ax.plot(x, wd, 'r-', linewidth=1.5, label='深层')
        ax.set_xticks(x[::max(1, len(x)//10)])
        ax.set_xticklabels([dates[i] for i in x[::max(1, len(x)//10)]], rotation=45)
    else:
        ax.plot(wu, 'g-', linewidth=1.5, label='上层')
        ax.plot(wl, 'b-', linewidth=1.5, label='下层')
        ax.plot(wd, 'r-', linewidth=1.5, label='深层')
    
    ax.set_xlabel('时间')
    ax.set_ylabel('土壤含水量 (mm)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_water_balance(
    precipitation: np.ndarray,
    evapotranspiration: np.ndarray,
    runoff: np.ndarray,
    title: str = "水量平衡分析",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
):
    """Plot water balance analysis.
    
    Args:
        precipitation: Precipitation array
        evapotranspiration: Evapotranspiration array
        runoff: Runoff array
        title: Plot title
        save_path: Optional save path
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate cumulative values
    cum_precip = np.cumsum(precipitation)
    cum_evap = np.cumsum(evapotranspiration)
    cum_runoff = np.cumsum(runoff)
    
    x = range(len(precipitation))
    
    ax.plot(x, cum_precip, 'b-', linewidth=2, label='累计降水')
    ax.plot(x, cum_evap, 'r-', linewidth=2, label='累计蒸散发')
    ax.plot(x, cum_runoff, 'g-', linewidth=2, label='累计径流')
    
    # Calculate storage change
    storage = cum_precip - cum_evap - cum_runoff
    ax.plot(x, storage, 'k--', linewidth=1.5, label='蓄水变化')
    
    ax.set_xlabel('时间步')
    ax.set_ylabel('水量 (mm)')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_comparison(
    result1: Dict,
    result2: Dict,
    labels: Tuple[str, str] = ('原始代码', '结构化代码'),
    title: str = "结果对比",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 10)
):
    """Plot comparison between two sets of results.
    
    Args:
        result1: First result dictionary
        result2: Second result dictionary
        labels: Labels for the two results
        title: Plot title
        save_path: Optional save path
        figsize: Figure size
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Plot runoff comparison
    if 'q_sim' in result1 and 'q_sim' in result2:
        ax = axes[0, 0]
        q1 = result1['q_sim'].flatten() if hasattr(result1['q_sim'], 'flatten') else result1['q_sim']
        q2 = result2['q_sim'].flatten() if hasattr(result2['q_sim'], 'flatten') else result2['q_sim']
        
        ax.plot(q1, 'b-', linewidth=1.5, label=labels[0])
        ax.plot(q2, 'r--', linewidth=1.5, label=labels[1])
        ax.set_xlabel('时间步')
        ax.set_ylabel('流量 (m³/s)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title('径流对比')
    
    # Plot evapotranspiration comparison
    if 'es' in result1 and 'es' in result2:
        ax = axes[0, 1]
        e1 = result1['es'].flatten() if hasattr(result1['es'], 'flatten') else result1['es']
        e2 = result2['es'].flatten() if hasattr(result2['es'], 'flatten') else result2['es']
        
        ax.plot(e1, 'b-', linewidth=1.5, label=labels[0])
        ax.plot(e2, 'r--', linewidth=1.5, label=labels[1])
        ax.set_xlabel('时间步')
        ax.set_ylabel('蒸散发 (mm)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title('蒸散发对比')
    
    # Plot difference
    if 'q_sim' in result1 and 'q_sim' in result2:
        ax = axes[1, 0]
        q1 = result1['q_sim'].flatten() if hasattr(result1['q_sim'], 'flatten') else result1['q_sim']
        q2 = result2['q_sim'].flatten() if hasattr(result2['q_sim'], 'flatten') else result2['q_sim']
        
        diff = q1 - q2
        ax.plot(diff, 'g-', linewidth=1.5)
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax.set_xlabel('时间步')
        ax.set_ylabel('差异')
        ax.grid(True, alpha=0.3)
        ax.set_title('径流差异')
    
    # Plot scatter comparison
    if 'q_sim' in result1 and 'q_sim' in result2:
        ax = axes[1, 1]
        q1 = result1['q_sim'].flatten() if hasattr(result1['q_sim'], 'flatten') else result1['q_sim']
        q2 = result2['q_sim'].flatten() if hasattr(result2['q_sim'], 'flatten') else result2['q_sim']
        
        ax.scatter(q1, q2, alpha=0.5, s=10)
        
        # Add 1:1 line
        min_val = min(np.min(q1), np.min(q2))
        max_val = max(np.max(q1), np.max(q2))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.grid(True, alpha=0.3)
        ax.set_title('散点对比')
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_runoff_components(
    rs: np.ndarray,
    ri: np.ndarray,
    rg: np.ndarray,
    dates: Optional[List[str]] = None,
    title: str = "径流成分分析",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
):
    """Plot runoff components (surface, interflow, groundwater).
    
    Args:
        rs: Surface runoff
        ri: Interflow
        rg: Groundwater runoff
        dates: Optional date strings
        title: Plot title
        save_path: Optional save path
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if dates is not None:
        x = range(len(dates))
        ax.stackplot(x, rs, ri, rg, labels=['地表径流', '壤中流', '地下径流'],
                     colors=['steelblue', 'forestgreen', 'darkorange'], alpha=0.7)
        ax.set_xticks(x[::max(1, len(x)//10)])
        ax.set_xticklabels([dates[i] for i in x[::max(1, len(x)//10)]], rotation=45)
    else:
        ax.stackplot(range(len(rs)), rs, ri, rg, labels=['地表径流', '壤中流', '地下径流'],
                     colors=['steelblue', 'forestgreen', 'darkorange'], alpha=0.7)
    
    ax.set_xlabel('时间')
    ax.set_ylabel('流量 (m³/s)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_performance_metrics(
    metrics: Dict,
    title: str = "模型性能评估",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6)
):
    """Plot model performance metrics.
    
    Args:
        metrics: Dictionary containing performance metrics
        title: Plot title
        save_path: Optional save path
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())
    
    bars = ax.bar(metric_names, metric_values, color=['steelblue', 'forestgreen', 'darkorange', 'crimson'])
    
    # Add value labels on bars
    for bar, value in zip(bars, metric_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{value:.4f}', ha='center', va='bottom')
    
    ax.set_ylabel('指标值')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
