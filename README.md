# 新安江模型 (XAJ Model) - 多模块耦合版本

## 项目简介

本项目是新安江模型的结构化重构版本，将原有的单文件代码拆分为多个独立模块，便于维护、扩展和复用。项目包含完整的对比验证功能，确保重构后的代码与原始代码计算结果完全一致。

## 项目结构

```
XAJ_Model/
│
├── main.py                 # 主流程入口（含对比验证功能）
├── config.py               # 配置模块（参数定义、默认值、取值范围）
├── generation.py           # 产流模块（蒸散发、产流计算）
├── routing.py              # 汇流模块（水源划分、河道汇流）
├── calibration.py          # 调参模块（参数读取、优化算法）
├── preprocessing.py        # 预处理模块（数据读取、清洗、合成数据生成）
├── visualization.py        # 可视化模块（水文过程线、对比图）
├── utils.py                # 工具模块（日志、文件操作、误差计算）
│
├── xaj_original.py         # 原始代码备份（用于对比验证）
├── xaj.py                  # 原始代码
├── xaj_slw.py              # 另一版本XAJ（SMS_3+LAG_3算法）
├── semi_xaj.py             # 半分布式XAJ版本
│
├── README.md               # 本文件
│
└── example_data/           # 示例数据目录
    ├── parameters.json     # 模型参数配置文件
    ├── synthetic_data.json # 日尺度合成数据（365天）
    └── hourly_data.csv     # 小时尺度数据（7天，含暴雨过程）
```

## 快速开始

### 1. 环境要求

- Python 3.8+
- NumPy
- SciPy
- Matplotlib
- Pandas（可选，用于数据处理）

### 2. 安装依赖

```bash
pip install numpy scipy matplotlib pandas
```

### 3. 运行示例

```bash
# 运行对比验证（默认模式）
python main.py

# 只运行原始代码
python -c "from main import main; main(mode='run_original')"

# 只运行结构化代码
python -c "from main import main; main(mode='run_new')"
```

## 模块详细说明

### 1. 配置模块 (config.py)

定义模型的核心配置：

- **默认参数值**：模型运行所需的15个参数的默认值
- **参数取值范围**：用于参数率定的上下限
- **参数说明**：每个参数的物理含义和单位
- **验证规则**：参数间的约束关系（如 ki + kg < 1）

**主要参数列表**：

| 参数 | 符号 | 单位 | 说明 |
|------|------|------|------|
| k | K | - | 蒸散发系数 |
| b | B | - | 蓄水容量曲线指数 |
| im | IMP | - | 不透水面积比例 |
| um | UM | mm | 上层土壤蓄水容量 |
| lm | LM | mm | 下层土壤蓄水容量 |
| dm | DM | mm | 深层土壤蓄水容量 |
| c | C | - | 深层蒸散发系数 |
| sm | SM | mm | 自由水蓄水容量 |
| ex | EX | - | 自由水容量曲线指数 |
| ki | KI | - | 壤中流出流系数 |
| kg | KG | - | 地下水出流系数 |
| cs | CS | - | 河网蓄水常数 |
| l | L | 时段 | 滞时 |
| ci | CI | - | 壤中流消退系数 |
| cg | CG | - | 地下水消退系数 |

### 2. 产流模块 (generation.py)

实现新安江模型的产流计算：

- **calculate_evap()**: 三层蒸散发计算
  - 上层蒸散发 (eu)
  - 下层蒸散发 (el)
  - 深层蒸散发 (ed)

- **calculate_prcp_runoff()**: 降雨径流计算
  - 蓄水容量曲线法
  - 不透水面积产流

- **calculate_w_storage()**: 土壤含水量更新
  - 三层土壤水分平衡

- **generation()**: 产流主函数
  - 整合上述子函数
  - 输出径流和更新后的土壤含水量

### 3. 汇流模块 (routing.py)

实现径流的水源划分和河道汇流：

- **sources()**: 水源划分
  - 地表径流 (Rs)
  - 壤中流 (Ri)
  - 地下径流 (Rg)

- **linear_reservoir()**: 线性水库调蓄
  - 用于壤中流和地下径流的调蓄

- **uh_gamma()**: 单位线汇流
  - Gamma分布单位线
  - 用于地表径流的汇流计算

- **run_routing()**: 汇流主函数
  - Muskingum河道演算
  - 滞时处理

### 4. 调参模块 (calibration.py)

实现参数管理和优化：

- **load_parameters()**: 从JSON文件加载参数
- **save_parameters()**: 保存参数到JSON文件
- **validate_params()**: 验证参数有效性
- **sce_ua_optimization()**: SCE-UA优化算法
- **parameter_sensitivity_analysis()**: 参数敏感性分析

### 5. 预处理模块 (preprocessing.py)

实现数据的读取和处理：

- **generate_synthetic_data()**: 生成合成测试数据
  - 支持多种降水模式：均匀、季节性、暴雨
  - 支持多种蒸散发模式：均匀、季节性

- **load_data_from_json()**: 从JSON文件加载数据
- **prepare_input_data()**: 准备模型输入数组
- **validate_data()**: 验证数据有效性
- **split_data()**: 数据分割（预热期、率定期、验证期）

### 6. 可视化模块 (visualization.py)

提供丰富的可视化功能：

- **plot_hydrograph()**: 水文过程线
  - 降水柱状图
  - 模拟/实测径流对比

- **plot_comparison()**: 结果对比图
  - 时间序列对比
  - 散点图对比
  - 差异分析

- **plot_soil_moisture()**: 土壤含水量变化图
- **plot_water_balance()**: 水量平衡分析图
- **plot_runoff_components()**: 径流成分分析图
- **plot_performance_metrics()**: 模型性能指标图

### 7. 工具模块 (utils.py)

提供通用工具函数：

- **setup_logger()**: 日志记录器配置
- **load_json() / save_json()**: JSON文件操作
- **calculate_metrics()**: 模型性能指标计算
  - MAE（平均绝对误差）
  - RMSE（均方根误差）
  - NSE（纳什效率系数）
  - PBIAS（百分比偏差）
  - R²（决定系数）

- **compare_results()**: 结果对比分析
- **print_comparison_report()**: 打印对比报告

## 示例数据说明

### 1. parameters.json

模型参数配置文件，包含：
- 15个模型参数的默认值
- 参数取值范围（用于率定）
- 率定算法配置

### 2. synthetic_data.json

日尺度合成数据（365天）：
- 时间范围：2020-01-01 至 2020-12-30
- 降水模式：季节性变化 + 随机暴雨
- 蒸散发模式：季节性变化

### 3. hourly_data.csv

小时尺度数据（7天，168小时）：
- 时间范围：2023-07-15 00:00 至 2023-07-21 23:00
- 包含一个完整的暴雨过程
- 降水：峰值约29 mm/h，总量约470 mm
- 蒸散发：日内变化，白天高（0.5 mm/h），夜间低（0.1 mm/h）

**CSV文件格式**：
```csv
timestamp,precipitation_mm,evapotranspiration_mm
2023-07-15 00:00:00,0.0,0.100
2023-07-15 01:00:00,0.0,0.100
...
```

## 对比验证功能

本项目的核心特色是**对比验证功能**，确保重构的正确性：

### 运行模式

```python
# 1. 对比模式（默认）- 同时运行两种实现并比较结果
main(mode="compare")

# 2. 原始模式 - 只运行原始代码
main(mode="run_original")

# 3. 新模式 - 只运行结构化代码
main(mode="run_new")
```

### 验证指标

- **最大绝对误差**：两组结果的最大差异
- **平均绝对误差**：两组结果的平均差异
- **容差阈值**：判断一致性的标准（默认 1e-10）

### 验证报告示例

```
============================================================
对比验证报告
============================================================
[PASS] Results are completely consistent!

Max runoff error: 0.00e+00
Mean runoff error: 0.00e+00
Max evapotranspiration error: 0.00e+00
Mean evapotranspiration error: 0.00e+00
Tolerance threshold: 1.00e-10
```

## 扩展指南

### 1. 添加新的汇流方法

在 `routing.py` 中添加新函数：

```python
def routing_new_method(rss, ris, rgs, params):
    """新的汇流方法实现"""
    # 实现代码
    return qs
```

然后在 `main.py` 中调用。

### 2. 添加新的优化算法

在 `calibration.py` 中添加新函数：

```python
def pso_optimization(objective_func, param_ranges, **kwargs):
    """粒子群优化算法"""
    # 实现代码
    return best_params, best_value
```

### 3. 添加新的可视化图表

在 `visualization.py` 中添加新函数：

```python
def plot_new_chart(data, **kwargs):
    """新的可视化图表"""
    # 实现代码
    plt.show()
```

## 常见问题

### Q1: 如何修改模型参数？

编辑 `example_data/parameters.json` 文件，或在代码中直接修改：

```python
from config import DEFAULT_PARAMS
DEFAULT_PARAMS['k'] = 0.9  # 修改蒸散发系数
```

### Q2: 如何使用自己的数据？

准备CSV文件（格式参考 `hourly_data.csv`），然后在代码中加载：

```python
import pandas as pd
from preprocessing import prepare_input_data

df = pd.read_csv('your_data.csv')
data = {
    'precipitation': df['precipitation_mm'].values.reshape(-1, 1, 1),
    'evapotranspiration': df['evapotranspiration_mm'].values.reshape(-1, 1, 1)
}
p_and_e = prepare_input_data(data)
```

### Q3: 对比验证失败怎么办？

1. 检查参数是否一致
2. 检查输入数据是否一致
3. 检查初始条件是否一致
4. 查看详细的误差报告定位问题

## 参考文献

1. 赵人俊. 流域水文模拟[M]. 北京: 水利电力出版社, 1984.
2. 包为民. 水文预报(第五版)[M]. 北京: 中国水利水电出版社, 2019.
3. 赵人俊. 新安江模型参数的分析[J]. 水文, 1988.

## 许可证

本项目遵循 MIT 许可证。

## 联系方式

如有问题或建议，请通过以下方式联系：
- 提交 Issue
- 发送邮件

---

**最后更新**: 2024年
**版本**: 2.0
