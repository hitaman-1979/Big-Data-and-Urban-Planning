"""
城市空间演进模拟（AI-Agent + UBI）

该模块提供一个可运行的、可扩展的城市空间演化仿真框架：
1. 使用二维网格表示城市空间。
2. 使用家庭代理（HouseholdAgent）表示居住与就业决策。
3. 引入 AI-Agent 覆盖率（提高匹配效率、降低通勤成本感知）。
4. 引入 UBI 政策（提高底层收入稳定性，缓解空间不平等）。
5. 输出可视化与验证指标。

依赖：numpy, pandas, matplotlib
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class SimulationConfig:
    """仿真参数配置。"""

    grid_size: int = 30
    n_households: int = 1600
    n_steps: int = 40
    seed: int = 42

    # 城市结构参数
    center_rent_peak: float = 40.0
    rent_gradient: float = 1.8
    transit_hubs: Tuple[Tuple[int, int], ...] = ((8, 8), (20, 12), (12, 24), (24, 23))

    # 经济参数
    base_income_mean: float = 100.0
    base_income_std: float = 20.0
    job_opportunity_base: float = 60.0

    # 行为权重
    alpha_job: float = 0.6
    beta_rent: float = 0.35
    gamma_access: float = 0.25

    # 干预参数（关键）
    ai_coverage: float = 0.0
    ubi_level: float = 0.0


class UrbanEvolutionSimulator:
    """城市空间演化模拟器。"""

    def __init__(self, config: SimulationConfig):
        self.cfg = config
        self.rng = np.random.default_rng(config.seed)

        self.grid_size = config.grid_size
        self.xy = self._build_grid_coordinates()
        self.center = np.array([self.grid_size // 2, self.grid_size // 2])

        self.base_rent = self._init_base_rent_surface()
        self.accessibility = self._init_accessibility_surface()

        self.agent_positions = self.rng.integers(0, self.grid_size, size=(config.n_households, 2))
        self.agent_income = self.rng.normal(
            config.base_income_mean, config.base_income_std, size=config.n_households
        ).clip(35, None)

        self.history: List[Dict[str, float]] = []
        self.last_density: np.ndarray | None = None

    def _build_grid_coordinates(self) -> np.ndarray:
        x, y = np.meshgrid(np.arange(self.grid_size), np.arange(self.grid_size), indexing="ij")
        return np.stack([x, y], axis=-1)

    def _distance_to_center(self) -> np.ndarray:
        return np.linalg.norm(self.xy - self.center, axis=-1)

    def _init_base_rent_surface(self) -> np.ndarray:
        distance = self._distance_to_center()
        rent = self.cfg.center_rent_peak - self.cfg.rent_gradient * distance
        local_noise = self.rng.normal(0.0, 1.0, size=rent.shape)
        return np.clip(rent + local_noise, 8.0, None)

    def _init_accessibility_surface(self) -> np.ndarray:
        acc = np.zeros((self.grid_size, self.grid_size), dtype=float)
        for hub in self.cfg.transit_hubs:
            d = np.linalg.norm(self.xy - np.array(hub), axis=-1)
            acc += np.exp(-d / 5.5)
        acc = (acc - acc.min()) / (acc.max() - acc.min() + 1e-9)
        return acc

    def _compute_job_surface(self) -> np.ndarray:
        """
        就业机会面：
        - 基础就业机会叠加可达性
        - AI 覆盖率提高匹配效率，尤其在可达性高区域更明显
        """
        cfg = self.cfg
        ai_boost = 1.0 + cfg.ai_coverage * (0.4 + 0.6 * self.accessibility)
        return cfg.job_opportunity_base * (0.7 + 0.8 * self.accessibility) * ai_boost

    def _cell_utility_for_income(self, income: float, rent_surface: np.ndarray, job_surface: np.ndarray) -> np.ndarray:
        """给定家庭收入计算全局区位效用。"""
        cfg = self.cfg

        effective_income = income + cfg.ubi_level

        # AI 影响：提高信息效率，降低实际通勤/搜寻摩擦成本（通过放大可达性收益体现）
        access_gain = (1.0 + 0.5 * cfg.ai_coverage) * self.accessibility

        disposable = np.maximum(effective_income - rent_surface, 1e-3)
        utility = (
            cfg.alpha_job * np.log1p(job_surface)
            + cfg.gamma_access * access_gain
            + (1.0 - cfg.alpha_job - cfg.gamma_access) * np.log(disposable)
            - cfg.beta_rent * (rent_surface / np.maximum(effective_income, 1.0))
        )

        return utility

    def _update_positions(self, dynamic_rent: np.ndarray, job_surface: np.ndarray) -> None:
        """家庭基于效用进行迁移选择（带随机扰动）。"""
        noise_scale = 0.08

        for i in range(self.cfg.n_households):
            utility = self._cell_utility_for_income(self.agent_income[i], dynamic_rent, job_surface)
            stochastic_utility = utility + self.rng.normal(0.0, noise_scale, size=utility.shape)
            best = np.unravel_index(np.argmax(stochastic_utility), stochastic_utility.shape)
            self.agent_positions[i] = np.array(best)

    def _density_map(self) -> np.ndarray:
        density = np.zeros((self.grid_size, self.grid_size), dtype=float)
        for pos in self.agent_positions:
            density[pos[0], pos[1]] += 1.0
        return density

    def _dynamic_rent(self, density: np.ndarray) -> np.ndarray:
        density_norm = density / np.maximum(density.max(), 1.0)
        # 拥挤导致租金抬升
        return self.base_rent * (1.0 + 0.9 * density_norm)

    def _gini(self, values: np.ndarray) -> float:
        v = np.sort(values)
        n = len(v)
        if n == 0:
            return 0.0
        cumsum = np.cumsum(v)
        return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n

    def _collect_metrics(self, density: np.ndarray, dynamic_rent: np.ndarray, job_surface: np.ndarray, step: int) -> None:
        incomes_with_ubi = self.agent_income + self.cfg.ubi_level
        affordability = np.mean(
            incomes_with_ubi / (dynamic_rent[self.agent_positions[:, 0], self.agent_positions[:, 1]] + 1e-3)
        )

        # 空间不平等：高收入与低收入群体在高机会区的占比差异
        threshold_hi = np.quantile(self.agent_income, 0.8)
        threshold_lo = np.quantile(self.agent_income, 0.2)
        hi_mask = self.agent_income >= threshold_hi
        lo_mask = self.agent_income <= threshold_lo

        job_values = job_surface[self.agent_positions[:, 0], self.agent_positions[:, 1]]
        hi_job_share = np.mean(job_values[hi_mask] > np.quantile(job_surface, 0.7))
        lo_job_share = np.mean(job_values[lo_mask] > np.quantile(job_surface, 0.7))
        spatial_inequality = hi_job_share - lo_job_share

        metric = {
            "step": step,
            "avg_rent": float(dynamic_rent.mean()),
            "avg_density": float(density.mean()),
            "income_gini": float(self._gini(incomes_with_ubi)),
            "affordability_idx": float(affordability),
            "spatial_inequality": float(spatial_inequality),
            "avg_job_opportunity": float(job_surface.mean()),
        }
        self.history.append(metric)

    def run(self) -> pd.DataFrame:
        """执行仿真并返回指标时间序列。"""
        density = self._density_map()

        for step in range(self.cfg.n_steps):
            dynamic_rent = self._dynamic_rent(density)
            job_surface = self._compute_job_surface()
            self._update_positions(dynamic_rent, job_surface)
            density = self._density_map()
            self._collect_metrics(density, dynamic_rent, job_surface, step)

        self.last_density = density
        return pd.DataFrame(self.history)


def run_policy_experiments(seed: int = 42) -> Dict[str, pd.DataFrame]:
    """
    对比三种情景：
    1) 基线（无AI、无UBI）
    2) AI-Agent扩散
    3) AI-Agent + UBI
    """
    scenarios = {
        "baseline": SimulationConfig(seed=seed, ai_coverage=0.0, ubi_level=0.0),
        "ai_only": SimulationConfig(seed=seed, ai_coverage=0.7, ubi_level=0.0),
        "ai_ubi": SimulationConfig(seed=seed, ai_coverage=0.7, ubi_level=25.0),
    }

    outputs: Dict[str, pd.DataFrame] = {}
    for name, cfg in scenarios.items():
        sim = UrbanEvolutionSimulator(cfg)
        outputs[name] = sim.run()
    return outputs


def plot_metric_comparison(results: Dict[str, pd.DataFrame], metric: str) -> None:
    """绘制不同政策情景下的指标对比曲线。"""
    plt.figure(figsize=(8, 4))
    for name, df in results.items():
        plt.plot(df["step"], df[metric], label=name)
    plt.title(f"Policy comparison: {metric}")
    plt.xlabel("Step")
    plt.ylabel(metric)
    plt.legend()
    plt.tight_layout()
    plt.show()


def visualize_final_density(config: SimulationConfig) -> None:
    """展示指定配置下的最终人口密度热力图。"""
    sim = UrbanEvolutionSimulator(config)
    sim.run()

    if sim.last_density is None:
        raise RuntimeError("Simulation density is empty. Run simulation first.")

    plt.figure(figsize=(6, 5))
    plt.imshow(sim.last_density.T, origin="lower", cmap="magma")
    plt.colorbar(label="Household density")
    plt.title(f"Final density | AI={config.ai_coverage}, UBI={config.ubi_level}")
    plt.tight_layout()
    plt.show()


def validate_model(results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    模拟验证（stylized facts）:
    - AI 提升平均就业机会
    - UBI 提升可负担性指数
    - AI+UBI 通常降低空间不平等
    """
    summary = []
    for name, df in results.items():
        last = df.iloc[-1]
        summary.append(
            {
                "scenario": name,
                "final_avg_job_opportunity": last["avg_job_opportunity"],
                "final_affordability_idx": last["affordability_idx"],
                "final_spatial_inequality": last["spatial_inequality"],
            }
        )

    out = pd.DataFrame(summary)
    return out.sort_values("scenario").reset_index(drop=True)


if __name__ == "__main__":
    policy_results = run_policy_experiments(seed=42)
    print("=== Validation summary ===")
    print(validate_model(policy_results))

    plot_metric_comparison(policy_results, "affordability_idx")
    plot_metric_comparison(policy_results, "spatial_inequality")

    visualize_final_density(SimulationConfig(seed=42, ai_coverage=0.7, ubi_level=25.0))
