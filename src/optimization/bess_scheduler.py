"""
CE 295 - Data Science for Energy
Phase 3: BESS Optimization (Linear Programming)
Author: Matthieu & Team

This module implements a deterministic Linear Program using CVXPY to optimize
the charge and discharge schedule of a Battery Energy Storage System (BESS).
It uses forecasted solar generation to minimize Time-of-Use (TOU) grid costs.
"""

import cvxpy as cp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Tuple

class BESSOptimizer:
    """
    Linear Programming Optimizer for Battery Energy Storage Systems.
    
    Attributes:
        capacity (float): Total battery capacity in kWh.
        p_max (float): Maximum charge/discharge power in kW.
        eta_ch (float): Charging efficiency (0 to 1).
        eta_dis (float): Discharging efficiency (0 to 1).
        soc_min (float): Minimum State of Charge bounds (kWh).
        soc_max (float): Maximum State of Charge bounds (kWh).
    """

    def __init__(self, config: Dict[str, float]):
        """
        Initializes the BESS optimizer with hardware specifications.
        
        Args:
            config: Dictionary containing battery parameters.
        """
        self.capacity = config.get('capacity', 100.0)
        self.p_max = config.get('p_max', 50.0)
        self.eta_ch = config.get('eta_ch', 0.95)
        self.eta_dis = config.get('eta_dis', 0.95)
        
        # SoC limits derived from percentages
        self.soc_min = config.get('soc_min_pct', 0.20) * self.capacity
        self.soc_max = config.get('soc_max_pct', 0.90) * self.capacity
        self.soc_initial = config.get('soc_initial_pct', 0.50) * self.capacity

    def solve(self, 
              solar_forecast: np.ndarray, 
              load_profile: np.ndarray, 
              tou_prices: np.ndarray) -> Tuple[pd.DataFrame, float]:
        """
        Solves the LP for a given 24-hour horizon.
        
        Args:
            solar_forecast: 1D array of predicted solar power (kW).
            load_profile: 1D array of site electrical load (kW).
            tou_prices: 1D array of grid electricity prices ($/kWh).
            
        Returns:
            Tuple containing:
                - DataFrame with the optimal schedule (P_grid, P_ch, P_dis, SoC).
                - Total grid cost ($) for the optimized horizon.
        """
        T = len(solar_forecast)
        dt = 1.0  # 1 hour time step
        
        # 1. Decision Variables
        # T+1 for SoC to handle initial state cleanly
        soc = cp.Variable(T + 1, name="SoC")
        p_grid = cp.Variable(T, name="P_grid")
        p_ch = cp.Variable(T, name="P_ch")
        p_dis = cp.Variable(T, name="P_dis")
        p_curtail = cp.Variable(T, name="P_curtail")
        
        # 2. Constraints List
        constraints = []
        
        # Initial State
        constraints.append(soc[0] == self.soc_initial)
        
        # Vectorized Constraints (apply to all t)
        constraints.extend([
            p_grid >= 0,
            p_ch >= 0,
            p_dis >= 0,
            p_curtail >= 0,
            p_curtail <= solar_forecast,
            p_ch <= self.p_max,
            p_dis <= self.p_max,
            soc >= self.soc_min,
            soc <= self.soc_max,
            
            # Power Balance Constraint (with solar curtailment)
            p_grid + (solar_forecast - p_curtail) + p_dis == load_profile + p_ch
        ])
        
        # SoC Dynamics Constraint (requires temporal loop due to state transition)
        for t in range(T):
            constraints.append(
                soc[t+1] == soc[t] + (self.eta_ch * p_ch[t] - (1 / self.eta_dis) * p_dis[t]) * dt
            )
            
        # Cyclic Boundary Constraint
        constraints.append(soc[T] >= self.soc_initial)
        
        # 3. Objective Function
        # Minimize total cost: sum(P_grid[t] * Price[t] * dt)
        objective = cp.Minimize(cp.sum(cp.multiply(tou_prices, p_grid) * dt))
        
        # 4. Solve the LP
        problem = cp.Problem(objective, constraints)
        
        # ECOS is a robust open-source solver for convex problems
        problem.solve(solver=cp.ECOS)
        
        if problem.status not in ["optimal", "optimal_inaccurate"]:
            raise ValueError(f"Optimizer failed to find a solution. Status: {problem.status}")
            
        # 5. Extract Results
        schedule = pd.DataFrame({
            "solar_forecast_kw": solar_forecast,
            "load_kw": load_profile,
            "price_usd_kwh": tou_prices,
            "p_grid_kw": p_grid.value,
            "p_ch_kw": p_ch.value,
            "p_dis_kw": p_dis.value,
            "p_curtail_kw": p_curtail.value,
            "soc_kwh": soc.value[1:] # Discard t=0 to match index size T
        })
        
        total_cost = problem.value
        return schedule, total_cost

# ─────────────────────────────────────────
# MAIN EXECUTION (FOR TESTING)
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("\n=== CE 295 - Phase 3: BESS LP Optimization ===\n")
    
    # Define Battery Specs
    bess_config = {
        'capacity': 200.0,      # kWh
        'p_max': 50.0,          # kW (C-rate = 0.25)
        'eta_ch': 0.95,
        'eta_dis': 0.95,
        'soc_min_pct': 0.20,
        'soc_max_pct': 0.90,
        'soc_initial_pct': 0.50
    }
    
    optimizer = BESSOptimizer(bess_config)
    
    # Generate Synthetic 24h Data (Mocking Phase 2 outputs)
    T = 24
    hours = np.arange(T)
    
    # Typical Californian Duck Curve TOU Prices ($/kWh)
    # Peak from 16:00 to 21:00
    tou_prices = np.where((hours >= 16) & (hours <= 21), 0.45, 0.15)
    
    # Synthetic Load Profile (Base load + evening peak)
    load_profile = 20 + 15 * np.sin(np.pi * (hours - 12) / 12)**2
    
    # Synthetic Solar Forecast (Bell curve centered at solar noon)
    solar_forecast = np.zeros(T)
    daytime = (hours >= 7) & (hours <= 18)
    solar_forecast[daytime] = 60 * np.sin(np.pi * (hours[daytime] - 7) / 11)
    
    try:
        print("Solving Linear Program...")
        schedule_df, optimal_cost = optimizer.solve(solar_forecast, load_profile, tou_prices)
        
        # Calculate baseline cost (Without BESS: Grid must supply Load - Solar)
        net_load_no_bess = np.maximum(0, load_profile - solar_forecast)
        baseline_cost = np.sum(net_load_no_bess * tou_prices)
        
        print(f"\nOptimization Successful!")
        print(f"Cost without BESS: ${baseline_cost:.2f}")
        print(f"Cost with BESS:    ${optimal_cost:.2f}")
        print(f"Daily Savings:     ${baseline_cost - optimal_cost:.2f} ({(baseline_cost - optimal_cost)/baseline_cost*100:.1f}%)\n")
        
        # Plotting the results
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        
        # 1. Power Balance
        axes[0].plot(hours, load_profile, label="Load", color='black', linestyle='--')
        axes[0].plot(hours, solar_forecast, label="Available Solar", color='orange', alpha=0.5)
        axes[0].plot(hours, solar_forecast - schedule_df["p_curtail_kw"], label="Used Solar", color='orange')
        axes[0].plot(hours, schedule_df["p_grid_kw"], label="Grid Import", color='red')
        axes[0].bar(hours, schedule_df["p_curtail_kw"], label="Curtailed Solar", color='grey', alpha=0.3)
        axes[0].set_ylabel("Power (kW)")
        axes[0].set_title("Power Flows")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. Battery Dispatch
        axes[1].bar(hours, schedule_df["p_ch_kw"], label="Charge", color='blue', alpha=0.7)
        axes[1].bar(hours, -schedule_df["p_dis_kw"], label="Discharge", color='green', alpha=0.7)
        axes[1].set_ylabel("Battery Power (kW)")
        axes[1].set_title("BESS Dispatch")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 3. State of Charge & Prices
        ax3 = axes[2]
        ax3_price = ax3.twinx()
        ax3.plot(hours, schedule_df["soc_kwh"], label="SoC", color='purple', linewidth=2)
        ax3.axhline(bess_config['capacity'] * bess_config['soc_max_pct'], color='purple', linestyle=':', label="SoC Max")
        ax3.axhline(bess_config['capacity'] * bess_config['soc_min_pct'], color='purple', linestyle=':', label="SoC Min")
        ax3_price.step(hours, tou_prices, where='mid', color='grey', alpha=0.5, label="TOU Price")
        
        ax3.set_xlabel("Hour of Day")
        ax3.set_ylabel("SoC (kWh)")
        ax3_price.set_ylabel("Price ($/kWh)")
        ax3.set_title("State of Charge vs. TOU Price")
        
        # Combine legends for ax3 and ax3_price
        lines, labels = ax3.get_legend_handles_labels()
        lines2, labels2 = ax3_price.get_legend_handles_labels()
        ax3.legend(lines + lines2, labels + labels2, loc='lower left')
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Optimization Error: {e}")