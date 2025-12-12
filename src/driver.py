import numpy as np
import matplotlib.pyplot as plt
import time

# Import utilities from function.py (make sure the file is named function.py)
from function import (
    ode_freefall_euler,
    ode_freefall_rk4,
    drop_time,
    sweep_drop_times,
    relative_errors,
    sensitivity,
)

# -----------------------------
# Parameters
# -----------------------------
g0 = 9.811636       # m/s^2
dg_dz = 3.086e-6    # (m/s^2)/m
cd_star = 1.87e-4   # 1/s
Hs = [10.0, 20.0, 40.0]

# Choose Δt grid (ensure smallest first for reference error)
dts = np.linspace(0.001, 0.08, 20)

# -----------------------------
# 1) Drop time vs Δt and relative error
# -----------------------------
times_euler = sweep_drop_times(g0, dg_dz, cd_star, Hs, dts, method='euler')
times_rk4   = sweep_drop_times(g0, dg_dz, cd_star, Hs, dts, method='rk4')

errs_euler = relative_errors(times_euler)
errs_rk4   = relative_errors(times_rk4)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for H in Hs:
    axes[0].plot(dts, times_euler[H], 'o-', label=f'Euler H={H} m')
    axes[0].plot(dts, times_rk4[H],   's-', label=f'RK4 H={H} m')
axes[0].set_xlabel('Δt (s)')
axes[0].set_ylabel('Drop time t* (s)')
axes[0].set_title('Drop time vs Δt')
axes[0].legend()
axes[0].grid(True)

for H in Hs:
    axes[1].plot(dts, errs_euler[H], 'o-', label=f'Euler H={H} m')
    axes[1].plot(dts, errs_rk4[H],   's-', label=f'RK4 H={H} m')
axes[1].set_xlabel('Δt (s)')
axes[1].set_ylabel('Relative error')
axes[1].set_title('Relative error vs Δt')
axes[1].set_yscale('log')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()

# -----------------------------
# 2) Simulation runtime vs Δt
# -----------------------------
times_runtime_euler = {H: [] for H in Hs}
times_runtime_rk4   = {H: [] for H in Hs}

for dt in dts:
    for H in Hs:
        start = time.perf_counter()
        ode_freefall_euler(g0, dg_dz, cd_star, H, dt)
        end = time.perf_counter()
        times_runtime_euler[H].append(end - start)

        start = time.perf_counter()
        ode_freefall_rk4(g0, dg_dz, cd_star, H, dt)
        end = time.perf_counter()
        times_runtime_rk4[H].append(end - start)

plt.figure(figsize=(10, 6))
for H in Hs:
    plt.plot(dts, times_runtime_euler[H], 'o-', label=f'Euler H={H} m')
    plt.plot(dts, times_runtime_rk4[H],   's-', label=f'RK4 H={H} m')
plt.xlabel('Δt (s)')
plt.ylabel('Simulation time (s)')
plt.title('Simulation runtime vs Δt')
plt.legend()
plt.grid(True)
plt.show()

# -----------------------------
# 3) Sensitivity analysis (Δt* for +1% parameter changes)
# -----------------------------
alpha = 0.01
dt_sens = 0.002  # small dt for accuracy in sensitivity

sens_euler = sensitivity(g0, dg_dz, cd_star, Hs, dt_sens, alpha=alpha, method='euler')
sens_rk4   = sensitivity(g0, dg_dz, cd_star, Hs, dt_sens, alpha=alpha, method='rk4')

def print_sens(title, sens):
    print(title)
    for H in Hs:
        s = sens[H]
        print(f"  H={H} m: baseline t* = {s['baseline']:.6f} s")
        print(f"    Δt* (+1% g0)  = {s['dg0']*1000:.3f} ms")
        print(f"    Δt* (+1% g')  = {s['dgdz']*1000:.3f} ms")
        print(f"    Δt* (+1% cD*) = {s['cd']*1000:.3f} ms")
    print()

print_sens("Sensitivity (Euler, α=1%)", sens_euler)
print_sens("Sensitivity (RK4, α=1%)", sens_rk4)

# Optional: estimate % change needed for ~10 ms shift using linear scaling
target_ms = 10.0
def percent_for_10ms(delta_ms):
    if abs(delta_ms) < 1e-12:
        return np.inf
    return target_ms / abs(delta_ms)  # since delta_ms corresponds to +1% change

print("Estimated % change needed for ~10 ms shift (RK4, α=1% baseline):")
for H in Hs:
    s = sens_rk4[H]
    pct_g0 = percent_for_10ms(s['dg0']*1000)
    pct_dg = percent_for_10ms(s['dgdz']*1000)
    pct_cd = percent_for_10ms(s['cd']*1000)
    print(f"  H={H} m: g0 ≈ {pct_g0:.2f}% | g' ≈ {pct_dg:.2f}% | cD* ≈ {pct_cd:.2f}%")
