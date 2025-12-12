import numpy as np

# -----------------------------
# ODE solvers
# -----------------------------
def ode_freefall_euler(g0, dg_dz, cd_star, H, dt):
    """
    Euler method for free-fall ODE.
    Returns arrays of time, position, velocity until z = H.
    """
    t, z, v = [0.0], [0.0], [0.0]
    while z[-1] < H:
        z_curr, v_curr = z[-1], v[-1]
        dzdt = v_curr
        dvdt = dg_dz * z_curr - cd_star * v_curr + g0

        z_next = z_curr + dzdt * dt
        v_next = v_curr + dvdt * dt
        t_next = t[-1] + dt

        # Final step adjustment to hit z = H exactly
        if z_next > H:
            frac = (H - z_curr) / (z_next - z_curr)
            dt_adj = dt * frac
            z_next = H
            v_next = v_curr + dvdt * dt_adj
            t_next = t[-1] + dt_adj

        z.append(z_next)
        v.append(v_next)
        t.append(t_next)

    return np.array(t), np.array(z), np.array(v)


def ode_freefall_rk4(g0, dg_dz, cd_star, H, dt):
    """
    RK4 method for free-fall ODE.
    Returns arrays of time, position, velocity until z = H.
    """
    t, z, v = [0.0], [0.0], [0.0]

    def f(z, v):
        dzdt = v
        dvdt = dg_dz * z - cd_star * v + g0
        return dzdt, dvdt

    while z[-1] < H:
        z_curr, v_curr = z[-1], v[-1]

        k1z, k1v = f(z_curr, v_curr)
        k2z, k2v = f(z_curr + 0.5*dt*k1z, v_curr + 0.5*dt*k1v)
        k3z, k3v = f(z_curr + 0.5*dt*k2z, v_curr + 0.5*dt*k2v)
        k4z, k4v = f(z_curr + dt*k3z, v_curr + dt*k3v)

        dzdt = (k1z + 2*k2z + 2*k3z + k4z) / 6.0
        dvdt = (k1v + 2*k2v + 2*k3v + k4v) / 6.0

        z_next = z_curr + dzdt * dt
        v_next = v_curr + dvdt * dt
        t_next = t[-1] + dt

        # Final step adjustment to hit z = H exactly
        if z_next > H:
            frac = (H - z_curr) / (z_next - z_curr)
            dt_adj = dt * frac

            k1z, k1v = f(z_curr, v_curr)
            k2z, k2v = f(z_curr + 0.5*dt_adj*k1z, v_curr + 0.5*dt_adj*k1v)
            k3z, k3v = f(z_curr + 0.5*dt_adj*k2z, v_curr + 0.5*dt_adj*k2v)
            k4z, k4v = f(z_curr + dt_adj*k3z, v_curr + dt_adj*k3v)

            dzdt = (k1z + 2*k2z + 2*k3z + k4z) / 6.0
            dvdt = (k1v + 2*k2v + 2*k3v + k4v) / 6.0

            z_next = H
            v_next = v_curr + dvdt * dt_adj
            t_next = t[-1] + dt_adj

        z.append(z_next)
        v.append(v_next)
        t.append(t_next)

    return np.array(t), np.array(z), np.array(v)


# -----------------------------
# Utilities: drop time, error, sensitivity
# -----------------------------
def drop_time(g0, dg_dz, cd_star, H, dt, method='rk4'):
    """
    Compute total drop time t* for given parameters and method.
    """
    if method.lower() == 'euler':
        t, z, v = ode_freefall_euler(g0, dg_dz, cd_star, H, dt)
    else:
        t, z, v = ode_freefall_rk4(g0, dg_dz, cd_star, H, dt)
    return t[-1]


def sweep_drop_times(g0, dg_dz, cd_star, Hs, dts, method='rk4'):
    """
    Compute drop times for multiple heights and time steps.
    Returns dict: H -> array of drop times for each dt.
    """
    times = {H: np.zeros_like(dts, dtype=float) for H in Hs}
    for i, dt in enumerate(dts):
        for H in Hs:
            times[H][i] = drop_time(g0, dg_dz, cd_star, H, dt, method)
    return times


def relative_errors(times_dict):
    """
    Compute relative errors compared to smallest dt solution.
    Returns dict: H -> array of relative errors.
    """
    errors = {}
    for H, arr in times_dict.items():
        ref = arr[0]  # smallest dt reference
        errors[H] = np.abs(arr - ref) / ref
    return errors


def sensitivity(g0, dg_dz, cd_star, Hs, dt, alpha=0.01, method='rk4'):
    """
    Sensitivity analysis: perturb each parameter by alpha (fractional change).
    Returns dict: H -> {'baseline': t0, 'dg0': Δt*, 'dgdz': Δt*, 'cd': Δt*}
    """
    out = {}
    for H in Hs:
        t0 = drop_time(g0, dg_dz, cd_star, H, dt, method)
        dt_g0 = drop_time(g0*(1+alpha), dg_dz, cd_star, H, dt, method) - t0
        dt_dg = drop_time(g0, dg_dz*(1+alpha), cd_star, H, dt, method) - t0
        dt_cd = drop_time(g0, dg_dz, cd_star*(1+alpha), H, dt, method) - t0
        out[H] = {'baseline': t0, 'dg0': dt_g0, 'dgdz': dt_dg, 'cd': dt_cd}
    return out
