"""
impact_model.py

A reduced-order (1-DOF) physics simulator for low-velocity impact on a
pressurized uPVC pipe, inspired by and validated against:

Ahmed & Bashar (2023), "Behavior of Clamped uPVC Pipe with Different
Internal Pressure Subjected to Low Velocity Impact", RUET BSc Thesis.

MODEL:
  A hammer of mass m strikes a pipe modeled as an elastoplastic spring.
  Spring stiffness increases with internal pressure (membrane stiffening,
  i.e. "string action" described in the thesis).

  Equation of motion (hammer displacement delta, penetration into pipe):
      m * delta'' + c * delta' + F_spring(delta, state) = 0

  F_spring is elastoplastic:
      - Elastic loading:      F = K * delta                  (delta < delta_y)
      - Plastic loading:      F = F_y                          (delta increasing beyond delta_y)
      - Elastic unloading:    F = F_y - K * (delta_max - delta)  (delta decreasing from delta_max)
      - Separation:           F = 0 when computed F would go negative (hammer leaves pipe)

  K = K0 * (1 + beta * pressure)   <-- pressure-dependent stiffness (Lesson 2)

This is NOT a replacement for full 3D FEA (ABAQUS). It is a fast,
physically-motivated approximation whose free parameters (K0, beta,
F_y, c) are calibrated against the real experimental data in the
thesis (see calibration.py).
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class PipeImpactParams:
    """Physical + calibration parameters for the impact model."""
    mass: float = 4.0          # hammer mass, kg (thesis: 4 kg)
    velocity0: float = 5.56    # impact velocity, m/s (thesis: constant 5.56 m/s)

    K0: float = 2.0e6          # baseline pipe stiffness at zero pressure, N/m (TO CALIBRATE)
    beta: float = 3.0e-4       # pressure stiffening coefficient, 1/kPa   (TO CALIBRATE)
    Fy0: float = 3200.0        # baseline yield force at zero pressure, N (TO CALIBRATE)
    gamma: float = 1.0e-4      # pressure effect on yield force, 1/kPa    (TO CALIBRATE)
    damping_ratio: float = 0.05  # fraction of critical damping (energy loss), (TO CALIBRATE)

    def pipe_stiffness(self, pressure_kpa: float) -> float:
        """K(p) = K0 * (1 + beta * p)  -- Lesson 2 formula."""
        return self.K0 * (1.0 + self.beta * pressure_kpa)

    def yield_force(self, pressure_kpa: float) -> float:
        """F_yield(p) = Fy0 * (1 + gamma * p) -- Lesson 3: pressure delays
        local yield via bending-to-membrane ('string action') transition."""
        return self.Fy0 * (1.0 + self.gamma * pressure_kpa)


def simulate_impact(pressure_kpa: float, params: PipeImpactParams,
                     t_max: float = 0.06, dt: float = 1e-6):
    """
    Time-step the elastoplastic impact oscillator using explicit
    integration (simple and transparent -- good for learning; we can
    swap in scipy's solve_ivp with events later if needed).

    Returns a dict of time series + summary scalars (max force,
    permanent deformation) that we will compare against Table 4.3
    of the thesis.
    """
    m = params.mass
    K = params.pipe_stiffness(pressure_kpa)
    Fy = params.yield_force(pressure_kpa)
    delta_y = Fy / K  # displacement at first yield

    # critical damping coefficient for this stiffness, then scale by ratio
    c = params.damping_ratio * 2 * np.sqrt(K * m)

    n_steps = int(t_max / dt)
    t = np.zeros(n_steps)
    delta = np.zeros(n_steps)      # penetration (pipe local deformation), m
    vel = np.zeros(n_steps)        # penetration velocity, m/s
    force = np.zeros(n_steps)      # contact/spring force, N

    vel[0] = params.velocity0
    delta_max_reached = 0.0        # tracks plastic history (max penetration so far)
    has_yielded = False            # explicit flag: has the spring ever reached F_yield?
    in_contact = True

    for i in range(1, n_steps):
        t[i] = t[i - 1] * 1.0 + dt

        d_prev = delta[i - 1]
        v_prev = vel[i - 1]

        if not in_contact:
            # hammer has separated from pipe -- free flight, no force
            F = 0.0
        else:
            delta_max_reached = max(delta_max_reached, d_prev)

            if not has_yielded:
                # BEFORE first yield: pure elastic loading, F ramps up from 0
                F = K * d_prev
                if F >= Fy:
                    has_yielded = True
                    F = Fy
            elif v_prev >= 0 and d_prev >= delta_max_reached - 1e-12:
                # AFTER yield, still loading further (pushing deeper than ever before):
                # perfectly plastic -- force caps at F_yield
                F = Fy
            else:
                # AFTER yield, now unloading (elastic unload along slope K from
                # the deepest point reached so far)
                F = Fy - K * (delta_max_reached - d_prev)
                F = max(F, 0.0)  # cannot pull (contact only pushes)
                if F <= 0.0 and v_prev < 0:
                    in_contact = False
                    F = 0.0

        force[i - 1] = F

        # equation of motion: m*a = -F - c*v  (spring + damper resist penetration)
        accel = (-F - c * v_prev) / m
        vel[i] = v_prev + accel * dt
        delta[i] = d_prev + vel[i] * dt

        if not in_contact:
            delta[i] = delta[i - 1]  # no more spring displacement once separated (rigid-body coast)

    force[-1] = force[-2]

    max_force = float(np.max(force))
    # Permanent deformation = the penetration left once the hammer has fully
    # separated and force has returned to zero. If contact never separates
    # within t_max, fall back to the theoretical unload point
    # (delta_max_reached - Fy/K), i.e. where the elastic-unload line hits F=0.
    if not in_contact:
        permanent_deformation = float(delta[-1])
    else:
        permanent_deformation = float(delta_max_reached - Fy / K)
    permanent_deformation = max(permanent_deformation, 0.0)

    return {
        "t": t, "delta": delta, "vel": vel, "force": force,
        "max_force": max_force,
        "permanent_deformation": permanent_deformation,
        "K": K, "delta_y": delta_y,
    }


if __name__ == "__main__":
    # Quick smoke test at the thesis's lowest pressure case
    params = PipeImpactParams()
    result = simulate_impact(pressure_kpa=689.48, params=params)
    print(f"Max contact force:      {result['max_force']:.1f} N   (thesis experimental-numerical: 3527-3776 N)")
    print(f"Permanent deformation:  {result['permanent_deformation']*1000:.2f} mm (thesis numerical: 15.1 mm)")
