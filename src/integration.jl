using LinearAlgebra
using SparseArrays
using Statistics

export simulation



"""
simulation(u0, x, dt, T, f; F=0, save_every=1)

Performs a time-stepping simulation using the 4th-order Runge-Kutta method (RK4) to solve a system of differential equations. The simulation progresses from an initial state `u0` to a final time `T`, with optional forcing and data saving at regular intervals.

# Arguments
- `u0`: Initial condition vector of the system.
- `x`: Spatial grid or other problem-specific parameters.
- `dt`: Time step size for the simulation.
- `T`: Final time for the simulation.
- `f`: Function representing the system's dynamics (du/dt = f(u, x, t)).
    - The function `f` should accept three arguments: the state `u`, spatial variable `x`, and time `t`.

# Keyword Arguments
- `F=0`: Optional forcing term that will be added to the function `f` during the simulation.
- `save_every=1`: Integer specifying how frequently to save data points. A value of 1 saves every time step.

# Returns
- `us`: A matrix where each column contains the state vector `u` at each saved time step.
- `dus`: A matrix where each column contains the rate of change of the state `du/dt` (from `f(u,x,t)`) at each saved time step.
- `ts`: A vector containing the time points corresponding to the saved states.

# Example Usage
```julia
u0 = [1.0, 0.0]   # Initial condition
x = [0.0, 1.0]    # Spatial grid
dt = 0.01         # Time step size
T = 1.0           # Final time
f = (u, x, t) -> u # A simple dynamic system

us, dus, ts = simulation(u0, x, dt, T, f; F=0.1, save_every=10)
"""
function simulation(u0,x,dt,T,f;F = 0,save_every= 1)

    f_plus_forcing(u,x,t) = f(u,x,t) .+ F

    t= 0.
    u = u0

    save_counter = save_every + 1
    counter = 0
    round_t = t
    ############
    while round_t <= T
        if save_counter > save_every
            #############################################
            save_counter = 1
            counter += 1
        end
        t += dt
        save_counter += 1
        round_t = round(t,digits =10)
    end
    t = 0.

    us = zeros(size(u0)[1], counter)
    dus = zeros(size(u0)[1], counter)
    ts = zeros(1,counter)
    ############
    save_counter = save_every + 1
    counter = 0
    round_t = t
    while round_t <= T
        du = RK4(u,x,t,dt,f_plus_forcing)
        if save_counter > save_every
            #############################################
            counter += 1
            us[:,counter] += u

            dus[:,counter] += f(u,x,t)

            ts[1,counter] += t
            #############################################
            save_counter = 1


        end
        u = u .+ dt*du

        t += dt
        save_counter += 1
        round_t = round(t,digits =10)
    end
    return us,dus,ts
end

"""
    RK4(u, x, t, dt, f)

Performs one time step of the 4th-order Runge-Kutta (RK4) integration method to solve an ordinary differential equation (ODE) of the form du/dt = f(u, x, t).

# Arguments
- `u`: The current state vector at time `t`.
- `x`: Spatial grid points as a vector.
- `t`: Current time value.
- `dt`: Time step size.
- `f`: A function representing the system's dynamics (du/dt = f(u, x, t)).
    - The function `f` should accept three arguments: the state `u`, spatial variable `x`, and time `t`.

# Returns
- The updated state vector after one time step using the RK4 method.

# Example Usage
```julia
u0 = [1.0, 0.0]   # Initial state
x = [0.0, 1.0]    # Spatial grid
t = 0.0           # Initial time
dt = 0.01         # Time step
f = (u, x, t) -> u # A simple dynamic system

u_next = RK4(u0, x, t, dt, f)
"""
function RK4(u,x,t,dt,f)
    k1 = f(u,x,t)
    k2 = f(u.+dt*k1/2,x,t+dt/2)
    k3 = f(u.+dt*k2/2,x,t+dt/2)
    k4 = f(u .+ dt*k3,x,t+dt)
    return 1/6*(k1.+2*k2.+2*k3.+k4)
end
