"""Navier-Stokes equations for PINN training."""

import torch


def navier_stokes_loss(
    pinn_output: torch.Tensor,
    coords: torch.Tensor,
    rho: float = 1.0,
    mu: float = 0.001,
) -> torch.Tensor:
    """
    Compute the residual of the Navier-Stokes equations.
    pinn_output: [N, 4] containing (u, v, w, p)
    coords: [N, 4] containing (x, y, z, t)
    """
    u = pinn_output[:, 0:1]
    v = pinn_output[:, 1:2]
    w = pinn_output[:, 2:3]
    p = pinn_output[:, 3:4]

    # Helper for gradients
    def grad(q, r):
        # Check if q is differentiable
        if not q.requires_grad:
            return torch.zeros_like(r)
            
        g = torch.autograd.grad(
            q, r, grad_outputs=torch.ones_like(q), create_graph=True, retain_graph=True, allow_unused=True
        )[0]
        if g is None:
            return torch.zeros_like(r)
        return g

    u_g = grad(u, coords)
    u_x, u_y, u_z, u_t = u_g[:, 0:1], u_g[:, 1:2], u_g[:, 2:3], u_g[:, 3:4]
    
    v_g = grad(v, coords)
    v_x, v_y, v_z, v_t = v_g[:, 0:1], v_g[:, 1:2], v_g[:, 2:3], v_g[:, 3:4]
    
    w_g = grad(w, coords)
    w_x, w_y, w_z, w_t = w_g[:, 0:1], w_g[:, 1:2], w_g[:, 2:3], w_g[:, 3:4]
    
    p_g = grad(p, coords)
    p_x, p_y, p_z = p_g[:, 0:1], p_g[:, 1:2], p_g[:, 2:3]

    # Second derivatives
    u_xx = grad(u_x, coords)[:, 0:1]
    u_yy = grad(u_y, coords)[:, 1:2]
    u_zz = grad(u_z, coords)[:, 2:3]

    v_xx = grad(v_x, coords)[:, 0:1]
    v_yy = grad(v_y, coords)[:, 1:2]
    v_zz = grad(v_z, coords)[:, 2:3]

    w_xx = grad(w_x, coords)[:, 0:1]
    w_yy = grad(w_y, coords)[:, 1:2]
    w_zz = grad(w_z, coords)[:, 2:3]

    # Continuity equation: div(V) = 0
    continuity = u_x + v_y + w_z

    # Momentum equations (Navier-Stokes)
    f_u = rho * (u_t + u * u_x + v * u_y + w * u_z) + p_x - mu * (u_xx + u_yy + u_zz)
    f_v = rho * (v_t + u * v_x + v * v_y + w * v_z) + p_y - mu * (v_xx + v_yy + v_zz)
    f_w = rho * (w_t + u * w_x + v * w_y + w * w_z) + p_z - mu * (w_xx + w_yy + w_zz)

    return torch.mean(continuity**2 + f_u**2 + f_v**2 + f_w**2)
