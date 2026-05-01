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

    x = coords[:, 0:1]
    y = coords[:, 1:2]
    z = coords[:, 2:3]
    t = coords[:, 3:4]

    # Gradients
    def grad(q, r):
        return torch.autograd.grad(
            q, r, grad_outputs=torch.ones_like(q), create_graph=True, retain_graph=True, allow_unused=True
        )[0] or torch.zeros_like(r)

    u_x = grad(u, x)
    u_y = grad(u, y)
    u_z = grad(u, z)
    u_t = grad(u, t)

    v_x = grad(v, x)
    v_y = grad(v, y)
    v_z = grad(v, z)
    v_t = grad(v, t)

    w_x = grad(w, x)
    w_y = grad(w, y)
    w_z = grad(w, z)
    w_t = grad(w, t)

    p_x = grad(p, x)
    p_y = grad(p, y)
    p_z = grad(p, z)

    # Second derivatives
    u_xx = grad(u_x, x)
    u_yy = grad(u_y, y)
    u_zz = grad(u_z, z)

    v_xx = grad(v_x, x)
    v_yy = grad(v_y, y)
    v_zz = grad(v_z, z)

    w_xx = grad(w_x, x)
    w_yy = grad(w_y, y)
    w_zz = grad(w_z, z)

    # Continuity equation: div(V) = 0
    continuity = u_x + v_y + w_z

    # Momentum equations (Navier-Stokes)
    # rho(V_t + V.grad(V)) = -grad(p) + mu * laplacian(V)
    f_u = rho * (u_t + u * u_x + v * u_y + w * u_z) + p_x - mu * (u_xx + u_yy + u_zz)
    f_v = rho * (v_t + u * v_x + v * v_y + w * v_z) + p_y - mu * (v_xx + v_yy + v_zz)
    f_w = rho * (w_t + u * w_x + v * w_y + w * w_z) + p_z - mu * (w_xx + w_yy + w_zz)

    return torch.mean(continuity**2 + f_u**2 + f_v**2 + f_w**2)
