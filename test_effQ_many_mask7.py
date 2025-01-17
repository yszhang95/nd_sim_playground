import torch
import numpy as np
import time
import tracemalloc

import torch.utils.benchmark as benchmark

import matplotlib.pyplot as plt
def gauss_conv_line_3d_yz2(Q, X0, X1, Sigma, x, y, z):
    """
    Simplified version of GaussConvLine3D to calculate charge distribution
    for a line segment in 3D space.
    """
    sqrt2 = 1.4142135623730951

   #  Q = Q.to(device, dtype=torch.float32)
   #  X0 = X0.to(device, dtype=torch.float32)
   #  X1 = X1.to(device, dtype=torch.float32)
   #  Sigma = Sigma.to(device, dtype=torch.float32)
   #  x = x.to(device, dtype=torch.float32)
   #  y = y.to(device, dtype=torch.float32)
   #  z = z.to(device, dtype=torch.float32)

   #  # Ensure correct shapes
   #  if len(X0.shape) != 2 or X0.shape[1] != 3:
   #      raise ValueError(f'X0 shape must be (N, 3), got {X0.shape}')
   #  if len(X1.shape) != 2 or X1.shape[1] != 3:
   #      raise ValueError(f'X1 shape must be (N, 3), got {X1.shape}')
   #  if len(Sigma.shape) != 2 or Sigma.shape[1] != 3:
   #      raise ValueError(f'Sigma shape must be (N, 3), got {Sigma.shape}')

    # Pre-allocate tensors for intermediate calculations
    batch_size = X0.size(0)

    if len(x.shape) != 4:
        raise ValueError('x.shape in (batch, nx, 1, 1,)')

    # Prepare for broadcasting
    x0, y0, z0 = [X0[:,i].view(-1, 1, 1, 1) for i in range(3)]
    x1, y1, z1 = [X1[:,i].view(-1, 1, 1, 1) for i in range(3)]
    sx, sy, sz = [Sigma[:,i].view(-1, 1, 1, 1) for i in range(3)]
    Q = Q.view(-1, 1, 1, 1)

    # Calculate differences
    dx01 = x0 - x1
    dy01 = y0 - y1
    dz01 = z0 - z1

    # Calculate squared terms
    sxsy2 = (sx*sy)**2
    sxsz2 = (sx*sz)**2
    sysz2 = (sy*sz)**2
    sx2 = sx**2
    sy2 = sy**2
    sz2 = sz**2

    # Calculate delta terms
    deltaSquare = (
        sysz2 * dx01**2 +
        sxsy2 * dz01**2 +
        sxsz2 * dy01**2
    )
    deltaSquareSqrt = torch.sqrt(deltaSquare)

    # Calculate charge distribution
    QoverDeltaSquareSqrt4pi = Q / (deltaSquareSqrt * 4 * np.pi)
    erfArgDenominator = sqrt2 * deltaSquareSqrt * sx * sy * sz

    # Pre-allocate the final result tensor


    # Run 10 times and accumulate results

    charge = ((-QoverDeltaSquareSqrt4pi * torch.exp(
        -sy2 * torch.pow(x * dz01 + (z1*x0 - z0*x1) - z * dx01, 2) *0.5/deltaSquare ))
              * torch.exp(
        -sx2 * torch.pow(y * dz01 + (z1*y0 - z0*y1) - z * dy01, 2) *0.5/deltaSquare)
              * torch.exp(
        -sz2 * torch.pow(y * dx01 + (x1*y0 - x0*y1) - x * dy01, 2) *0.5/deltaSquare)) * (
        torch.erf((
            sysz2 * (x - x0) * dx01 +
            sxsy2 * (z - z0) * dz01 +
            sxsz2 * (y - y0) * dy01
        )/erfArgDenominator) -
        torch.erf((
            sysz2 * (x - x1) * dx01 +
            sxsy2 * (z - z1) * dz01 +
            sxsz2 * (y - y1) * dy01
        )/erfArgDenominator)
    )
    return charge

def gauss_conv_line_3d_yz(Q, X0, X1, Sigma, x, y, z, device='cuda'):
    """
    Simplified version of GaussConvLine3D to calculate charge distribution
    for a line segment in 3D space.
    """
    sqrt2 = np.sqrt(2)

    Q = Q.to(device, dtype=torch.float32)
    X0 = X0.to(device, dtype=torch.float32)
    X1 = X1.to(device, dtype=torch.float32)
    Sigma = Sigma.to(device, dtype=torch.float32)
    x = x.to(device, dtype=torch.float32)
    y = y.to(device, dtype=torch.float32)
    z = z.to(device, dtype=torch.float32)

    # Ensure correct shapes
    if len(X0.shape) != 2 or X0.shape[1] != 3:
        raise ValueError(f'X0 shape must be (N, 3), got {X0.shape}')
    if len(X1.shape) != 2 or X1.shape[1] != 3:
        raise ValueError(f'X1 shape must be (N, 3), got {X1.shape}')
    if len(Sigma.shape) != 2 or Sigma.shape[1] != 3:
        raise ValueError(f'Sigma shape must be (N, 3), got {Sigma.shape}')

    # Pre-allocate tensors for intermediate calculations
    batch_size = X0.size(0)

    # Prepare for broadcasting
    x0, y0, z0 = [X0[:,i].view(-1, 1, 1, 1) for i in range(3)]
    x1, y1, z1 = [X1[:,i].view(-1, 1, 1, 1) for i in range(3)]
    sx, sy, sz = [Sigma[:,i].view(-1, 1, 1, 1) for i in range(3)]
    Q = Q.view(-1, 1, 1, 1)

    # Calculate differences
    dx01 = x0 - x1
    dy01 = y0 - y1
    dz01 = z0 - z1

    # Calculate squared terms
    sxsy2 = (sx*sy)**2
    sxsz2 = (sx*sz)**2
    sysz2 = (sy*sz)**2
    sx2 = sx**2
    sy2 = sy**2
    sz2 = sz**2

    # Calculate delta terms
    deltaSquare = (
        sysz2 * dx01**2 +
        sxsy2 * dz01**2 +
        sxsz2 * dy01**2
    )
    deltaSquareSqrt = torch.sqrt(deltaSquare)

    # Calculate charge distribution
    QoverDeltaSquareSqrt4pi = Q / (deltaSquareSqrt * 4 * np.pi)
    erfArgDenominator = sqrt2 * deltaSquareSqrt * sx * sy * sz

    # Pre-allocate the final result tensor


    # Run 10 times and accumulate results
    charge = _compute_charge_component(
        QoverDeltaSquareSqrt4pi, erfArgDenominator,
        x, y, z, x0, y0, z0, x1, y1, z1,
        sx2, sy2, sz2, sxsy2, sxsz2, sysz2,
        dx01, dy01, dz01, deltaSquare
    )
    return charge


@torch.jit.script
def _compute_charge_component(QoverDelta, erfArgDenom,
                            xpos, ypos, zpos, x0, y0, z0, x1, y1, z1,
                            sx2, sy2, sz2, sxsy2, sxsz2, sysz2,
                            dx01, dy01, dz01, deltaSquare):
    """
    JIT-compiled helper function to compute the final charge values
    """
    charge_vals = ((-QoverDelta * torch.exp(
        -sy2 * torch.pow(xpos * dz01 + (z1*x0 - z0*x1) - zpos * dx01, 2) *0.5/deltaSquare ))
              * torch.exp(
        -sx2 * torch.pow(ypos * dz01 + (z1*y0 - z0*y1) - zpos * dy01, 2) *0.5/deltaSquare)
              * torch.exp(
        -sz2 * torch.pow(ypos * dx01 + (x1*y0 - x0*y1) - xpos * dy01, 2) *0.5/deltaSquare)) * (
        torch.erf((
            sysz2 * (xpos - x0) * dx01 +
            sxsy2 * (zpos - z0) * dz01 +
            sxsz2 * (ypos - y0) * dy01
        )/erfArgDenom) -
        torch.erf((
            sysz2 * (xpos - x1) * dx01 +
            sxsy2 * (zpos - z1) * dz01 +
            sxsz2 * (ypos - y1) * dy01
        )/erfArgDenom)
    )

    return charge_vals

def create_grid_3d(origin, spacing, shape, device='cuda'):
    """
    Create a 3D grid for charge calculation.
    """
    # Create 1D tensors first
    x = torch.arange(origin[0], origin[0] + shape[0] * spacing[0], spacing[0],
                    device=device, dtype=torch.float32)
    y = torch.arange(origin[1], origin[1] + shape[1] * spacing[1], spacing[1],
                    device=device, dtype=torch.float32)
    z = torch.arange(origin[2], origin[2] + shape[2] * spacing[2], spacing[2],
                    device=device, dtype=torch.float32)

    # Create meshgrid
    x, y, z = torch.meshgrid(x, y, z, indexing='ij')
    return x, y, z

def create_grid_1d(origin, spacing, shape, device='cuda'):
    """
    Create a 3D grid for charge calculation.
    """
    # Create 1D tensors first
    x = torch.arange(origin[0], origin[0] + shape[0] * spacing[0], spacing[0],
                    device=device, dtype=torch.float32)
    y = torch.arange(origin[1], origin[1] + shape[1] * spacing[1], spacing[1],
                    device=device, dtype=torch.float32)
    z = torch.arange(origin[2], origin[2] + shape[2] * spacing[2], spacing[2],
                    device=device, dtype=torch.float32)
    return x, y, z

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    torch.cuda.empty_cache()  # Clear any existing allocations

    ndim = 33
    # nevent = 1_600
    nevent = 3_000

    # Define grid parameters
    origin = (0.0, 0.0, 0.0)
    spacing = (0.1, 0.1, 0.1)
    shape = (ndim, ndim, ndim)

    # Create grid on GPU

    x1d, y1d, z1d = create_grid_1d(origin, spacing, shape, device)

    # Define line segment parameters (directly on GPU)
    Q = torch.ones(nevent, device=device)
    X0 = torch.rand(nevent, 3, device=device)
    X1 = torch.rand(nevent, 3, device=device)
    Sigma = torch.full((nevent, 3), 0.2, device=device)

    ns = []
    tyzscript = []
    tyz = []

    for n in range(600, nevent, 200):
        Qi = torch.ones(n, device=device)
        X0i = torch.rand(n, 3, device=device)
        X1i = torch.rand(n, 3, device=device)
        Sigmai = torch.full((n, 3), 0.2, device=device)
        ns.append(n)
        s = slice(0, n)
        with torch.no_grad():
            x1di = x1d.view(1, len(x1d), 1, 1).expand(n, len(x1d), 1, 1).clone()
            y1di = y1d.view(1, 1, len(y1d), 1).expand(n, 1, len(y1d), 1).clone()
            z1di = z1d.view(1, 1, 1, len(z1d)).expand(n, 1, 1, len(z1d)).clone()
            tstats = benchmark.Timer(
                stmt = 'gauss_conv_line_3d_yz(Q, X0, X1, Sigma, x, y, z, device)',
                setup = 'from __main__ import gauss_conv_line_3d_yz',
                globals={'Q' : Qi, 'X0' : X0i, 'X1': X1i, 'Sigma' : Sigmai, 'x' : x1di, 'y' : y1di, 'z' : z1di,
                         'device' : device}
                )
            m = tstats.blocked_autorange(min_run_time=1)
            tyzscript.append(m.mean)
            tstats = benchmark.Timer(
                stmt = 'gauss_conv_line_3d_yz2(Q, X0, X1, Sigma, x, y, z)',
                setup = 'from __main__ import gauss_conv_line_3d_yz2',
                globals={'Q' : Qi, 'X0' : X0i, 'X1': X1i, 'Sigma' : Sigmai, 'x' : x1di, 'y' : y1di, 'z' : z1di, }
                )
            m = tstats.blocked_autorange(min_run_time=1)
            tyz.append(m.mean)
            print(len(Qi), nevent)
            print(len(Qi), nevent)
        q1 = gauss_conv_line_3d_yz(Qi, X0i, X1i, Sigmai, x1di, y1di, z1di)
        q2 = gauss_conv_line_3d_yz2(Qi, X0i, X1i, Sigmai, x1di, y1di, z1di)
        print('output', q1.shape)

        assert q1.allclose(q2, atol=1E-5, rtol=1E-5)

    return ns, tyzscript, tyz


if __name__ == "__main__":
    # GPU Operation

    ns, tyzscript, tyz = main()
    tyzscript = np.array(tyzscript)* 1E3
    tyz = np.array(tyz)* 1E3
    plt.plot(ns, tyzscript, 'ro-', label='w/o mask;x,y,z,broadcast, JIT')
    plt.plot(ns, tyz, 'bo-', label='w/o mask;x,y,z,broadcast, python')
    plt.xlabel('nsteps')
    plt.ylabel('mean of execution time [ms]')
    plt.legend(title='full calculation; output size nsteps x (33,33,33)')
    plt.savefig('profile_masks_opt2_v7.png')
    plt.yscale('log')
    plt.savefig('profile_masks_opt2_log_v7.png')
