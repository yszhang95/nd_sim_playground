import torch
import numpy as np
import time
import tracemalloc

import torch.utils.benchmark as benchmark

import matplotlib.pyplot as plt

def gauss_conv_line_3d_orig(Q, X0, X1, Sigma, x, y, z, device='cuda'):
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
    result_shape = x.size()
    grid_size = result_shape[1:]
    charge = torch.zeros(result_shape, device=device, dtype=torch.float32)

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

    charge += -QoverDeltaSquareSqrt4pi * torch.exp(-0.5 * (
        sy2 * torch.pow(x * dz01 + (z1*x0 - z0*x1) - z * dx01, 2) +
        sx2 * torch.pow(y * dz01 + (z1*y0 - z0*y1) - z * dy01, 2) +
        sz2 * torch.pow(y * dx01 + (x1*y0 - x0*y1) - x * dy01, 2)
    )/deltaSquare) * (
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

def gauss_conv_line_3d_mask_optimized(Q, X0, X1, Sigma, x, y, z, mask, device='cuda'):
    """
    Optimized version of gauss_conv_line_3d_mask that avoids unnecessary tensor expansions
    """
    sqrt2 = np.sqrt(2)

    # Move tensors to device and ensure correct dtype
    Q = Q.to(device, dtype=torch.float32)
    X0 = X0.to(device, dtype=torch.float32)
    X1 = X1.to(device, dtype=torch.float32)
    Sigma = Sigma.to(device, dtype=torch.float32)
    mask = mask.to(device)

    # Shape validations
    if len(X0.shape) != 2 or X0.shape[1] != 3:
        raise ValueError(f'X0 shape must be (N, 3), got {X0.shape}')
    if len(X1.shape) != 2 or X1.shape[1] != 3:
        raise ValueError(f'X1 shape must be (N, 3), got {X1.shape}')
    if len(Sigma.shape) != 2 or Sigma.shape[1] != 3:
        raise ValueError(f'Sigma shape must be (N, 3), got {Sigma.shape}')

    result_shape = x.size()
    charge = torch.zeros(result_shape, device=device, dtype=torch.float32)
    
    # Get masked indices only once
    b_indices, x_indices, y_indices, z_indices = torch.where(mask)
    
    if len(b_indices) == 0:
        return charge

    # Extract coordinates only for masked points
    xpos = x[b_indices, x_indices, y_indices, z_indices]
    ypos = y[b_indices, x_indices, y_indices, z_indices]
    zpos = z[b_indices, x_indices, y_indices, z_indices]

    # Get batch indices for parameter lookups
    batch_idx = b_indices

    # Extract parameters directly for the relevant batch indices
    Q_masked = Q[batch_idx]
    x0, y0, z0 = X0[batch_idx].T  # Transpose to get separate coordinates
    x1, y1, z1 = X1[batch_idx].T
    sx, sy, sz = Sigma[batch_idx].T

    # Calculate differences
    dx01 = x0 - x1
    dy01 = y0 - y1
    dz01 = z0 - z1

    # Calculate squared terms
    sx2 = sx**2
    sy2 = sy**2
    sz2 = sz**2
    sxsy2 = (sx * sy)**2
    sxsz2 = (sx * sz)**2
    sysz2 = (sy * sz)**2

    # Calculate delta terms
    deltaSquare = (
        sysz2 * dx01**2 +
        sxsy2 * dz01**2 +
        sxsz2 * dy01**2
    )
    deltaSquareSqrt = torch.sqrt(deltaSquare)

    # Calculate denominators
    QoverDelta = Q_masked / (deltaSquareSqrt * 4.0 * np.pi)
    erfArgDenominator = sqrt2 * deltaSquareSqrt * sx * sy * sz

    # Calculate exponential term
    exp_term = torch.exp(-0.5 * (
        sy2 * torch.pow(xpos * dz01 + (z1*x0 - z0*x1) - zpos * dx01, 2) +
        sx2 * torch.pow(ypos * dz01 + (z1*y0 - z0*y1) - zpos * dy01, 2) +
        sz2 * torch.pow(ypos * dx01 + (x1*y0 - x0*y1) - xpos * dy01, 2)
    ) / deltaSquare)

    # Calculate error function term
    erf_term = (
        torch.erf((
            sysz2 * (xpos - x0) * dx01 +
            sxsy2 * (zpos - z0) * dz01 +
            sxsz2 * (ypos - y0) * dy01
        ) / erfArgDenominator) -
        torch.erf((
            sysz2 * (xpos - x1) * dx01 +
            sxsy2 * (zpos - z1) * dz01 +
            sxsz2 * (ypos - y1) * dy01
        ) / erfArgDenominator)
    )

    # Calculate masked charge values and assign to output
    masked_charge = -QoverDelta * exp_term * erf_term
    charge[b_indices, x_indices, y_indices, z_indices] = masked_charge

    return charge


def gauss_conv_line_3d_mask(Q, X0, X1, Sigma, x, y, z, mask, device='cuda'):
    """
    Calculate the 3D Gaussian convolution along a line segment with masking.
    This function computes the charge distribution from a line-segment source
    convoluted with a 3D Gaussian distribution, but only at the positions
    where `mask` is True.

    Args:
        Q (torch.Tensor): Charge values, shape: [batch_size]
        X0 (torch.Tensor): Starting points of line segments, shape: [batch_size, 3]
        X1 (torch.Tensor): Ending points of line segments, shape: [batch_size, 3]
        Sigma (torch.Tensor): Std dev in x, y, z directions, shape: [batch_size, 3]
        x (torch.Tensor): x-coordinates of the 3D grid, shape: [nx, ny, nz]
        y (torch.Tensor): y-coordinates of the 3D grid, shape: [nx, ny, nz]
        z (torch.Tensor): z-coordinates of the 3D grid, shape: [nx, ny, nz]
            (Typically created from something like `torch.meshgrid(x1d, y1d, z1d, indexing='ij')`)
        mask (torch.Tensor): Boolean mask, shape: [nx, ny, nz], where True indicates
            that we should compute the Gaussian convolution at that grid point.
        device (str, optional): Device to perform calculations on. Defaults to 'cuda'.

    Returns:
        torch.Tensor: A tensor of shape [batch_size, nx, ny, nz] containing
            the computed charge distribution, with non-zero values only
            where `mask` is True.
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
    # grid_size = x.size()
    # result_shape = (batch_size,) + grid_size
    result_shape = x.size()
    grid_size = result_shape[1:]
    charge = torch.zeros(result_shape, device=device, dtype=torch.float32)

    mask = mask.to(device)
    # Get masked indices
    b_indices, x_indices, y_indices, z_indices = torch.where(mask)

    xpos = x[b_indices, x_indices, y_indices, z_indices] # (Nmask,)
    ypos = y[b_indices, x_indices, y_indices, z_indices] # (Nmask,)
    zpos = z[b_indices, x_indices, y_indices, z_indices] # (Nmask,)

    # Reshape parameters for batch computation
    Q = Q.view([batch_size]+ len(grid_size) * [1]).expand(result_shape)
    x0 = X0[:, 0].view([batch_size]+ len(grid_size) * [1]).expand(result_shape)
    y0 = X0[:, 1].view([batch_size]+ len(grid_size) * [1]).expand(result_shape)
    z0 = X0[:, 2].view([batch_size]+ len(grid_size) * [1]).expand(result_shape)
    x1 = X1[:, 0].view([batch_size]+ len(grid_size) * [1]).expand(result_shape)
    y1 = X1[:, 1].view([batch_size]+ len(grid_size) * [1]).expand(result_shape)
    z1 = X1[:, 2].view([batch_size]+ len(grid_size) * [1]).expand(result_shape)

    sx = Sigma[:, 0].view([batch_size]+ len(grid_size) * [1]).expand(result_shape)
    sy = Sigma[:, 1].view([batch_size]+ len(grid_size) * [1]).expand(result_shape)
    sz = Sigma[:, 2].view([batch_size]+ len(grid_size) * [1]).expand(result_shape)

    Q = Q[b_indices, x_indices, y_indices, z_indices]
    x0 = x0[b_indices, x_indices, y_indices, z_indices]
    y0 = y0[b_indices, x_indices, y_indices, z_indices]
    z0 = z0[b_indices, x_indices, y_indices, z_indices]
    x1 = x1[b_indices, x_indices, y_indices, z_indices]
    y1 = y1[b_indices, x_indices, y_indices, z_indices]
    z1 = z1[b_indices, x_indices, y_indices, z_indices]
    sx = sx[b_indices, x_indices, y_indices, z_indices]
    sy = sy[b_indices, x_indices, y_indices, z_indices]
    sz = sz[b_indices, x_indices, y_indices, z_indices]

    # Prepare coordinates for broadcasting
    # xpos = xpos.unsqueeze(0)  # [1, Nmask]
    # ypos = ypos.unsqueeze(0)  # [1, Nmask]
    # zpos = zpos.unsqueeze(0)  # [1, Nmask]

    # Calculate differences [batch_size, 1]
    # [Nmask,]
    dx01 = x0 - x1
    dy01 = y0 - y1
    dz01 = z0 - z1

    # Calculate squared terms [batch_size, 1]
    # (Nmask,)
    sx2 = sx**2
    sy2 = sy**2
    sz2 = sz**2
    sxsy2 = (sx * sy)**2
    sxsz2 = (sx * sz)**2
    sysz2 = (sy * sz)**2

    # Calculate delta terms [batch_size, 1]
    # (Nmask,)
    deltaSquare = (
        sysz2 * dx01**2 +
        sxsy2 * dz01**2 +
        sxsz2 * dy01**2
    )
    deltaSquareSqrt = torch.sqrt(deltaSquare)

    # Calculate denominators [batch_size, 1]
    QoverDelta = Q / (deltaSquareSqrt * 4.0 * np.pi)
    erfArgDenominator = sqrt2 * deltaSquareSqrt * sx * sy * sz


    # Calculate exponential term [batch_size, Nmask]
    exp_term = torch.exp(-0.5 * (
        sy2 * torch.pow(xpos * dz01 + (z1*x0 - z0*x1) - zpos * dx01, 2) +
        sx2 * torch.pow(ypos * dz01 + (z1*y0 - z0*y1) - zpos * dy01, 2) +
        sz2 * torch.pow(ypos * dx01 + (x1*y0 - x0*y1) - xpos * dy01, 2)
    ) / deltaSquare)

    # Calculate error function term [batch_size, Nmask]
    erf_term = (
        torch.erf((
            sysz2 * (xpos - x0) * dx01 +
            sxsy2 * (zpos - z0) * dz01 +
            sxsz2 * (ypos - y0) * dy01
        ) / erfArgDenominator) -
        torch.erf((
            sysz2 * (xpos - x1) * dx01 +
            sxsy2 * (zpos - z1) * dz01 +
            sxsz2 * (ypos - y1) * dy01
        ) / erfArgDenominator)
    )

    # Calculate masked charge values [batch_size, Nmask]
    masked_charge = -QoverDelta * exp_term * erf_term

    # Assign computed values back to the full grid
    charge[b_indices, x_indices, y_indices, z_indices] = masked_charge

    return charge

def gauss_conv_line_3d_mask_optimized2(Q, X0, X1, Sigma, x, y, z, mask, device='cuda'):
    """
    Further optimized version that minimizes memory usage and computations
    """
    sqrt2 = np.sqrt(2)

    # Move tensors to device and ensure correct dtype
    Q = Q.to(device, dtype=torch.float32)
    X0 = X0.to(device, dtype=torch.float32)
    X1 = X1.to(device, dtype=torch.float32)
    Sigma = Sigma.to(device, dtype=torch.float32)
    mask = mask.to(device)

    # Shape validations
    if len(X0.shape) != 2 or X0.shape[1] != 3:
        raise ValueError(f'X0 shape must be (N, 3), got {X0.shape}')
    
    result_shape = x.size()
    charge = torch.zeros(result_shape, device=device, dtype=torch.float32)
    
    # Get masked indices only once
    batch_idx, x_idx, y_idx, z_idx = torch.where(mask)
    
    if len(batch_idx) == 0:
        return charge

    # Extract coordinates only for masked points (avoid expanding tensors)
    xpos = x[batch_idx, x_idx, y_idx, z_idx]
    ypos = y[batch_idx, x_idx, y_idx, z_idx]
    zpos = z[batch_idx, x_idx, y_idx, z_idx]

    # Get parameters for the relevant indices
    Q_batch = Q[batch_idx]
    x0, y0, z0 = X0[batch_idx].T
    x1, y1, z1 = X1[batch_idx].T
    sx, sy, sz = Sigma[batch_idx].T

    # Pre-compute common terms (in-place where possible)
    dx01 = x0.sub_(x1)  # In-place subtraction
    dy01 = y0.sub_(y1)
    dz01 = z0.sub_(z1)

    # Square terms (in-place)
    sx2 = sx.square_()
    sy2 = sy.square_()
    sz2 = sz.square_()
    
    # Pre-compute products (reuse memory)
    sxy = sx.mul(sy)
    sxz = sx.mul(sz)
    syz = sy.mul(sz)
    sxsy2 = sxy.square_()
    sxsz2 = sxz.square_()
    sysz2 = syz.square_()

    # Calculate delta terms (reusing memory where possible)
    deltaSquare = torch.addcmul(
        torch.addcmul(
            sysz2.mul_(dx01.square_()), 
            sxsy2, 
            dz01.square_()
        ), 
        sxsz2,
        dy01.square_()
    )
    
    deltaSquareSqrt = deltaSquare.sqrt_()
    
    # Compute charge values directly
    QoverDelta = Q_batch.div(deltaSquareSqrt.mul_(4.0 * np.pi))
    erfArgDenom = deltaSquareSqrt.mul_(sqrt2).mul_(sx).mul_(sy).mul_(sz)

    # Calculate final values efficiently
    charge_vals = torch.empty_like(xpos)
    charge_vals = _compute_charge_component(
        charge_vals, QoverDelta, erfArgDenom,
        xpos, ypos, zpos, x0, y0, z0, x1, y1, z1,
        sx2, sy2, sz2, sxsy2, sxsz2, sysz2,
        dx01, dy01, dz01, deltaSquare
    )

    # Assign results
    charge[batch_idx, x_idx, y_idx, z_idx] = charge_vals
    return charge

@torch.jit.script
def _compute_charge_component(charge_vals, QoverDelta, erfArgDenom,
                            xpos, ypos, zpos, x0, y0, z0, x1, y1, z1,
                            sx2, sy2, sz2, sxsy2, sxsz2, sysz2,
                            dx01, dy01, dz01, deltaSquare):
    """
    JIT-compiled helper function to compute the final charge values
    """
    # Compute exp term components
    exp_term = (
        sy2 * (xpos * dz01 + (z1*x0 - z0*x1) - zpos * dx01).square_() +
        sx2 * (ypos * dz01 + (z1*y0 - z0*y1) - zpos * dy01).square_() +
        sz2 * (ypos * dx01 + (x1*y0 - x0*y1) - xpos * dy01).square_()
    ).div_(deltaSquare).mul_(-0.5).exp_()

    # Compute erf term components
    erf_term = torch.erf((
        sysz2 * (xpos - x0) * dx01 +
        sxsy2 * (zpos - z0) * dz01 +
        sxsz2 * (ypos - y0) * dy01
    ).div_(erfArgDenom))
    
    erf_term.sub_(torch.erf((
        sysz2 * (xpos - x1) * dx01 +
        sxsy2 * (zpos - z1) * dz01 +
        sxsz2 * (ypos - y1) * dy01
    ).div_(erfArgDenom)))

    return -QoverDelta * exp_term * erf_term

def test_consistency(Q, X0, X1, Sigma, x, y, z, mask, device='cuda'):
    result1 = gauss_conv_line_3d_orig(Q, X0, X1, Sigma, x, y, z, device)
    result2 = gauss_conv_line_3d_mask_optimized2(Q, X0, X1, Sigma, x, y, z, mask, device)



    # result1_mask = result1[:, mask]  # This will broadcast the batch dimension across the masked elements
    # result2_mask = result2[:, mask]  # This will broadcast the batch dimension across the masked elements
    result1_mask = result1[mask]
    result2_mask = result2[mask]

    # print(result1_mask)
    # print(result2_mask)
    # print(result1_mask - result2_mask)

    relative_diff = torch.abs(result1_mask - result2_mask) / (torch.abs(result1_mask) + 1e-10)
    max_diff = torch.max(relative_diff)
    mean_diff = torch.mean(relative_diff)

    #print(result1- result2)
    max_element = torch.max(result1_mask - result2_mask)
    min_element = torch.min(result1_mask - result2_mask)
    print(f"Max, Min element in result1 - result2: {max_element.item()}, {min_element.item()}")

    return max_diff, mean_diff

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

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    torch.cuda.empty_cache()  # Clear any existing allocations

    ndim = 33
    nevent = 1_000

    # Define grid parameters
    origin = (0.0, 0.0, 0.0)
    spacing = (0.1, 0.1, 0.1)
    shape = (ndim, ndim, ndim)

    # Create grid on GPU
    x, y, z = create_grid_3d(origin, spacing, shape, device)
    xyzshape = (nevent, x.shape[0], x.shape[1], x.shape[2])
    x = x.unsqueeze(0).expand(xyzshape).clone()
    y = y.unsqueeze(0).expand(xyzshape).clone()
    z = z.unsqueeze(0).expand(xyzshape).clone()

    # Define line segment parameters (directly on GPU)
    Q = torch.ones(nevent, device=device)
    X0 = torch.rand(nevent, 3, device=device)
    X1 = torch.rand(nevent, 3, device=device)
    Sigma = torch.full((nevent, 3), 0.2, device=device)

    # # Making a mask
    # mask = torch.zeros(ndim, ndim, ndim, dtype=torch.bool)
    # num_ones = int(0.01 * mask.numel())  # 1% of total elements
    # indices = torch.randint(0, mask.numel(), (num_ones,))
    # mask.view(-1)[indices] = True


    # Warm-up runs
    # print("Warming up...")
    # for _ in range(3):
    #     with torch.no_grad():
    #         _ = gauss_conv_line_3d_orig(Q, X0, X1, Sigma, x, y, z, device)
    #         _ = gauss_conv_line_3d_mask(Q, X0, X1, Sigma, x, y, z, mask, device)
    #     torch.cuda.synchronize()
    #     torch.cuda.empty_cache()

    ffs = [0.01 * i for i in range(1, 55, 2)]
    tmask = []
    torig = []
    norig = []
    nmask = []

    # Run multiple times
    for ff in ffs:
        # print(f"\nRun {run + 1}/10")
        mask = torch.zeros(xyzshape, dtype=torch.bool)
        n1 = int(ff * mask.numel())  # ffx100% of total elements
        indices = torch.randint(0, mask.numel(), (n1,))
        mask.view(-1)[indices] = True

        # Define line segment parameters (directly on GPU)
        # Q = torch.ones(nevent, device=device)
        # X0 = torch.rand(nevent, 3, device=device)
        # X1 = torch.rand(nevent, 3, device=device)
        # Sigma = torch.full((nevent, 3), 0.2, device=device)

        # Original calculation
        print("Original Calculation")
        torch.cuda.reset_peak_memory_stats()
        start_time = time.time()
        with torch.no_grad():
            # charge = gauss_conv_line_3d_orig(Q, X0, X1, Sigma, x, y, z, device)
            tstats = benchmark.Timer(
                stmt = 'gauss_conv_line_3d_orig(Q, X0, X1, Sigma, x, y, z, device)',
                setup = 'from __main__ import gauss_conv_line_3d_orig',
                globals={'Q' : Q, 'X0' : X0, 'X1': X1, 'Sigma' : Sigma, 'x' : x, 'y' : y, 'z' : z, 'device' : device}
                )
            m = tstats.adaptive_autorange(min_run_time=0.5)
            print(m)
            torig.append(m.mean)
            # norig.append(tstats.collect_callgrind().counts(denoise=True))

        # gpu_time = time.time() - start_time

        # current_mem = torch.cuda.memory_allocated() / 1024**2
        # peak_mem = torch.cuda.max_memory_allocated() / 1024**2
        # print(f"Current GPU memory usage: {current_mem:.2f} MB")
        # print(f"Peak GPU memory usage: {peak_mem:.2f} MB")
        # print(f"GPU Time: {gpu_time:.4f} seconds")

        # del charge
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # New Calculation
        print("New Calculation")
        torch.cuda.reset_peak_memory_stats()
        start_time = time.time()
        with torch.no_grad():
            # charge = gauss_conv_line_3d_mask(Q, X0, X1, Sigma, x, y, z, mask, device)
            tstats = benchmark.Timer(
                stmt = 'gauss_conv_line_3d_mask_optimized2(Q, X0, X1, Sigma, x, y, z, mask, device)',
                setup = 'from __main__ import gauss_conv_line_3d_mask_optimized2',
                globals={'Q' : Q, 'X0' : X0, 'X1': X1, 'Sigma' : Sigma, 'x' : x, 'y' : y, 'z' : z,
                         'mask': mask, 'device' : device}
                )
            m = tstats.blocked_autorange(min_run_time=1)
            print(m)
            tmask.append(m.mean)
            # nmask.append(tstats.collect_callgrind().counts(denoise=True))

        # gpu_time = time.time() - start_time

        # current_mem = torch.cuda.memory_allocated() / 1024**2
        # peak_mem = torch.cuda.max_memory_allocated() / 1024**2
        # print(f"Current GPU memory usage: {current_mem:.2f} MB")
        # print(f"Peak GPU memory usage: {peak_mem:.2f} MB")
        # print(f"GPU Time: {gpu_time:.4f} seconds")

        print("Difference between old and new calculations")
        with torch.no_grad():
            test_consistency(Q, X0, X1, Sigma, x, y, z, mask, device)

        # del charge
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


    # print(torig, tmask)
    return ffs, torig, tmask, norig, nmask

    # print(f"Grid shape: {charge.shape}")
    # print(f"Total charge: {torch.sum(charge).item():.6f}")
    # print(f"Max charge density: {torch.max(charge).item():.6f}")
    # print(f"Min charge density: {torch.min(charge).item():.6f}")


if __name__ == "__main__":
    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name(0)}\n")


    # Start tracing memory allocation
    tracemalloc.start()

    # GPU Operation

    ffs, torig, tmask, norig, nmask = main()
    torig = np.array(torig).mean(axis=0)
    plt.plot(ffs, tmask, 'o-', label='w/ mask')
    plt.hlines(torig, xmin=ffs[0], xmax=ffs[-1], linestyles='dashed', label='w/o masks')
    plt.xlabel('Filling factor of mask')
    plt.ylabel('mean of execution time [ms]')
    plt.legend()
    plt.savefig('profile_masks2.png')

    # norig = np.array(norig).mean(axis=0)
    # plt.figure()
    # plt.plot(ffs, nmask, 'o-', label='w/ mask')
    # plt.hlines(norig, xmin=ffs[0], xmax=ffs[-1], linestyles='dashed', label='w/o masks')
    # plt.xlabel('Filling factor of mask')
    # plt.ylabel('mean of number of instructions')
    # plt.savefig('profile_masks2_ninstructions.png')

    # Get peak memory (CPU) usage
    # current, peak = tracemalloc.get_traced_memory()
    # print(f"\nCurrent memory usage: {current / 1024 / 1024:.2f} MB")
    # print(f"Peak memory usage: {peak / 1024 / 1024:.2f} MB")