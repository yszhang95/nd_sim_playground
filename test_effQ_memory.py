import torch
import numpy as np
import time
import tracemalloc


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
    grid_size = x.size()
    result_shape = (batch_size,) + grid_size

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
    #charge = torch.zeros(result_shape, device=device, dtype=torch.float32)

    #Calculate final charge

    charge = -QoverDeltaSquareSqrt4pi * torch.exp(-0.5 * (
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

def gauss_conv_line_3d(Q, X0, X1, Sigma, x, y, z, device='cuda'):
    """
    Simplified version of GaussConvLine3D to calculate charge distribution
    for a line segment in 3D space.
    """
    sqrt2 = np.sqrt(2)
    
    # Move inputs to GPU and ensure they're float32
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
    grid_size = x.size()
    result_shape = (batch_size,) + grid_size
    
    # Reshape x, y, z to match broadcasting requirements
    # x_expanded = x.unsqueeze(0)  # Shape: [1, nx, ny, nz]
    # y_expanded = y.unsqueeze(0)  # Shape: [1, nx, ny, nz]
    # z_expanded = z.unsqueeze(0)  # Shape: [1, nx, ny, nz]
    
    # Pre-allocate the final result tensor
    charge = torch.zeros(result_shape, device=device, dtype=torch.float32)

    # Reshape start and end points for broadcasting
    x0, y0, z0 = [X0[:,i].view(-1, 1, 1, 1) for i in range(3)]  # Shape: [batch, 1, 1, 1]
    x1, y1, z1 = [X1[:,i].view(-1, 1, 1, 1) for i in range(3)]  # Shape: [batch, 1, 1, 1]
    sx, sy, sz = [Sigma[:,i].view(-1, 1, 1, 1) for i in range(3)]  # Shape: [batch, 1, 1, 1]
    
    # Calculate squared terms
    sxsy = sx.mul(sy)
    sxsz = sx.mul(sz)
    sysz = sy.mul(sz)
    
    # Square the results in-place
    sxsy2 = sxsy**2
    sxsz2 = sxsz**2
    sysz2 = sysz**2
    sx2 = sx**2
    sy2 = sy**2
    sz2 = sz**2
    
    # # Calculate differences
    dx01 = (X0[:,0] - X1[:,0]).view(-1, 1, 1, 1)
    dy01 = (X0[:,1] - X1[:,1]).view(-1, 1, 1, 1)
    dz01 = (X0[:,2] - X1[:,2]).view(-1, 1, 1, 1)

    # # Calculate delta terms
    # deltaSquare = (
    #     sysz2 * dx01**2 +
    #     sxsy2 * dz01**2 + 
    #     sxsz2 * dy01**2
    # )
    # deltaSquareSqrt = torch.sqrt(deltaSquare)

    # # Calculate deltaSquare
    deltaSquare = sysz2.mul(dx01.pow(2))
    deltaSquare.add_(sxsy2.mul(dz01.pow(2)))
    deltaSquare.add_(sxsz2.mul(dy01.pow(2)))
    # Calculate deltaSquareSqrt (in-place)
    deltaSquareSqrt = deltaSquare.sqrt()


    # Calculate charge distribution
    # QoverDeltaSquareSqrt4pi = Q / (deltaSquareSqrt * 4 * np.pi)
    erfArgDenominator = sqrt2 * deltaSquareSqrt * sx * sy * sz

    # Calculate QoverDeltaSquareSqrt4pi
    QoverDeltaSquareSqrt4pi = Q.view(-1, 1, 1, 1) / (deltaSquareSqrt * 4 * np.pi)
    # Calculate erfArgDenominator
    erfArgDenominator = deltaSquareSqrt.mul(sqrt2).mul_(sx).mul_(sy).mul_(sz)
        
    #Calculate final charge
    # Use a single temporary tensor and perform in-place operations to reduce memory usage
    chunk_size = 16  # example slice size along z-dimension
    charge_segments = []

    for z_start in range(0, z.size(2), chunk_size):
        z_end = min(z_start + chunk_size, z.size(2))
        z_slice = z[..., z_start:z_end]  # slice the z dimension

        # Do the same math on the sliced tensors
        exp_term = torch.exp(-0.5 * (
            sy2 * torch.pow(x[..., z_start:z_end] * dz01 + (z1*x0 - z0*x1) - z_slice * dx01, 2) +
            sx2 * torch.pow(y[..., z_start:z_end] * dz01 + (z1*y0 - z0*y1) - z_slice * dy01, 2) +
            sz2 * torch.pow(y[..., z_start:z_end] * dx01 + (x1*y0 - x0*y1) - x[..., z_start:z_end] * dy01, 2)
        ) / deltaSquare)

        erf_term = (
            torch.erf((sysz2 * (x[..., z_start:z_end] - x0) * dx01 +
                       sxsy2 * (z_slice - z0) * dz01 +
                       sxsz2 * (y[..., z_start:z_end] - y0) * dy01) / erfArgDenominator)
            - 
            torch.erf((sysz2 * (x[..., z_start:z_end] - x1) * dx01 +
                       sxsy2 * (z_slice - z1) * dz01 +
                       sxsz2 * (y[..., z_start:z_end] - y1) * dy01) / erfArgDenominator)
        )

        partial_charge = -QoverDeltaSquareSqrt4pi * exp_term * erf_term
        charge_segments.append(partial_charge)

    # Reconstruct the full charge tensor
    charge = torch.cat(charge_segments, dim=3)


    # failed approach ...
    # # Calculate exponential terms with proper broadcasting
    # # Pre-allocate temporary tensor for calculations
    # temp1 = torch.zeros_like(charge)  # Shape: [batch, nx, ny, nz]
    # temp2 = torch.zeros_like(charge)  # Shape: [batch, nx, ny, nz]
    # temp3 = torch.zeros_like(charge)  # Shape: [batch, nx, ny, nz]
    # # Calculate terms with proper broadcasting
    # # temp = x.mul(dz01) + (z1.mul(x0) - z0.mul(x1)) - z.mul(dx01)
    # # exp_term.add_(sy.pow(2).mul(temp.pow(2)))
    # # Perform calculations in-place
    # # Compute safely with in-place operations
    # temp2 = temp1.copy_(x).mul_(dz01).add_(z1.mul(x0) - z0.mul(x1))  
    # temp2.sub_(temp1.copy_(z).mul_(dx01)) #.pow_(2)
    # #charge = temp1.copy_(sy).pow_(2).mul_(temp2)
    # charge = temp2
    # # temp = y_expanded * dz01 + (z1 * y0 - z0 * y1) - z_expanded * dy01
    # # exp_term += sx.pow(2) * temp.pow(2)
    # temp2 = temp1.copy_(y).mul_(dz01).add_(z1.mul(y0) - z0.mul(y1))  
    # temp2.sub_(temp1.copy_(z).mul_(dy01)).pow_(2)
    # charge.add_(temp1.copy_(sx).pow_(2).mul_(temp2))
    # # temp = y_expanded * dx01 + (x1 * y0 - x0 * y1) - x_expanded * dy01
    # # exp_term += sz.pow(2) * temp.pow(2)
    # temp2 = temp1.copy_(y).mul_(dx01).add_(x1.mul(y0) - x0.mul(y1))  
    # temp2.sub_(temp1.copy_(x).mul_(dy01)).pow_(2)
    # charge.add_(temp1.copy_(sz).pow_(2).mul_(temp2))
    # #exp_term = (-0.5 * exp_term / deltaSquare).exp()
    # charge = charge.mul_(-0.5).div_(deltaSquare).exp_()
    # # # Calculate error function terms
    # # erf_term1 = (sysz2 * (x_expanded - x0) * dx01 +
    # #              sxsy2 * (z_expanded - z0) * dz01 +
    # #              sxsz2 * (y_expanded - y0) * dy01) / erfArgDenominator
    # temp2 = temp1.copy_(x).sub_(x0).mul_(dx01).mul_(sysz2)
    # temp2.add_(temp1.copy_(z).sub_(z0).mul_(dz01).mul_(sxsy2))
    # temp2.add_(temp1.copy_(y).sub_(y0).mul_(dy01).mul_(sxsz2))
    # temp2.div_(erfArgDenominator) 
    # temp3 = torch.erf(temp2)
    # # erf_term2 = (sysz2 * (x_expanded - x1) * dx01 +
    # #              sxsy2 * (z_expanded - z1) * dz01 +
    # #              sxsz2 * (y_expanded - y1) * dy01) / erfArgDenominator
    # temp2 = temp1.copy_(x).sub_(x1).mul_(dx01).mul_(sysz2)
    # temp2.add_(temp1.copy_(z).sub_(z1).mul_(dz01).mul_(sxsy2))
    # temp2.add_(temp1.copy_(y).sub_(y1).mul_(dy01).mul_(sxsz2))
    # temp2.div_(erfArgDenominator)
    # temp3.sub_(torch.erf(temp2))
    # # Calculate final charge
    # # charge = -QoverDeltaSquareSqrt4pi * exp_term * (torch.erf(erf_term1) - torch.erf(erf_term2))
    # charge = charge.mul_(-QoverDeltaSquareSqrt4pi).mul_(temp3)

    return charge


def test_consistency(Q, X0, X1, Sigma, x, y, z, device='cuda'):
    result1 = gauss_conv_line_3d_orig(Q, X0, X1, Sigma, x, y, z, device)
    result2 = gauss_conv_line_3d(Q, X0, X1, Sigma, x, y, z, device)
    
    relative_diff = torch.abs(result1 - result2) / (torch.abs(result1) + 1e-10)
    max_diff = torch.max(relative_diff)
    mean_diff = torch.mean(relative_diff)
    
    #print(result1- result2)
    max_element = torch.max(result1 - result2)
    min_element = torch.min(result1 - result2)
    print(f"Max, Min element in result1 - result2: {max_element.item()}, {min_element.item()}")

    return max_diff, mean_diff

def create_grid_3d(origin, spacing, shape, device='cuda'):
    """
    Create a 3D grid for charge calculation.
    """
    # x = torch.arange(origin[0], origin[0] + shape[0] * spacing[0], spacing[0], device=device)
    # y = torch.arange(origin[1], origin[1] + shape[1] * spacing[1], spacing[1], device=device)
    # z = torch.arange(origin[2], origin[2] + shape[2] * spacing[2], spacing[2], device=device)
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

    ndim = 100
    nevent = 100

    # Define grid parameters
    origin = (0.0, 0.0, 0.0)
    spacing = (0.1, 0.1, 0.1)
    shape = (ndim, ndim, ndim)
    
    # Create grid on GPU
    x, y, z = create_grid_3d(origin, spacing, shape, device)
    
    # Define line segment parameters (directly on GPU)
    Q = torch.ones(nevent, device=device)
    X0 = torch.rand(nevent, 3, device=device)
    X1 = torch.rand(nevent, 3, device=device)
    Sigma = torch.full((nevent, 3), 0.2, device=device)
    

    # Original calculation 
    # Reset memory statistics
    print("Original Calculation")
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    # Calculate charge distribution
    with torch.no_grad():
        #test_consistency(Q, X0, X1, Sigma, x, y, z, device)
        charge = gauss_conv_line_3d_orig(Q, X0, X1, Sigma, x, y, z, device)
    gpu_time = time.time() - start_time
    
    current_mem = torch.cuda.memory_allocated() / 1024**2  # In MB
    peak_mem = torch.cuda.max_memory_allocated() / 1024**2  # In MB
    print(f"\nCurrent GPU memory usage: {current_mem:.2f} MB")
    print(f"Peak GPU memory usage: {peak_mem:.2f} MB")
    print(f"GPU Time: {gpu_time:.4f} seconds")

    del charge

    # NEw Calculation
    print("New Calculation")
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    # Calculate charge distribution
    with torch.no_grad():
        #test_consistency(Q, X0, X1, Sigma, x, y, z, device)
        charge = gauss_conv_line_3d(Q, X0, X1, Sigma, x, y, z, device)
    gpu_time = time.time() - start_time
    
    current_mem = torch.cuda.memory_allocated() / 1024**2  # In MB
    peak_mem = torch.cuda.max_memory_allocated() / 1024**2  # In MB
    print(f"\nCurrent GPU memory usage: {current_mem:.2f} MB")
    print(f"Peak GPU memory usage: {peak_mem:.2f} MB")
    print(f"GPU Time: {gpu_time:.4f} seconds")

    print("\nDifference between old and new calculations")
    with torch.no_grad():
        test_consistency(Q, X0, X1, Sigma, x, y, z, device)



    # print(f"Grid shape: {charge.shape}")
    # print(f"Total charge: {torch.sum(charge).item():.6f}")
    # print(f"Max charge density: {torch.max(charge).item():.6f}")
    # print(f"Min charge density: {torch.min(charge).item():.6f}")

    # # Clean up
    # del x, y, z, Q, X0, X1, Sigma, charge
    # torch.cuda.empty_cache()

if __name__ == "__main__":
    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name(0)}\n")

    
    
    # Start tracing memory allocation
    tracemalloc.start()

    # GPU Operation
    
    main()
    

    

    # Get peak memory (CPU) usage
    # current, peak = tracemalloc.get_traced_memory()
    # print(f"\nCurrent memory usage: {current / 1024 / 1024:.2f} MB")
    # print(f"Peak memory usage: {peak / 1024 / 1024:.2f} MB")