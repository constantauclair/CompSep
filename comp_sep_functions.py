import numpy as np
import torch
import sys
import time
from scipy.optimize import curve_fit

def create_batch(n, device, batch_number, batch_size, N):
    """
    Creates a batch of noise maps.

    Parameters
    ----------
    n : numpy 3D array
        Vector of noise maps.
    device : int or str
        Device on which the batch are put.
    batch_number : int
        Number of batch.
    batch_size : int
        Number of maps in each batch.
    N : int
        Pixel size of the maps.

    Returns
    -------
    torch 4D tensor
        Tensor of batches of noise maps.

    """
    batch = torch.zeros([batch_number,batch_size,N,N])
    for i in range(batch_number):
        batch[i] = torch.from_numpy(n)[i*batch_size:(i+1)*batch_size,:,:]
    return batch.to(device)

def compute_bias_std(x, noise_batch, wph_op, pbc, Mn, batch_number, batch_size, device):
    """
    Computes the noise-induced bias on the WPH statistics of x, as well as the corresponding std.

    Parameters
    ----------
    x : torch 2D tensor
        Map on which the bias is computed.
    noise_batch : torch 4D tensor
        Noise batches.
    wph_op : pywph.wph_op
        The WPH operator.
    pbc : bool
        True for periodic boundary conditions.
    Mn : int
        Number of noise maps.
    batch_number : int
        Number of batch.
    batch_size : int
        Number of maps in each batch.
    device : int or str
        Device on which the computations are done.

    Returns
    -------
    torch 1D tensor
        Tensor of biases on the WPH statistics.
    torch 1D tensor
        Tensor of standard deviations of the WPH statistics.

    """
    # Computes the noise-induced bias on x and the corresponding std
    coeffs_ref = wph_op.apply(x, norm=None, pbc=pbc)
    coeffs_number = coeffs_ref.size(-1)
    ref_type = coeffs_ref.type()
    COEFFS = torch.zeros((Mn,coeffs_number)).type(dtype=ref_type)
    for i in range(batch_number):
        batch_COEFFS = torch.zeros((batch_size,coeffs_number)).type(dtype=ref_type)
        u, nb_chunks = wph_op.preconfigure(x + noise_batch[i], pbc=pbc)
        for j in range(nb_chunks):
            coeffs_chunk, indices = wph_op.apply(u, j, norm=None, ret_indices=True, pbc=pbc)
            batch_COEFFS[:,indices] = coeffs_chunk.type(dtype=ref_type) - coeffs_ref[indices].type(dtype=ref_type)
            del coeffs_chunk, indices
        COEFFS[i*batch_size:(i+1)*batch_size] = batch_COEFFS
        del u, nb_chunks, batch_COEFFS
        sys.stdout.flush() # Flush the standard output
    bias = torch.cat((torch.unsqueeze(torch.mean(torch.real(COEFFS),axis=0),dim=0),torch.unsqueeze(torch.mean(torch.imag(COEFFS),axis=0),dim=0)))
    std = torch.cat((torch.unsqueeze(torch.std(torch.real(COEFFS),axis=0),dim=0),torch.unsqueeze(torch.std(torch.imag(COEFFS),axis=0),dim=0)))
    return bias.to(device), std.to(device)

def get_thresh(coeffs):
    """
    Computes the appropriate threshold for the WPH statistics in the loss function.

    Parameters
    ----------
    coeffs : torch 1D tensor
        Tensor of the WPH statistics.

    Returns
    -------
    float
        Threshold value.

    """
    coeffs_for_hist = np.abs(coeffs.cpu().numpy().flatten())
    non_zero_coeffs_for_hist = coeffs_for_hist[np.where(coeffs_for_hist>0)]
    hist, bins_edges = np.histogram(np.log10(non_zero_coeffs_for_hist),bins=100,density=True)
    bins = (bins_edges[:-1] + bins_edges[1:]) / 2
    x = bins
    y = hist
    def func(x, mu1, sigma1, amp1, mu2, sigma2, amp2):
        y = amp1 * np.exp( -((x - mu1)/sigma1)**2) + amp2 * np.exp( -((x - mu2)/sigma2)**2)
        return y
    guess = [x[0]+(x[-1]-x[0])/4, 1, 0.3, x[0]+3*(x[-1]-x[0])/4, 1, 0.3]
    popt, pcov = curve_fit(func, x, y, p0=guess)
    thresh = 10**((popt[0]+popt[3])/2)
    return thresh

def compute_mask(step, x, std, wph_op, wph_model, pbc, device):
    """
    Computes the mask for the WPH statistics.

    Parameters
    ----------
    step : int
        Choose the step of the algorithm.
    x : torch 2D tensor
        Reference map for the mask computation.
    std : torch 1D tensor
        Standard deviations of the WPH statistics.
    wph_op : pywph.wph_op
        WPH statistics operator.
    wph_model : list
        Set of WPH coefficients.
    pbc : bool
        True for periodic boundary conditions.
    device : int or str
        Device on which the computations are done.

    Returns
    -------
    torch 1D tensor
        WPH statistics mask.

    """
    if step == 1:
        wph_op.load_model(wph_model)
        full_coeffs = wph_op.apply(x,norm=None,pbc=pbc)
        thresh = get_thresh(full_coeffs)
        wph_op.load_model(['S11'])
        coeffs = wph_op.apply(x,norm=None,pbc=pbc)
    if step == 2:
        coeffs = wph_op.apply(x,norm=None,pbc=pbc)
        thresh = get_thresh(coeffs)
    mask_real = torch.logical_and(torch.abs(torch.real(coeffs)).to(device) > thresh, std[0].to(device) > 0)
    mask_imag = torch.logical_and(torch.abs(torch.imag(coeffs)).to(device) > thresh, std[1].to(device) > 0)
    print("Real mask computed :",int(100*(mask_real.sum()/mask_real.size(dim=0)).item()),"% of coeffs kept !")
    print("Imaginary mask computed :",int(100*(mask_imag.sum()/mask_imag.size(dim=0)).item()),"% of coeffs kept !")
    mask = torch.cat((torch.unsqueeze(mask_real,dim=0),torch.unsqueeze(mask_imag,dim=0)))
    return mask.to(device)

def compute_loss_B(x, coeffs_target, std, mask, device, Mn, wph_op, noise, pbc):
    """
    Computes the loss in "Bruno's formalism".

    Parameters
    ----------
    x : torch 2D tensor
        Running map.
    coeffs_target : torch 1D tensor
        Target WPH statistics.
    std : torch 1D tensor
        Standard deviations of the WPH statistics.
    mask : torch 1D tensor
        Mask for the WPH statistics.
    device : int or str
        Device on which the computation are done.
    Mn : int
        Number of noise maps.
    wph_op : pywph.wph_op
        WPH statistics operator.
    noise : numpy 3D array
        Noise maps.
    pbc : bool
        True for periodic boundary conditions.

    Returns
    -------
    float
        The total loss value.

    """
    coeffs_target = coeffs_target.to(device)
    std = std.to(device)
    mask = mask.to(device)
    loss_tot = torch.zeros(1)
    for j in range(Mn):
        x_noisy, nb_chunks = wph_op.preconfigure(x + torch.from_numpy(noise[j]).to(device), requires_grad=True, pbc=pbc)
        for i in range(nb_chunks):
            coeffs_chunk, indices = wph_op.apply(x_noisy, i, norm=None, ret_indices=True, pbc=pbc)
            loss = ( torch.sum(torch.abs( (torch.real(coeffs_chunk)[mask[0,indices]] - coeffs_target[0][indices][mask[0,indices]]) / std[0][indices][mask[0,indices]] ) ** 2) + torch.sum(torch.abs( (torch.imag(coeffs_chunk)[mask[1,indices]] - coeffs_target[1][indices][mask[1,indices]]) / std[1][indices][mask[1,indices]] ) ** 2) ) / ( mask[0].sum() + mask[1].sum() ) / Mn
            loss.backward(retain_graph=True)
            loss_tot += loss.detach().cpu()
            del coeffs_chunk, indices, loss
    return loss_tot

def compute_loss_JM(x, coeffs_target, std, mask, device, Mn, wph_op, pbc):
    """
    Computes the loss in "Jean-Marc's formalism".

    Parameters
    ----------
    x : torch 2D tensor
        Running map.
    coeffs_target : torch 1D tensor
        Target WPH statistics.
    std : torch 1D tensor
        Standard deviations of the WPH statistics.
    mask : torch 1D tensor
        Mask for the WPH statistics.
    device : int or str
        Device on which the computation are done.
    Mn : int
        Number of noise maps.
    wph_op : pywph.wph_op
        WPH statistics operator.
    pbc : bool
        True for periodic boundary conditions.

    Returns
    -------
    float
        The total loss value.

    """
    coeffs_target = coeffs_target.to(device)
    std = std.to(device)
    mask = mask.to(device)
    loss_tot = torch.zeros(1)
    u, nb_chunks = wph_op.preconfigure(x, requires_grad=True, pbc=pbc)
    for i in range(nb_chunks):
        coeffs_chunk, indices = wph_op.apply(u, i, norm=None, ret_indices=True, pbc=pbc)
        loss = ( torch.sum(torch.abs( (torch.real(coeffs_chunk)[mask[0,indices]] - coeffs_target[0][indices][mask[0,indices]]) / std[0][indices][mask[0,indices]] ) ** 2) + torch.sum(torch.abs( (torch.imag(coeffs_chunk)[mask[1,indices]] - coeffs_target[1][indices][mask[1,indices]]) / std[1][indices][mask[1,indices]] ) ** 2) ) / ( mask[0].sum() + mask[1].sum() ) / Mn
        loss.backward(retain_graph=True)
        loss_tot += loss.detach().cpu()
        del coeffs_chunk, indices, loss
    return loss_tot

def objective(x, device, style, coeffs_target, std, mask, wph_op, noise, pbc, N, Mn):
    """
    Computes the loss and the corresponding gradient.

    Parameters
    ----------
    x : torch 2D tensor
        Running map.
    device : int or str
        Device on which the computations are done.
    style : str
        'B' for "Bruno's formalism", 'JM' for "Jean-Marc's formalism".
    coeffs_target : torch 1D tensor
        Target WPH statistics.
    std : torch 1D tensor
        Standard deviations of the WPH statistics.
    mask : torch 1D tensor
        Mask for the WPH statistics.
    wph_op : pywph.wph_op
        WPH statistics operator.
    noise : numpy 3D array
        Noise maps.
    pbc : bool
        True for periodic boundary conditions.
    N : int
        Map size.
    Mn : int
        Number of noise maps.

    Returns
    -------
    float
        Loss value.
    torch 1D tensor
        Gradient of the loss.

    """
    global eval_cnt
    print(f"Evaluation: {eval_cnt}")
    start_time = time.time()
    u = x.reshape((N, N)) # Reshape x
    u = torch.from_numpy(u).to(device).requires_grad_(True) # Track operations on u
    if style == 'B':
        L = compute_loss_B(u, coeffs_target, std, mask, device, Mn, wph_op, noise, pbc) # Compute the loss 'à la Bruno'
    if style == 'JM':
        L = compute_loss_JM(u, coeffs_target, std, mask, device, Mn, wph_op, pbc) # Compute the loss 'à la Jean-Marc'
    u_grad = u.grad.cpu().numpy().astype(x.dtype) # Compute the gradient
    print("L = "+str(round(L.item(),3)))
    print("(computed in "+str(round(time.time() - start_time,3))+"s)")
    print("")
    eval_cnt += 1
    return L.item(), u_grad.ravel()