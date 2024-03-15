import warnings
warnings.filterwarnings('ignore')
import time
import sys
import numpy as np
import torch
from scipy.optimize import curve_fit
import scipy.optimize as opt
import pywph as pw

'''
This component separation algorithm aims to separate the statistics of a non-Gaussian field from noise
for which we have a model, or at least some realizations. 

This algorithm is based on works described in Régaldo et al. 2021, Delouis et al. 2022 and Auclair et al. 2023.

The quantities involved are d (the noisy map), s (the pure map) and n (the noise map).

We also denote by u the running map.

This algorithm solves the inverse problem d = s + n from a statistical point of view.

''' 

###############################################################################
# INPUT DATA
###############################################################################

s = np.load('s.npy').astype(np.float64) # Load the contaminated data

###############################################################################
# INPUT PARAMETERS
###############################################################################

SNR = 2 # Signal-to-noise ratio

style = 'B' # Component separation style : 'B' for 'à la Bruno' and 'JM' for 'à la Jean-Marc'

file_name="separation_results_"+style+".npy" # Name of the ouput file

(N, N) = np.shape(s) # Size of the maps
Mn = 50 # Number of noise realizations
d = s + np.random.normal(0,np.std(s)/SNR,size=(N,N)).astype(np.float64) # Creates the noisy map
n = np.random.normal(0,np.std(s)/SNR,size=(Mn,N,N)).astype(np.float64) # Creates the set of noise realizations
J = int(np.log2(N)-2) # Maximum scale to take into account
L = 4 # Number of wavelet orientations in [0,pi]
method = 'L-BFGS-B' # Optimizer
pbc = False # Periodic boundary conditions
dn = 5 # Number of translations
n_step = 3 # Number of steps of optimization
iter_per_step = 30 # Number of iterations in each step
device = 0 # GPU to use
batch_size = 5 # Size of the batches for WPH computations
batch_number = int(Mn/batch_size) # Number of batches
wph_model = ["S11","S00","S01","Cphase","C01","C00","L"] # Set of WPH coefficient to use

###############################################################################
# USEFUL FUNCTIONS
###############################################################################

def create_batch(n, device):
    # Creates a batches of noise maps to speed up the std computations.
    batch = torch.zeros([batch_number,batch_size,N,N])
    for i in range(batch_number):
        batch[i] = torch.from_numpy(n)[i*batch_size:(i+1)*batch_size,:,:]
    return batch.to(device)

def compute_bias_std(x, noise_batch):
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
    # Computes the appropriate threshold for the WPH coefficients
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

def compute_mask_S11(x):
    # Computes the mask for S11 coeffs (at the first step)
    wph_op.load_model(wph_model)
    full_coeffs = wph_op.apply(x,norm=None,pbc=pbc)
    thresh = get_thresh(full_coeffs)
    wph_op.load_model(['S11'])
    coeffs = wph_op.apply(x,norm=None,pbc=pbc)
    mask_real = torch.abs(torch.real(coeffs)).to(device) > thresh
    mask_imag = torch.abs(torch.imag(coeffs)).to(device) > thresh
    print("Real mask computed :",int(100*(mask_real.sum()/mask_real.size(dim=0)).item()),"% of coeffs kept !")
    print("Imaginary mask computed :",int(100*(mask_imag.sum()/mask_imag.size(dim=0)).item()),"% of coeffs kept !")
    mask = torch.cat((torch.unsqueeze(mask_real,dim=0),torch.unsqueeze(mask_imag,dim=0)))
    return mask.to(device)

def compute_mask(x,std):
    # Computes the mask for the full set of coeffs (at the second step)
    coeffs = wph_op.apply(x,norm=None,pbc=pbc)
    thresh = get_thresh(coeffs)
    mask_real = torch.logical_and(torch.abs(torch.real(coeffs)).to(device) > thresh, std[0].to(device) > 0)
    mask_imag = torch.logical_and(torch.abs(torch.imag(coeffs)).to(device) > thresh, std[1].to(device) > 0)
    print("Real mask computed :",int(100*(mask_real.sum()/mask_real.size(dim=0)).item()),"% of coeffs kept !")
    print("Imaginary mask computed :",int(100*(mask_imag.sum()/mask_imag.size(dim=0)).item()),"% of coeffs kept !")
    mask = torch.cat((torch.unsqueeze(mask_real,dim=0),torch.unsqueeze(mask_imag,dim=0)))
    return mask.to(device)

def compute_loss_B(x,coeffs_target,std,mask):
    # Computes the loss 'à la Bruno'
    coeffs_target = coeffs_target.to(device)
    std = std.to(device)
    mask = mask.to(device)
    loss_tot = torch.zeros(1)
    for j in range(Mn):
        x_noisy, nb_chunks = wph_op.preconfigure(x + torch.from_numpy(n[j]).to(device), requires_grad=True, pbc=pbc)
        for i in range(nb_chunks):
            coeffs_chunk, indices = wph_op.apply(x_noisy, i, norm=None, ret_indices=True, pbc=pbc)
            loss = ( torch.sum(torch.abs( (torch.real(coeffs_chunk)[mask[0,indices]] - coeffs_target[0][indices][mask[0,indices]]) / std[0][indices][mask[0,indices]] ) ** 2) + torch.sum(torch.abs( (torch.imag(coeffs_chunk)[mask[1,indices]] - coeffs_target[1][indices][mask[1,indices]]) / std[1][indices][mask[1,indices]] ) ** 2) ) / ( mask[0].sum() + mask[1].sum() ) / Mn
            loss.backward(retain_graph=True)
            loss_tot += loss.detach().cpu()
            del coeffs_chunk, indices, loss
    return loss_tot

def compute_loss_JM(x,coeffs_target,std,mask):
    # Computes the loss 'à la Jean-Marc'
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

def objective(x):
    # Computes the loss and the corresponding gradient 
    global eval_cnt
    print(f"Evaluation: {eval_cnt}")
    start_time = time.time()
    u = x.reshape((N, N)) # Reshape x
    u = torch.from_numpy(u).to(device).requires_grad_(True) # Track operations on u
    if style == 'B':
        L = compute_loss_B(u, coeffs_target, std, mask) # Compute the loss 'à la Bruno'
    if style == 'JM':
        L = compute_loss_JM(u, coeffs_target, std, mask) # Compute the loss 'à la Jean-Marc'
    u_grad = u.grad.cpu().numpy().astype(x.dtype) # Compute the gradient
    print("L = "+str(round(L.item(),3)))
    print("(computed in "+str(round(time.time() - start_time,3))+"s)")
    print("")
    eval_cnt += 1
    return L.item(), u_grad.ravel()

###############################################################################
# MINIMIZATION
###############################################################################

if __name__ == "__main__":
    total_start_time = time.time()
    print("Starting component separation...")
    print("Building operator...")
    start_time = time.time()
    wph_op = pw.WPHOp(N, N, J, L=L, dn=dn, device=device)
    wph_op.load_model(["S11"])
    print("Done ! (in {:}s)".format(time.time() - start_time))
    n_batch = create_batch(n, device)
    
    ## First minimization
    print("Starting first minimization...")
    eval_cnt = 0
    s_tilde0 = d # The optimzation starts from d
    for i in range(n_step):
        print("Starting era "+str(i+1)+"...")
        s_tilde0 = torch.from_numpy(s_tilde0).to(device) # Initialization of the map
        print('Computing stuff...')
        bias, std = compute_bias_std(s_tilde0, n_batch) # Computation of the bias and std
        coeffs = wph_op.apply(torch.from_numpy(d).to(device), norm=None, pbc=pbc) # Coeffs computation
        if style == 'B':
            coeffs_target = torch.cat((torch.unsqueeze(torch.real(coeffs),dim=0),torch.unsqueeze(torch.imag(coeffs),dim=0)))
        if style == 'JM':
            coeffs_target = torch.cat((torch.unsqueeze(torch.real(coeffs)-bias[0],dim=0),torch.unsqueeze(torch.imag(coeffs)-bias[1],dim=0)))
        mask = compute_mask_S11(s_tilde0) # Mask computation
        print('Stuff computed !')
        print('Beginning optimization...')
        result = opt.minimize(objective, s_tilde0.cpu().ravel(), method=method, jac=True, tol=None, options={"maxiter": iter_per_step, "gtol": 1e-14, "ftol": 1e-14, "maxcor": 20})
        final_loss, s_tilde0, niter, msg = result['fun'], result['x'], result['nit'], result['message']
        # Reshaping
        s_tilde0 = s_tilde0.reshape((N, N)).astype(np.float32)
        print("Era "+str(i+1)+" done !")
        
    ## Second minimization
    print("Starting second minimization...")
    eval_cnt = 0
    s_tilde = s_tilde0 # The second step starts from the result of the first step
    wph_op.load_model(wph_model)
    for i in range(n_step):
        print("Starting era "+str(i+1)+"...")
        s_tilde = torch.from_numpy(s_tilde).to(device) # Initialization of the map
        print('Computing stuff...')
        bias, std = compute_bias_std(s_tilde, n_batch) # Computation of the bias and std
        coeffs = wph_op.apply(torch.from_numpy(d).to(device), norm=None, pbc=pbc) # Coeffs computation
        if style == 'B':
            coeffs_target = torch.cat((torch.unsqueeze(torch.real(coeffs),dim=0),torch.unsqueeze(torch.imag(coeffs),dim=0)))
        if style == 'JM':
            coeffs_target = torch.cat((torch.unsqueeze(torch.real(coeffs)-bias[0],dim=0),torch.unsqueeze(torch.imag(coeffs)-bias[1],dim=0)))
        mask = compute_mask(s_tilde, std) # Mask computation
        print('Stuff computed !')
        print('Beginning optimization...')
        result = opt.minimize(objective, s_tilde.cpu().ravel(), method=method, jac=True, tol=None, options={"maxiter": iter_per_step, "gtol": 1e-14, "ftol": 1e-14, "maxcor": 20})
        final_loss, s_tilde, niter, msg = result['fun'], result['x'], result['nit'], result['message']
        # Reshaping
        s_tilde = s_tilde.reshape((N, N)).astype(np.float32)
        print("Era "+str(i+1)+" done !")
        
    ## Output
    print("Denoising done ! (in {:}s)".format(time.time() - total_start_time))
    if file_name is not None:
        np.save(file_name, np.array([d,s,s_tilde,s_tilde0]))
