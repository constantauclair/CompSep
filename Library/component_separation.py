import warnings
warnings.filterwarnings('ignore')
import time
import numpy as np
import torch
import scipy.optimize as opt
import pywph as pw
import os 
cwd = os.getcwd()
import sys
sys.path.append(cwd)
from comp_sep_functions import create_batch, compute_bias_std, compute_mask, compute_loss_B, compute_loss_JM

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

s = np.load('../Data/intensity_map.npy').astype(np.float64) # Load the clean data

###############################################################################
# INPUT PARAMETERS
###############################################################################

SNR = 2 # Signal-to-noise ratio

style = 'B' # Component separation style : 'B' for 'à la Bruno' and 'JM' for 'à la Jean-Marc'

file_name="../Results/separation_results_"+style+".npy" # Name of the ouput file

(N, N) = np.shape(s) # Size of the maps
Mn = 100 # Number of noise realizations
d = s + np.random.normal(0,np.std(s)/SNR,size=(N,N)).astype(np.float64) # Creates the noisy map
noise = np.random.normal(0,np.std(s)/SNR,size=(Mn,N,N)).astype(np.float64) # Creates the set of noise realizations
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
# OBJECTIVE FUNCTION
###############################################################################

def objective(x):
    """
    Computes the loss and the corresponding gradient.

    Parameters
    ----------
    x : torch 2D tensor
        Running map.

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
    n_batch = create_batch(noise, device, batch_number, batch_size, N)
    
    ## First minimization
    print("Starting first minimization...")
    eval_cnt = 0
    s_tilde0 = d # The optimzation starts from d
    for i in range(n_step):
        print("Starting era "+str(i+1)+"...")
        s_tilde0 = torch.from_numpy(s_tilde0).to(device) # Initialization of the map
        print('Computing stuff...')
        bias, std = compute_bias_std(s_tilde0, n_batch, wph_op, pbc, Mn, batch_number, batch_size, device) # Computation of the bias and std
        coeffs = wph_op.apply(torch.from_numpy(d).to(device), norm=None, pbc=pbc) # Coeffs computation
        if style == 'B':
            coeffs_target = torch.cat((torch.unsqueeze(torch.real(coeffs),dim=0),torch.unsqueeze(torch.imag(coeffs),dim=0)))
        if style == 'JM':
            coeffs_target = torch.cat((torch.unsqueeze(torch.real(coeffs)-bias[0],dim=0),torch.unsqueeze(torch.imag(coeffs)-bias[1],dim=0)))
        mask = compute_mask(1, s_tilde0, std, wph_op, wph_model, pbc, device) # Mask computation
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
        bias, std = compute_bias_std(s_tilde, n_batch, wph_op, pbc, Mn, batch_number, batch_size, device) # Computation of the bias and std
        coeffs = wph_op.apply(torch.from_numpy(d).to(device), norm=None, pbc=pbc) # Coeffs computation
        if style == 'B':
            coeffs_target = torch.cat((torch.unsqueeze(torch.real(coeffs),dim=0),torch.unsqueeze(torch.imag(coeffs),dim=0)))
        if style == 'JM':
            coeffs_target = torch.cat((torch.unsqueeze(torch.real(coeffs)-bias[0],dim=0),torch.unsqueeze(torch.imag(coeffs)-bias[1],dim=0)))
        mask = compute_mask(2, s_tilde, std, wph_op, wph_model, pbc, device) # Mask computation
        print('Stuff computed !')
        print('Beginning optimization...')
        result = opt.minimize(objective, s_tilde.cpu().ravel(), method=method, jac=True, tol=None, options={"maxiter": iter_per_step, "gtol": 1e-14, "ftol": 1e-14, "maxcor": 20})
        final_loss, s_tilde, niter, msg = result['fun'], result['x'], result['nit'], result['message']
        # Reshaping
        s_tilde = s_tilde.reshape((N, N)).astype(np.float32)
        print("Era "+str(i+1)+" done !")
        
    ## Output
    print("Denoising done ! (in {:}s)".format(time.time() - total_start_time))
    np.save(file_name, np.array([d,s,s_tilde,s_tilde0]))
