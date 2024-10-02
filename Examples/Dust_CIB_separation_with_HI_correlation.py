# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings('ignore')
import time
import sys
import multiprocessing as mp
from functools import partial
import numpy as np
import torch
import scipy.optimize as opt
import pywph as pw
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

'''
This component separation algorithm aims to separate the dust intensity emission from the CIB contamination 
on Herschel data.
For any question: constant.auclair@phys.ens.fr

The quantities involved are d (the total map), u (the optimized dust map), c (the CIB map)
and HI (the HI map).

Loss terms:
    
# Dust 
L1 : (u + c) = d                    

# HI correlation
L2 : (u + c) x HI = d x HI

# CIB  
L3 : (d - u) = c 

# Dust CIB independence
L4 : u x (d - u) = s0 x c

''' 

#######
# INPUT PARAMETERS
#######

M, N = 700,700
J = 7
L = 4
dn = 5
pbc = False
alpha = 10 # L2 factor
beta = 60 # L3 factor
gamma = 200 # L4 factor

output_filename="separation_Magellan_HI_2steps_gamma=200_25_5_50.npy"

Mn = 50 # Number of noises per iteration

norm = "auto"   # Normalization

devices = [0,1] # List of GPUs to use

optim_params0 = {"maxiter": 25, "gtol": 1e-14, "ftol": 1e-14, "maxcor": 20}
optim_params1 = {"maxiter": 5, "gtol": 1e-14, "ftol": 1e-14, "maxcor": 20}
optim_params2 = {"maxiter": 50, "gtol": 1e-14, "ftol": 1e-14, "maxcor": 20}

###############################################################################
# DATA
###############################################################################

# Dust 
d = np.load('../Magellan_separation/Dust_Galactic.npy').astype(np.float32)

# CIB
c = np.load('../Magellan_separation/CIB.npy').astype(np.float32)
c_syn = np.load('../Magellan_separation/100_CIB.npy').astype(np.float32)[:Mn]

# HI map
HI_gal = np.load('../Magellan_separation/HI_Galactic.npy').astype(np.float32)
HI_mag = np.load('../Magellan_separation/HI_Magellan.npy').astype(np.float32)
HI = HI_gal + HI_mag

#######
# SEPARATION
#######
    
def objective_per_gpu_first(u, coeffs_target, wph_op, work_list, device_id):
    """
        Compute part of the loss and of the corresponding gradient on the target device (device_id).
    """
    # Select GPU and move data in its corresponding memory
    device = devices[device_id]
    wph_op.to(device)
    
    coeffs_target = coeffs_target.to(device_id)
    
    # Select work_list for device
    work_list = work_list[device_id]
    
    # Track operations on u
    u = torch.from_numpy(u).to(device).requires_grad_(True)
    
    # Define the CIB maps
    c_syn_gpu = torch.from_numpy(c_syn[work_list]).to(device)
    
    # Compute the loss
    loss_tot = torch.zeros(1)
    for i in range(len(work_list)):
        u_noisy, nb_chunks = wph_op.preconfigure(u + c_syn_gpu[i], pbc=pbc)
        for j in range(nb_chunks):
            coeffs_chunk, indices = wph_op.apply(u_noisy, j, norm=norm, ret_indices=True, pbc=pbc)
            loss = torch.sum(torch.abs(coeffs_chunk - coeffs_target[indices]) ** 2) / Mn
            loss.backward(retain_graph=True)
            loss_tot += loss.detach().cpu()
            del coeffs_chunk, indices, loss
        sys.stdout.flush() # Flush the standard output
    
    # Extract the corresponding gradient
    x_grad = u.grad.cpu().numpy()
    
    del c_syn_gpu, u # To free GPU memory
    
    return loss_tot.item(), x_grad

def objective_per_gpu_second(u, coeffs_target, wph_op, work_list, device_id):
    """
        Compute part of the loss and of the corresponding gradient on the target device (device_id).
    """
    # Select GPU and move data in its corresponding memory
    device = devices[device_id]
    wph_op.to(device)
    
    coeffs_tar_1 = coeffs_target[0].to(device_id)
    coeffs_tar_2 = coeffs_target[1].to(device_id)
    coeffs_tar_3 = coeffs_target[2].to(device_id)
    norm_map_1 = torch.from_numpy(coeffs_target[3]).to(device_id)
    norm_map_2 = torch.from_numpy(coeffs_target[4]).to(device_id)
    norm_map_3 = torch.from_numpy(coeffs_target[5]).to(device_id)
    
    # Select work_list for device
    work_list = work_list[device_id]
    
    # Track operations on u
    u = torch.from_numpy(u).to(device).requires_grad_(True)
    
    # Define the CIB maps
    c_syn_gpu = torch.from_numpy(c_syn[work_list]).to(device)
    
    # Compute L1
    wph_op.clear_normalization()
    wph_op.apply(norm_map_1, norm=norm, pbc=pbc)
    loss_tot1 = torch.zeros(1)
    for i in range(len(work_list)):
        u_noisy, nb_chunks = wph_op.preconfigure(u + c_syn_gpu[i], pbc=pbc)
        for j in range(nb_chunks):
            coeffs_chunk, indices = wph_op.apply(u_noisy, j, norm=norm, ret_indices=True, pbc=pbc)
            loss = torch.sum(torch.abs(coeffs_chunk - coeffs_tar_1[indices]) ** 2) / Mn
            loss.backward(retain_graph=True)
            loss_tot1 += loss.detach().cpu()
            del coeffs_chunk, indices, loss
        sys.stdout.flush() # Flush the standard output
        
    # Compute L2
    wph_op.clear_normalization()
    wph_op.apply([norm_map_2,norm_map_1], norm=norm, pbc=pbc, cross=True)
    loss_tot2 = torch.zeros(1)
    for i in range(len(work_list)):
        u_noisy, nb_chunks = wph_op.preconfigure([torch.from_numpy(HI).to(device),u + c_syn_gpu[i]], pbc=pbc, cross=True)
        for j in range(nb_chunks):
            coeffs_chunk, indices = wph_op.apply(u_noisy, j, norm=norm, ret_indices=True, pbc=pbc, cross=True)
            loss = torch.sum(torch.abs(coeffs_chunk - coeffs_tar_2[indices]) ** 2) / Mn
            loss = loss*alpha
            loss.backward(retain_graph=True)
            loss_tot2 += loss.detach().cpu()
            del coeffs_chunk, indices, loss
        sys.stdout.flush() # Flush the standard output
    
    # Compute L3
    wph_op.clear_normalization()
    wph_op.apply(norm_map_3, norm=norm, pbc=pbc)
    loss_tot3 = torch.zeros(1)
    u_noisy, nb_chunks = wph_op.preconfigure(torch.from_numpy(d).to(device) - u, pbc=pbc)
    for j in range(nb_chunks):
        coeffs_chunk, indices = wph_op.apply(u_noisy, j, norm=norm, ret_indices=True, pbc=pbc)
        loss = torch.sum(torch.abs(coeffs_chunk - coeffs_tar_3[indices]) ** 2) / Mn
        loss = loss*beta
        loss.backward(retain_graph=True)
        loss_tot3 += loss.detach().cpu()
        del coeffs_chunk, indices, loss
    sys.stdout.flush() # Flush the standard output
    
    if device == 0:
        print("L1 =",loss_tot1)
        print("L2 =",loss_tot2)
        print("L3 =",loss_tot3)
        
    # Total loss
    Total_loss = loss_tot1 + loss_tot2 + loss_tot3
    
    # Extract the corresponding gradient
    x_grad = u.grad.cpu().numpy()
    
    del c_syn_gpu, u # To free GPU memory
    
    return Total_loss.item(), x_grad

def objective_per_gpu_third(u, coeffs_target, wph_op, work_list, device_id):
    """
        Compute part of the loss and of the corresponding gradient on the target device (device_id).
    """
    # Select GPU and move data in its corresponding memory
    device = devices[device_id]
    wph_op.to(device)
    
    coeffs_tar_1 = coeffs_target[0].to(device_id)
    coeffs_tar_2 = coeffs_target[1].to(device_id)
    coeffs_tar_3 = coeffs_target[2].to(device_id)
    coeffs_tar_4 = coeffs_target[3].to(device_id)
    norm_map_1 = torch.from_numpy(coeffs_target[4]).to(device_id)
    norm_map_2 = torch.from_numpy(coeffs_target[5]).to(device_id)
    norm_map_3 = torch.from_numpy(coeffs_target[6]).to(device_id)
    
    # Select work_list for device
    work_list = work_list[device_id]
    
    # Track operations on u
    u = torch.from_numpy(u).to(device).requires_grad_(True)
    
    # Define the CIB maps
    c_syn_gpu = torch.from_numpy(c_syn[work_list]).to(device)
    
    # Compute L1
    wph_op.clear_normalization()
    wph_op.apply(norm_map_1, norm=norm, pbc=pbc)
    loss_tot1 = torch.zeros(1)
    for i in range(len(work_list)):
        u_noisy, nb_chunks = wph_op.preconfigure(u + c_syn_gpu[i], pbc=pbc)
        for j in range(nb_chunks):
            coeffs_chunk, indices = wph_op.apply(u_noisy, j, norm=norm, ret_indices=True, pbc=pbc)
            loss = torch.sum(torch.abs(coeffs_chunk - coeffs_tar_1[indices]) ** 2) / Mn
            loss.backward(retain_graph=True)
            loss_tot1 += loss.detach().cpu()
            del coeffs_chunk, indices, loss
        sys.stdout.flush() # Flush the standard output
        
    # Compute L2
    wph_op.clear_normalization()
    wph_op.apply([norm_map_2,norm_map_1], norm=norm, pbc=pbc, cross=True)
    loss_tot2 = torch.zeros(1)
    for i in range(len(work_list)):
        u_noisy, nb_chunks = wph_op.preconfigure([torch.from_numpy(HI).to(device),u + c_syn_gpu[i]], pbc=pbc, cross=True)
        for j in range(nb_chunks):
            coeffs_chunk, indices = wph_op.apply(u_noisy, j, norm=norm, ret_indices=True, pbc=pbc, cross=True)
            loss = torch.sum(torch.abs(coeffs_chunk - coeffs_tar_2[indices]) ** 2) / Mn
            loss = loss*alpha
            loss.backward(retain_graph=True)
            loss_tot2 += loss.detach().cpu()
            del coeffs_chunk, indices, loss
        sys.stdout.flush() # Flush the standard output
    
    # Compute L3
    wph_op.clear_normalization()
    wph_op.apply(norm_map_3, norm=norm, pbc=pbc)
    loss_tot3 = torch.zeros(1)
    u_noisy, nb_chunks = wph_op.preconfigure(torch.from_numpy(d).to(device) - u, pbc=pbc)
    for j in range(nb_chunks):
        coeffs_chunk, indices = wph_op.apply(u_noisy, j, norm=norm, ret_indices=True, pbc=pbc)
        loss = torch.sum(torch.abs(coeffs_chunk - coeffs_tar_3[indices]) ** 2) / Mn
        loss = loss*beta
        loss.backward(retain_graph=True)
        loss_tot3 += loss.detach().cpu()
        del coeffs_chunk, indices, loss
    sys.stdout.flush() # Flush the standard output
    
    # Compute L4
    wph_op.clear_normalization()
    wph_op.apply([norm_map_1,norm_map_3], norm=norm, pbc=pbc, cross=True)
    loss_tot4 = torch.zeros(1)
    u_noisy, nb_chunks = wph_op.preconfigure([u,torch.from_numpy(d).to(device) - u], pbc=pbc, cross=True)
    for j in range(nb_chunks):
        coeffs_chunk, indices = wph_op.apply(u_noisy, j, norm=norm, ret_indices=True, pbc=pbc, cross=True)
        loss = torch.sum(torch.abs(coeffs_chunk - coeffs_tar_4[indices]) ** 2) / Mn
        loss = loss*gamma
        loss.backward(retain_graph=True)
        loss_tot4 += loss.detach().cpu()
        del coeffs_chunk, indices, loss
    sys.stdout.flush() # Flush the standard output
    
    if device == 0:
        print("L1 =",loss_tot1)
        print("L2 =",loss_tot2)
        print("L3 =",loss_tot3)
        print("L4 =",loss_tot4)
        
    # Total loss
    Total_loss = loss_tot1 + loss_tot2 + loss_tot3 + loss_tot4
    
    # Extract the corresponding gradient
    x_grad = u.grad.cpu().numpy()
    
    del c_syn_gpu, u # To free GPU memory
    
    return Total_loss.item(), x_grad

def objective_first(x):
    
    global eval_cnt
    print("Evaluation : {:}".format(eval_cnt))
    start_time = time.time()
    
    # Reshape u
    u = x.reshape((M, N))
    
    # Multi-gpu computation
    nb_processes = len(devices)
    work_list = np.array_split(np.arange(Mn), nb_processes)
    pool = mp.get_context("spawn").Pool(processes=nb_processes) # "spawn" context demanded by CUDA
    closure_per_gpu_loc = partial(objective_per_gpu_first, u, COEFFS0, wph_op, work_list)
    results = pool.map(closure_per_gpu_loc, range(nb_processes))
    
    # Get results and close pool
    loss_tot = 0.0
    grad_tot = np.zeros_like(x.reshape(M, N))
    for i in range(len(results)):
        loss, grad = results[i]
        loss_tot += loss
        grad_tot += grad
    pool.close()
    
    print("L1 = {:} (computed in {:}s)".format(loss_tot,time.time() - start_time))
    eval_cnt += 1
    return loss_tot, grad_tot.ravel()

def objective_second(x):
    
    global eval_cnt
    print("Evaluation : {:}".format(eval_cnt))
    start_time = time.time()
    
    # Reshape u
    u = x.reshape((M, N))
    
    # Multi-gpu computation
    nb_processes = len(devices)
    work_list = np.array_split(np.arange(Mn), nb_processes)
    pool = mp.get_context("spawn").Pool(processes=nb_processes) # "spawn" context demanded by CUDA
    closure_per_gpu_loc = partial(objective_per_gpu_second, u, COEFFS1, wph_op, work_list)
    results = pool.map(closure_per_gpu_loc, range(nb_processes))
    
    # Get results and close pool
    loss_tot = 0.0
    grad_tot = np.zeros_like(x.reshape(M, N))
    for i in range(len(results)):
        loss, grad = results[i]
        loss_tot += loss
        grad_tot += grad
    pool.close()
    
    print("L = {:} (computed in {:}s)".format(loss_tot,time.time() - start_time))
    eval_cnt += 1
    return loss_tot, grad_tot.ravel()

def objective_third(x):
    
    global eval_cnt
    print("Evaluation : {:}".format(eval_cnt))
    start_time = time.time()
    
    # Reshape u
    u = x.reshape((M, N))
    
    # Multi-gpu computation
    nb_processes = len(devices)
    work_list = np.array_split(np.arange(Mn), nb_processes)
    pool = mp.get_context("spawn").Pool(processes=nb_processes) # "spawn" context demanded by CUDA
    closure_per_gpu_loc = partial(objective_per_gpu_third, u, COEFFS2, wph_op, work_list)
    results = pool.map(closure_per_gpu_loc, range(nb_processes))
    
    # Get results and close pool
    loss_tot = 0.0
    grad_tot = np.zeros_like(x.reshape(M, N))
    for i in range(len(results)):
        loss, grad = results[i]
        loss_tot += loss
        grad_tot += grad
    pool.close()
    
    print("L = {:} (computed in {:}s)".format(loss_tot,time.time() - start_time))
    eval_cnt += 1
    return loss_tot, grad_tot.ravel()


if __name__ == "__main__":
    
    ###############################
    # FIRST STEP
    ###############################
    
    print("Building operator for first step...")
    start_time = time.time()
    wph_op = pw.WPHOp(M, N, J, L=L, dn=dn, device=devices[0])
    print("Done ! (in {:}s)".format(time.time() - start_time))
    
    print("Computing stats of target image...")
    start_time = time.time()
    wph_op.load_model(["S11","L"])
    COEFFS0 = wph_op.apply(d, norm=norm, pbc=pbc).to("cpu")
    wph_op.to("cpu") # Move back to CPU before the actual denoising algorithm
    if torch.cuda.is_available():
        torch.cuda.empty_cache() # Empty the memory cache to clear devices[0] memory
    print(f"Done! (in {time.time() - start_time}s)")
    
    ## Minimization
    
    eval_cnt = 0
    
    # We perform a minimization of the objective function, using the noisy map as the initial map
    x0 = d
    result = opt.minimize(objective_first, x0.ravel(), method='L-BFGS-B', jac=True, tol=None, options=optim_params0)
    final_loss, s0_tilde, niter, msg = result['fun'], result['x'], result['nit'], result['message']
    s0_tilde = s0_tilde.reshape((M, N)).astype(np.float32) # Reshaping
    print(f"First step of denoising ended in {niter} iterations with optimizer message: {msg}")

    ###############################
    # SECOND STEP
    ###############################
    
    print("Renormalizing for second step...")
    start_time = time.time()
    wph_op.clear_normalization()
    wph_op.to(devices[0])
    print("Computing stats of target image...")
    
    norm1 = s0_tilde
    norm2 = HI
    norm3 = c
    
    print("Computing stats of target image...")
    start_time = time.time()
    wph_op.load_model(["S11","S00","S01","S10","Cphase","C01","C10","C00","L"])
    wph_op.apply(norm1, norm=norm, pbc=pbc).to("cpu")
    coeffs_1 = wph_op.apply(d, norm=norm, pbc=pbc).to("cpu")
    wph_op.clear_normalization()
    wph_op.apply([norm2,norm1], norm=norm, pbc=pbc, cross=True).to("cpu")
    coeffs_2 = wph_op.apply([HI,d], norm=norm, pbc=pbc, cross=True).to("cpu")
    wph_op.clear_normalization()
    wph_op.apply(norm3, norm=norm, pbc=pbc).to("cpu")
    coeffs_3 = wph_op.apply(c, norm=norm, pbc=pbc).to("cpu")
    COEFFS1 = [coeffs_1,coeffs_2,coeffs_3,norm1,norm2,norm3]
    wph_op.to("cpu") # Move back to CPU before the actual denoising algorithm
    if torch.cuda.is_available():
        torch.cuda.empty_cache() # Empty the memory cache to clear devices[0] memory
    print("Done! (in {:}s)".format(time.time() - start_time))
    
    ## Minimization
    eval_cnt = 0
    
    # We perform a minimization of the objective function, using the noisy map as the initial map
    x0 = d
    result = opt.minimize(objective_second, x0.ravel(), method='L-BFGS-B', jac=True, tol=None, options=optim_params1)
    final_loss, s1_tilde, niter, msg = result['fun'], result['x'], result['nit'], result['message']
    s1_tilde = s1_tilde.reshape((M, N)).astype(np.float32) # Reshaping
    print(f"Second step of denoising ended in {niter} iterations with optimizer message: {msg}")
    
    ###############################
    # THIRD STEP
    ###############################
    
    print("Renormalizing for third step...")
    start_time = time.time()
    wph_op.clear_normalization()
    wph_op.to(devices[0])
    print("Computing stats of target image...")
    
    norm1 = s0_tilde
    norm2 = HI
    norm3 = c
    
    print("Computing stats of target image...")
    start_time = time.time()
    wph_op.load_model(["S11","S00","S01","S10","Cphase","C01","C10","C00","L"])
    wph_op.apply(norm1, norm=norm, pbc=pbc).to("cpu")
    coeffs_1 = wph_op.apply(d, norm=norm, pbc=pbc).to("cpu")
    wph_op.clear_normalization()
    wph_op.apply([norm2,norm1], norm=norm, pbc=pbc, cross=True).to("cpu")
    coeffs_2 = wph_op.apply([HI,d], norm=norm, pbc=pbc, cross=True).to("cpu")
    wph_op.clear_normalization()
    wph_op.apply(norm3, norm=norm, pbc=pbc).to("cpu")
    coeffs_3 = wph_op.apply(c, norm=norm, pbc=pbc).to("cpu")
    wph_op.clear_normalization()
    wph_op.apply([norm1,norm3], norm=norm, pbc=pbc, cross=True).to("cpu")
    coeffs_4 = wph_op.apply([s0_tilde,c], norm=norm, pbc=pbc, cross=True).to("cpu")
    COEFFS2 = [coeffs_1,coeffs_2,coeffs_3,coeffs_4,norm1,norm2,norm3]
    wph_op.to("cpu") # Move back to CPU before the actual denoising algorithm
    if torch.cuda.is_available():
        torch.cuda.empty_cache() # Empty the memory cache to clear devices[0] memory
    print("Done! (in {:}s)".format(time.time() - start_time))
    
    ## Minimization
    eval_cnt = 0
    
    # We perform a minimization of the objective function, using the noisy map as the initial map
    x0 = s1_tilde
    result = opt.minimize(objective_third, x0.ravel(), method='L-BFGS-B', jac=True, tol=None, options=optim_params2)
    final_loss, s_tilde, niter, msg = result['fun'], result['x'], result['nit'], result['message']
    s_tilde = s_tilde.reshape((M, N)).astype(np.float32) # Reshaping
    print(f"Second step of denoising ended in {niter} iterations with optimizer message: {msg}")

    ###############################
    # END
    ###############################
    
    print("Denoising ended in {:} iterations with optimizer message: {:}".format(niter,msg))
    
    if output_filename is not None:
        np.save(output_filename, [d,HI,c,s_tilde,d-s_tilde,s0_tilde,s1_tilde])
