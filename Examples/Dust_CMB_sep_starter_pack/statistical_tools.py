import numpy as np
import torch
import matplotlib.pyplot as plt
import time
import pywph as pw
import pywst
from pywavan import powspec
import scipy.optimize as opt
from cycler import cycler
import scipy.interpolate as interp
import warnings
from astropy.io import fits
from astropy.coordinates import SkyCoord
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from astropy.wcs import WCS
#import sys
#sys.path.append('/home/auclair/Thèse_LPENS/Informatique/Packages/')
#import scattering as scat
from copy import deepcopy
import pickle
import pandas as pd
import pymaster as nmt
import healpy as hp

warnings.filterwarnings("ignore")
plt.rcParams['text.usetex'] = True
color = ['#377EB8', '#FF7F00', '#4DAF4A','#F781BF', '#A65628', '#984EA3','#999999', '#E41A1C', '#DEDE00']
style = ['-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-']
default_cycler = (cycler(color=['#377EB8','#FF7F00','#4DAF4A','#FF7F00','#4DAF4A']))
plt.rc('axes', prop_cycle=default_cycler)

def gmean(x,axis=0):
    return np.exp(np.mean(np.log(x),axis=axis))

def gstd(x,axis=0):
    return np.exp(np.std(np.log(x),axis=axis))

def plot(image,cmap='inferno',logscale=False,ampcol=3,size=6,unit=r'$\rm{I \ [MJy/sr]}$',title='',x_title=0.45,y_title=0.85,colorbar=True,no_show=False):
    plt.figure(figsize=(size,size))
    if logscale == True:
        image = np.log(image)
    plt.imshow(image,cmap=cmap,origin='lower',vmin=image.mean()-ampcol*image.std(),vmax=image.mean()+ampcol*image.std())
    plt.suptitle(title,x=x_title,y=y_title,fontsize=20)
    if colorbar:
        cb = plt.colorbar(shrink=0.8,pad=0.05)
        cb.set_label(unit,rotation=270,fontsize=20,labelpad=15)
    if not no_show:
        plt.show()
    return 
    
def apodize(na, nb, radius):
    na = int(na)
    nb = int(nb)
    ni = int(radius * na)
    nj = int(radius * nb)
    dni = na - ni
    dnj = nb - nj
    tap1d_x = np.zeros(na) + 1.0
    tap1d_y = np.zeros(nb) + 1.0
    tap1d_x[:dni] = (np.cos(3*np.pi/2.+np.pi/2.*(1.*np.linspace(0,dni-1,dni)/(dni-1)) ))
    tap1d_x[na-dni:] = (np.cos(0.+np.pi/2.*(1.*np.linspace(0,dni-1,dni)/(dni-1)) ))
    tap1d_y[:dnj] = (np.cos(3*np.pi/2.+np.pi/2.*(1.*np.linspace(0,dnj-1,dnj)/(dnj-1)) ))
    tap1d_y[nb-dnj:] = (np.cos(0.+np.pi/2.*(1.*np.linspace(0,dnj-1,dnj)/(dnj-1)) ))
    tapper = np.zeros([na,nb])
    for i in range(nb):
        tapper[:,i] = tap1d_x
    for i in range(na):
        tapper[i,:] = tapper[i,:] * tap1d_y
    return tapper

def phase_random(x):
    [M,N] = np.shape(x) # M and N must be multiples of 2
    ft_x = np.fft.fftshift(np.fft.fft2(x))
    mod_ft_x = np.abs(ft_x)
    gauss = np.random.normal(0,1,size=(M,N))
    ft_gauss = np.fft.fftshift(np.fft.fft2(gauss))
    angle_ft_gauss = np.angle(ft_gauss)
    new_ft_x = mod_ft_x * np.exp(1j*angle_ft_gauss)
    rand_x = np.fft.ifft2(np.fft.ifftshift(new_ft_x))
    return np.real(rand_x)

def plot_PS(images,labels,colors=color,styles=style,reso=1,apo=0.95,N_bin=100,a=4,cross=None,ret_PS=False,abs_cross=True,axis='k',fontsize=20,plot=True):
    # cross has to be a list of ["index 1","index 2","color","style"]
    if axis == 'k':
        k_to_l = 1
    if axis == 'l':
        k_to_l = 2 * 60 * 180
    if len(np.shape(images)) == 2:
        (M,N) = np.shape(images)
        n = 1
        t = 1
    if len(np.shape(images)) == 3:
        (n,M,N) = np.shape(images)
        t = 1
    if len(np.shape(images)) == 4:
        (t,n,M,N) = np.shape(images)
    if len(np.shape(images)) != 2 and len(np.shape(images)) != 3 and len(np.shape(images)) != 4:
        print("Invalid data shape !")
    if t == 1:
        # PS computation
        if n == 1:
            PS = powspec(images,reso=reso,apo=apo,N_bin=N_bin,a=a)
        else:
            PS = []
            for i in range(n):
                PS.append(powspec(images[i],reso=reso,apo=apo,N_bin=N_bin,a=a))
            if cross is not None:
                for i in range(len(cross)):
                    if not abs_cross:
                        PS.append(powspec(images[cross[i][0]],im2=images[cross[i][1]],reso=reso,apo=apo,N_bin=N_bin,a=a))
                    else:
                        PS.append(np.abs(powspec(images[cross[i][0]],im2=images[cross[i][1]],reso=reso,apo=apo,N_bin=N_bin,a=a)))
        PS = np.array(PS)
        # Plot
        if plot:
            plt.figure(figsize=(10,7))
            plt.loglog()
            if n == 1:
                plt.plot(PS[0]*k_to_l,PS[1],label=labels[0],color=colors[0],linestyle=styles[0])
            else:
                for i in range(n):
                    plt.plot(PS[i][0]*k_to_l,PS[i][1],label=labels[i],color=colors[i],linestyle=styles[i])
                if cross is not None:
                    for i in range(len(cross)):
                            plt.plot(PS[n+i][0]*k_to_l,PS[n+i][1],label=labels[cross[i][0]]+r' $\times$ '+labels[cross[i][1]],color=cross[i][2],linestyle=cross[i][3])
    if t > 1:
        # PS computation
        power_spectra = []
        for j in range(t):
            if n == 1:
                power_spectra.append(powspec(images[j,0],reso=reso,apo=apo,N_bin=N_bin,a=a))
            else:
                PS = []
                for i in range(n):
                    PS.append(powspec(images[j,i],reso=reso,apo=apo,N_bin=N_bin,a=a))
                if cross is not None:
                    for i in range(len(cross)):
                        if not abs_cross:
                            PS.append(powspec(images[j,cross[i][0]],im2=images[j,cross[i][1]],reso=reso,apo=apo,N_bin=N_bin,a=a))
                        else:
                            PS.append(np.abs(powspec(images[j,cross[i][0]],im2=images[j,cross[i][1]],reso=reso,apo=apo,N_bin=N_bin,a=a)))
                power_spectra.append(PS)
        power_spectra = np.array(power_spectra)
        # Plot
        if plot:
            plt.figure(figsize=(10,7))
            plt.loglog()
            if n == 1:
                plt.plot(power_spectra[0,0]*k_to_l,np.mean(power_spectra[:,1],axis=0),label=labels[0],color=colors[0],linestyle=styles[0])
                plt.fill_between(power_spectra[0,0]*k_to_l,np.mean(power_spectra[:,1],axis=0)-np.std(power_spectra[:,1],axis=0),np.mean(power_spectra[:,1],axis=0)+np.std(power_spectra[:,1],axis=0),color=colors[0],alpha=0.4)
            else:
                for i in range(n):
                    plt.plot(power_spectra[0,0,0]*k_to_l,np.mean(power_spectra[:,i,1],axis=0),label=labels[i],color=colors[i],linestyle=styles[i])
                    plt.fill_between(power_spectra[0,0,0]*k_to_l,np.mean(power_spectra[:,i,1],axis=0)-np.std(power_spectra[:,i,1],axis=0),np.mean(power_spectra[:,i,1],axis=0)+np.std(power_spectra[:,i,1],axis=0),color=colors[i],alpha=0.4)
                if cross is not None:
                    for i in range(len(cross)):
                        plt.plot(power_spectra[0,0,0]*k_to_l,np.mean(power_spectra[:,n+i,1],axis=0),label=labels[cross[i][0]]+r' $\times$ '+labels[cross[i][1]],color=cross[i][2],linestyle=cross[i][3])
                        plt.fill_between(power_spectra[0,0,0]*k_to_l,np.mean(power_spectra[:,n+i,1],axis=0)-np.std(power_spectra[:,n+i,1],axis=0),np.mean(power_spectra[:,n+i,1],axis=0)+np.std(power_spectra[:,n+i,1],axis=0),color=cross[i][2],alpha=0.4)
    if plot:
        plt.legend(prop={'size': fontsize})
        if axis == 'k':
            plt.xlabel(r'$\rm{k} \ [\rm{arcmin^{-1}}]$',fontsize=fontsize)
            plt.ylabel(r'$\rm{PS(k)} \ [\rm{Jy}^2/\rm{sr}]$',fontsize=fontsize)
        if axis == 'l':
            plt.xlabel('$l$',fontsize=fontsize)
            plt.ylabel('$C_l$ [$\mu K^2$]',fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.grid(True)
    if ret_PS:
        if t == 1:
            return PS[0,0]*k_to_l, PS[:,1], PS[:,1]*0
        if t > 1:
            return power_spectra[0,0,0]*k_to_l, np.mean(power_spectra[:,:,1],axis=0), np.std(power_spectra[:,:,1],axis=0)
    else:
        return
    
def plot_wph(images,labels=None,colors=None,styles=None,J=None,L=4,dn=0,pbc=False,ret_coeffs=False):
    if len(np.shape(images)) == 2:
        (M,N) = np.shape(images)
        n = 1
        t = 1
    if len(np.shape(images)) == 3:
        (n,M,N) = np.shape(images)
        t = 1
    if len(np.shape(images)) == 4:
        (t,n,M,N) = np.shape(images)
    if len(np.shape(images)) not in [2,3,4]:
        print("Invalid data shape !")
    if J==None:
        J = int(np.log2(min(M,N)) - 2)
    wph_op = pw.WPHOp(M, N, J, L=L, dn=dn, device="cpu")
    wph_model = ['S11','S00','S01','Cphase','C00','C01']
    wph_op.load_model(wph_model)
    wph = wph_op.apply(images,norm=None,ret_wph_obj=True,pbc=pbc).to_isopar()
    if t == 1:
        coeffs = []
        for i in range(len(wph_model)):
            coeffs.append(np.abs(wph.get_coeffs(wph_model[i])[0]))
    if t > 1:
        coeffs = []
        for i in range(len(wph_model)):
            coeff = np.abs(wph.get_coeffs(wph_model[i])[0])
            coeffs.append([np.mean(coeff,axis=0),np.std(coeff,axis=0)])
    plt.rcParams.update({'font.size': 15})
    fig, ax = plt.subplots(2, 3, figsize=(9, 7), sharex=False, sharey=False)
    for i in range(n):
        for j in range(len(wph_model)):
            if t == 1:
                if n==1:
                    ax[j//3,j%3].plot(coeffs[j])
                if n > 1:
                    ax[j//3,j%3].plot(coeffs[j][i],label=labels[i],color=colors[i],linestyle=styles[i])
            if t > 1:
                ax[j//3,j%3].plot(coeffs[j][0][i],label=labels[i],color=colors[i],linestyle=styles[i])
                ax[j//3,j%3].fill_between(np.arange(len(coeffs[j][0][i])),coeffs[j][0][i]-coeffs[j][1][i],coeffs[j][0][i]+coeffs[j][1][i],color=colors[i],alpha=0.4)
    ax[0,0].grid(False)
    ax[0,0].set_xlabel('$j_1$')
    ax[0,0].set_title('$S^{11}$')
    ax[0,0].set_yscale('log')
    ax[0,1].grid(False)
    ax[0,1].set_xlabel('$j_1$')
    ax[0,1].set_title('$S^{00}$')
    ax[0,2].grid(False)
    ax[0,2].set_xlabel('$j_1$')
    ax[0,2].set_title('$S^{01}$')
    if n>1:
        ax[0,2].legend(loc=3,bbox_to_anchor=(1.05,0.05))
    ax[1,1].grid(False)
    ax[1,1].set_xlabel('($j_1$,$j_2$,$\Delta l$)')
    ax[1,1].set_title('$C^{00}$')
    ax[1,2].grid(False)
    ax[1,2].set_xlabel('($j_1$,$j_2$,$\Delta l$)')
    ax[1,2].set_title('$C^{01}$')
    ax[1,0].grid(False)
    ax[1,0].set_xlabel('($j_1$,$j_2$)')
    ax[1,0].set_title('$C^{phase}$')
    plt.tight_layout()
    plt.show()
    if ret_coeffs==True:
        return coeffs
    else:
        return
    
def plot_hist(images,labels=None,colors=None,styles=None,n_bins=100,log=False,density=True,value_range=None,fontsize=15,nature=r'$\rm{I}$'):
    if len(np.shape(images)) == 2:
        images = np.array([images])
    (n,N,N) = np.shape(images)
    bins = []
    hist = []
    for i in range(n):
        h,b_edges = np.histogram(images[i], bins=n_bins, range=value_range, density=density, weights=None)
        bins.append((b_edges[:-1]+b_edges[1:])/2)
        hist.append(h)
    if labels is None:
        labels = np.arange(1,n+1).astype(str).tolist()
    if colors is None:
        colors = color[:n]
    if styles is None:
        styles = ['-']*n
    plt.figure(figsize=(10,6))
    for i in range(n):
        plt.plot(bins[i],hist[i],label=labels[i],color=colors[i],ls=styles[i])
    plt.legend(fontsize=fontsize)
    if log:
        plt.yscale('log')
    plt.ylabel(r'$\rm{p(}$'+nature+r'$\rm{)}$',fontsize=fontsize)
    plt.xlabel(nature,fontsize=fontsize)
    plt.show()
    return
    
def synthesis_wph(data,n_syn=1,n_step=50,print_loss=False,pbc=False,verbose=False,device='cpu',dn=2,tau_grid='exp'):
    image = np.copy(data)
    (M,N) = np.shape(image)
    J = int(np.log2(min(M,N)) - 2)
    L = 4
    norm = "auto" 
    optim_params = {"maxiter": n_step, "gtol": 1e-14, "ftol": 1e-14, "maxcor": 20}
    if not pbc:
        image_std = image[2**(J-1):-2**(J-1),2**(J-1):-2**(J-1)].std()
    else:
        image_std = image.std()
    image /= image_std
    if verbose:
        print("Building operator...")
    start_time = time.time()
    wph_op = pw.WPHOp(M ,N , J, L=L, dn=dn, device=device)
    if verbose:
        print(f"Done! (in {time.time() - start_time}s)")
    model=["S11","S00","S01","Cphase","C01","C00","L"]
    wph_op.load_model(model,tau_grid=tau_grid)
    if verbose:
        print("Computing stats of target image...")
    start_time = time.time()
    coeffs = wph_op.apply(image, norm=norm, pbc=pbc)
    if verbose:
        print(f"Done! (in {time.time() - start_time}s)")
    def objective(x):
        start_time = time.time()
        # Reshape x
        x_curr = x.reshape((M, N))
        # Compute the loss (squared 2-norm)
        loss_tot = torch.zeros(1)
        x_curr, nb_chunks = wph_op.preconfigure(x_curr, requires_grad=True, pbc=pbc)
        for i in range(nb_chunks):
            coeffs_chunk, indices = wph_op.apply(x_curr, i, norm=norm, ret_indices=True, pbc=pbc)
            loss = torch.sum(torch.abs(coeffs_chunk - coeffs[indices]) ** 2)
            loss.backward(retain_graph=True)
            loss_tot += loss.detach().cpu()
            del coeffs_chunk, indices, loss
        # Reshape the gradient
        x_grad = x_curr.grad.cpu().numpy().astype(x.dtype)
        if print_loss:
            print(f"Loss: {loss_tot.item()} (computed in {time.time() - start_time}s)")
        return loss_tot.item(), x_grad.ravel()
    
    total_start_time = time.time()
    
    if n_syn == 1:
        x0 = np.random.normal(image.mean(), image.std(), (M,N))
        x0 = torch.from_numpy(x0)
        result = opt.minimize(objective, x0.ravel(), method='L-BFGS-B', jac=True, tol=None, options=optim_params)
        _, x_final, niter, msg = result['fun'], result['x'], result['nit'], result['message']
        if verbose:
            print(f"Synthesis ended in {niter} iterations with optimizer message: {msg}")
            print(f"Synthesis time: {time.time() - total_start_time}s")
        x_final = x_final.reshape((M, N)).astype(np.float32)
        x_final = x_final * image_std
        return x_final
    if n_syn > 1:
        syntheses=np.zeros([n_syn,M,N])
        for i in range(n_syn):
            x0 = np.random.normal(image.mean(), image.std(), (M,N))
            x0 = torch.from_numpy(x0)
            result = opt.minimize(objective, x0.ravel(), method='L-BFGS-B', jac=True, tol=None, options=optim_params)
            _, x_final, niter, msg = result['fun'], result['x'], result['nit'], result['message']
            if verbose:
                print(f"Synthesis ended in {niter} iterations with optimizer message: {msg}")
                print(f"Synthesis time: {time.time() - total_start_time}s")
            x_final = x_final.reshape((M, N)).astype(np.float32)
            x_final = x_final * image_std
            syntheses[i]=x_final
        return syntheses
    
# def compute_S11(st_calc,x):
#     S11 = st_calc.scattering_cov_constant(x,only_S11=True)
#     if len(np.shape(x)) == 2:
#         return S11[0]
#     if len(np.shape(x)) == 3:
#         return S11
    
# def compute_mask(st_calc,x,threshold,norm=True):
#     return scat.compute_threshold_mask(st_calc.scattering_cov_constant(x,normalization=norm),threshold)
    
# def compute_coeffs(st_calc,x,mask,norm=False,use_ref=False):
#     coeffs = scat.threshold_coeffs(st_calc.scattering_cov_constant(x,normalization=norm,use_ref=use_ref),mask)
#     if len(np.shape(x)) == 2:
#         return coeffs[0]
#     if len(np.shape(x)) == 3:
#         return coeffs

# def synthesis_ScatCov(image,threshold,n_syn=1,n_step=50,device='cuda',print_loss=False,pbc=False,verbose=False,norm=True,use_ref=True):
#     torch.autograd.set_detect_anomaly(True)
#     (M,N) = np.shape(image)
#     J = int(np.log2(min(M,N)) - 1)
#     L = 4
#     if not torch.cuda.is_available(): device='cpu'
#     print(f"Device = {device}")
#     optim_params = {"maxiter": n_step, "gtol": 1e-14, "ftol": 1e-14, "maxcor": 20}
#     if verbose:
#         print("Building operator...")
#     start_time = time.time()
#     st_calc = scat.Scattering2d(M, N, J, L, device) 
#     if verbose:
#         print(f"Done! (in {time.time() - start_time}s)")
#         print("Computing stats of target image...")
#     start_time = time.time()
#     mask = compute_mask(st_calc,torch.from_numpy(image).to(device),threshold)
#     print(f"{mask.sum().item()} coefficients selected !")
#     st_calc.add_ref_constant(image)
#     coeffs = compute_coeffs(st_calc,torch.from_numpy(image).to(device),mask,norm=norm,use_ref=use_ref)
#     if verbose:
#         print(f"Done! (in {time.time() - start_time}s)")
#     def objective(x):
#         start_time = time.time()
#         # Reshape x
#         x_curr = x.reshape((M, N))
#         # Track operations on x
#         x_curr = torch.from_numpy(x_curr).to(device).requires_grad_(True)
#         loss_tot = torch.sum(torch.abs( compute_coeffs(st_calc,x_curr,mask,norm=norm,use_ref=use_ref) - coeffs ) ** 2)
#         loss_tot.backward(retain_graph=True)
#         # Reshape the gradient
#         x_grad = x_curr.grad.cpu().numpy().astype(x.dtype)
#         if print_loss:
#             print(f"Loss: {loss_tot.item()} (computed in {time.time() - start_time}s)")
#         return loss_tot.item(), x_grad.ravel()
    
#     total_start_time = time.time()
    
#     if n_syn == 1:
#         x0 = np.random.normal(image.mean(), image.std(), (M,N))
#         x0 = torch.from_numpy(x0)
#         result = opt.minimize(objective, x0.ravel(), method='L-BFGS-B', jac=True, tol=None, options=optim_params)
#         _, x_final, niter, msg = result['fun'], result['x'], result['nit'], result['message']
#         if verbose:
#             print(f"Synthesis ended in {niter} iterations with optimizer message: {msg}")
#             print(f"Synthesis time: {time.time() - total_start_time}s")
#         x_final = x_final.reshape((M, N)).astype(np.float32)
#         return x_final
#     if n_syn > 1:
#         syntheses=np.zeros([n_syn,M,N])
#         for i in range(n_syn):
#             x0 = np.random.normal(image.mean(), image.std(), (M,N))
#             x0 = torch.from_numpy(x0)
#             result = opt.minimize(objective, x0.ravel(), method='L-BFGS-B', jac=True, tol=None, options=optim_params)
#             _, x_final, niter, msg = result['fun'], result['x'], result['nit'], result['message']
#             if verbose:
#                 print(f"Synthesis ended in {niter} iterations with optimizer message: {msg}")
#                 print(f"Synthesis time: {time.time() - total_start_time}s")
#             x_final = x_final.reshape((M, N)).astype(np.float32)
#             syntheses[i]=x_final
#         return syntheses
    
### ERWAN'S FUNCTIONS

def plot_rC_WPH(map_pairs,labels=None,colors=None,styles=None,fontsize=15,one_scale=False):
    plt.rcParams.update({'font.size': fontsize})
    n_pairs,_,M,N = np.shape(map_pairs)
    if labels == None:
        labels = np.linspace(1,n_pairs,n_pairs).astype(int).astype(str).tolist()
    if colors == None:
        colors = ['#377EB8', '#FF7F00', '#4DAF4A','#F781BF', '#A65628', '#984EA3','#999999', '#E41A1C', '#DEDE00'][:n_pairs]
    if styles == None:
        styles = ['-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-',][:n_pairs]
    J = int(np.log2(M)-2)
    L = 4
    dn = 0
    full_angle = True
    device = "cpu"
    pbc = False
    wavelet = "Morlet"
    wph_model = ['C01','C10']
    wph_op = prep_operator(M=M, N=N, J=J, L=L, dn=dn, wavelet=wavelet, full_angle=full_angle, device=device, wph_model=wph_model)
    wph_list = []
    for i in range(n_pairs):
        wph_list.append(WPH_comp(Compute_WPH(map_pairs[i,0], mapB=map_pairs[i,1], operator=wph_op, pbc=pbc), comp='iso'))
    C01 = []
    C10 = []
    for i in range(n_pairs):
        coeffs_C01, ind_C01 = wph_list[i].get_coeffs("C01")
        coeffs_C10, ind_C10 = wph_list[i].get_coeffs("C10")
        if one_scale:
            C01.append(coeffs_C01[np.where(ind_C01[:,0]==ind_C01[:,3])])
            C10.append(coeffs_C10[np.where(ind_C10[:,0]==ind_C10[:,3])])
        else:
            C01.append(coeffs_C01)
            C10.append(coeffs_C10)
    C01 = np.real(np.array(C01))
    C10 = np.real(np.array(C10))
    rC01 = []
    rC10 = []
    for i in range(n_pairs):
        rC01.append(C_FullToReduce(C01[i], L=4))
        rC10.append(C_FullToReduce(C10[i], L=4))
    rC01 = np.array(rC01)
    rC10 = np.array(rC10)
    mean_rC01 = rC01[:,0,:]
    amp_rC01 = rC01[:,1,:]
    ang_rC01 = rC01[:,2,:]
    mean_rC10 = rC10[:,0,:]
    amp_rC10 = rC10[:,1,:]
    ang_rC10 = rC10[:,2,:]
    fig, ax = plt.subplots(2, 3, figsize=(12, 8))
    for i in range(n_pairs):
        ax[0,0].plot(mean_rC01[i],label=labels[i],color=colors[i],linestyle=styles[i])
        ax[0,1].plot(amp_rC01[i],label=labels[i],color=colors[i],linestyle=styles[i])
        ax[0,2].plot(ang_rC01[i],label=labels[i],color=colors[i],linestyle=styles[i])
        ax[1,0].plot(mean_rC10[i],label=labels[i],color=colors[i],linestyle=styles[i])
        ax[1,1].plot(amp_rC10[i],label=labels[i],color=colors[i],linestyle=styles[i])
        ax[1,2].plot(ang_rC10[i],label=labels[i],color=colors[i],linestyle=styles[i])
    ax[0,0].grid(False)
    ax[0,0].set_title(r'\rm{Mean} $C^{01}$')
    ax[0,1].grid(False)
    ax[0,1].set_title(r'\rm{Amplitude} $C^{01}$')
    ax[0,2].grid(False)
    ax[0,2].set_title(r'\rm{Phase} $C^{01}$')
    ax[0,2].legend(loc='best')
    ax[1,0].grid(False)
    ax[1,0].set_title(r'\rm{Mean} $C^{10}$')
    ax[1,1].grid(False)
    ax[1,1].set_title(r'\rm{Amplitude} $C^{10}$')
    ax[1,2].grid(False)
    ax[1,2].set_title(r'\rm{Phase} $C^{10}$')
    for i in range(2):
        for j in range(3):
            if one_scale:
                ax[i,j].set_xlabel('$j_1$')
            else:
                ax[i,j].set_xlabel('($j_1$,$j_2$)')
    fig.tight_layout()
    return

def wph_object_to_dataframe(wph_object, mapname='mapA', attrs={}):
    """
    This is a basic function to take a WPH object and put the coefficients and indices
    into a queryable pandas dataframe. 
    
    Input : wph_object : Assumes WHP coefficient order [j1, theta1, p1, j2, theta2, p2, n, alpha, pseudo]
    Output: pandas dataframe containing WPH coefficients and their indices
    
    mapname : succinct name of map
    attrs   : dictionary of any other metadata that should be stored as attributes. For instance:
                {'DEC':70, 'Note':'NPIPE A map'}
    """
    
    # Coefficient labels, with order taken from the WPH object
    coefflabels = ["j1", "t1", "p1", "j2", "t2", "p2", "n", "a", "pseudo"]
    
    # Put coefficient indices into a dictionary
    coeff_index_dict = {}
    for (_i, label) in enumerate(coefflabels):
        coeff_index_dict[label] = wph_object.wph_coeffs_indices[:, _i]
        
    # Put into dataframe
    df = pd.DataFrame({**{'coeff':wph_object.wph_coeffs}, **coeff_index_dict})
    
    # Assign types of coefficients
    df["type"] = ""
    
    # S11: j1 == j2, p0, p1 = 1. A
    df.loc[(df.j1 == df.j2) & (df.p1 == 1) & (df.p2 == 1) & (df.pseudo == 0), ["type"]] = "S11"
    
    # C01: p0, p1 = 1. And so forth.
    df.loc[(df.p1 == 0) & (df.p2 == 0), ["type"]] = "C00"
    df.loc[(df.p1 == 0) & (df.p2 == 1), ["type"]] = "C01"
    df.loc[(df.p1 == 1) & (df.p2 == 0), ["type"]] = "C10"
    
    df.map = mapname
    df.L = wph_object.L
    df.J = wph_object.J
    df.attrs = attrs
    
    return df


def prep_operator(M=512, N=512, J=6, L=4, dn=0, wavelet="Morlet", full_angle=False, device="cpu", wph_model = ['S11','S00','S01','C00','C01','C10']):
    """
    Prepares the WPH operator.
    """
    wph_op = pw.WPHOp(M, N, J, L=L, dn=dn, bp_filter_cls=wavelet, full_angle=full_angle, device=device)
    wph_op.load_model(wph_model)
    return wph_op

def Compute_WPH(mapA, mapB = None, operator=None, norm=None, pbc=False):
    """ 
    Compute a single WPH object from auto or cross (if mapB is given)
    operator: Takes a WPH operator defined by prep_operator
    
    Note that the pseudo S11 coefficients are removed
    """ 
    if mapB is None:
        wph = operator.apply(mapA, norm=norm, ret_wph_obj=True, pbc=pbc)
    else:
        wph = operator.apply([mapA, mapB], cross=True, norm=norm, ret_wph_obj=True, pbc=pbc)
        
    # remove pseudo S11 coefficients
    if wph.wph_coeffs.ndim == 1:
        wph.wph_coeffs = wph.wph_coeffs[wph.wph_coeffs_indices[:,-1]==0]
    else:
        wph.wph_coeffs = wph.wph_coeffs[:,wph.wph_coeffs_indices[:,-1]==0]
    wph.wph_coeffs_indices = wph.wph_coeffs_indices[wph.wph_coeffs_indices[:,-1]==0]
    
    return wph

def Compute_WPH_AB(mapA, mapB, operator=None, norm=None, pbc=False):
    """ 
    Compute auto and cross wph from 2 maps
    Return a list of 3 wph objects (wphA, wphB, wphAB) 
    """
    return [Compute_WPH(mapA, operator=operator, norm=norm, pbc=pbc), Compute_WPH(mapB, operator=operator, norm=norm, pbc=pbc), Compute_WPH(mapA, mapB, operator=operator, norm=norm, pbc=pbc)]


def get_norm_vector(wph, norm="S11"):
    """ Get normalization vector from wph object
    Can be either S00 ("S00"), S11 ("S11"), dim J*2L
    Or concatenate [S00,S11] ("S01"), dim (2,J*2L)
    Coefficients are indexed as (2L)*j + theta
    
    Input : single map wph_object (assume 2L angles for S coefficients)
    Output: normalization vector from S11 and S00 coefficient
    
    norm: "S11", "S00" or "S01", related to normalization scheme
    """
    J, L = wph.J, wph.L
    if norm == "S00" or norm == "S11":
        s, s_indices = wph.get_coeffs(norm)
        norm_vector = np.empty(J*2*L)
        for i in range(s.shape[0]):
            if (s_indices[i,-3:] == [0,0,0]).all():
                ind = s_indices[i,0]*2*L + s_indices[i,1] # 2L*j + theta
                norm_vector[ind] = np.real(s[i])
    elif norm == "S01":
        nv_S00 = get_norm_vector(wph,norm="S00")
        nv_S11 = get_norm_vector(wph,norm="S11")
        norm_vector = np.concatenate((nv_S00[None,:], nv_S11[None,:]), axis=0)
        
    return norm_vector

def normalize_wph(wph, norm="S11", wph_ref1=None, wph_ref2=None):
    """ normalize a wph object, with norm "S11", "S00", or "S01"
    Can be normalized by 1 or 2 ref wph objects, for j1 and j2 resp.
    Is normalized by itself if no ref is given
    
    Input: wph object from one single map
    Output: normalized wph object
    
    norm: "S11", "S00" or "S01", related to normalization scheme
    df_wph_refi: reference df_wph object from one or two maps
    
    """
    L = wph.L
    if wph_ref1 is None: 
        wph_ref1 = wph
    
    # Get norm vectors from S11
    norm_vector1 = get_norm_vector(wph_ref1, norm=norm)
    if wph_ref2 is not None:
        norm_vector2 = get_norm_vector(wph_ref2, norm=norm)
    else:
        norm_vector2 = norm_vector1
        
    # Get list of normalization index
    indices = wph.wph_coeffs_indices
    ind1 = list(indices[:,0]*2*L + indices[:,1]) # 2L*j1 + theta1
    ind2 = list(indices[:,3]*2*L + indices[:,4]) # 2L*j2 + theta2
    if norm == "S01": # Recover the S00 or S11 norm type for each ind
        ind01_1 = indices[:,2]
        ind01_1[ind01_1 > 1] = 1 # Normalisation is with S11 for p>1
        ind01_1 = list(ind01_1)
        ind01_2 = indices[:,5]
        ind01_2[ind01_2 > 1] = 1
        ind01_2 = list(ind01_2)
        
    # Compute normalized wph
    wph_norm = deepcopy(wph)
    if norm == "S00" or norm == "S11":
        wph_norm.wph_coeffs /= np.sqrt(norm_vector1[ind1] * norm_vector2[ind2])
    elif norm == "S01":
        wph_norm.wph_coeffs /= np.sqrt(norm_vector1[ind01_1,ind1] * norm_vector2[ind01_2,ind2])
    
    return wph_norm

def Normalize_trio(trio_wph, norm="S11"):
    """
    Normalize a trio of 3 wph objects (wphA, wphB, wphAB)
    wphA and wphB are self-normalized
    wphAB is normalized by wphA x wphB
    
    Input: List of 3 wph objects (wphA, wphB, wphAB)
    Output: Normalized list of wph object (wphA, wphB, wphAB)
    """
    # 
    wphA, wphB, wphAB = trio_wph[0], trio_wph[1], trio_wph[2]
    
    # Normalize wph
    wphA_norm = normalize_wph(wphA, norm=norm)
    wphB_norm = normalize_wph(wphB, norm=norm)
    wphAB_norm = normalize_wph(wphAB, norm=norm, wph_ref1=wphA, wph_ref2=wphB)
    return [wphA_norm, wphB_norm, wphAB_norm]

def WPH_comp(wph, comp = "isopar"):
    """ Compute "iso" or "isopar" compression of wph object
    
    Input: single wph object
    Output: compressed wph object
    
    comp: "iso" or "isopar"
    """   
    
    if comp == "iso":
        # Def periodic distance t2-t1 (between 0 and 2*L)
        def periodic_distance(t1, t2, per):
            return (t2-t1)%per
    elif comp == "isopar":
        # Def periodic distance |t2-t1| (between 0 and L)
        def periodic_distance(t1, t2, per):
            if t2 > t1:
                return min(t2 - t1, t1 - t2 + per)
            else:
                return min(t1 - t2, t2 - t1 + per)

    # Filling
    L = wph.L
    indices_cnt = {}
    wph_comp = {}
    for i in range(wph.wph_coeffs_indices.shape[0]):
        j1, t1, p1, j2, t2, p2, n, a, pseudo = tuple(wph.wph_coeffs_indices[i])
        dt = periodic_distance(t1, t2, 2 * L)
        if (j1, 0, p1, j2, dt, p2, n, a, pseudo) in indices_cnt.keys():
            indices_cnt[(j1, 0, p1, j2, dt, p2, n, a, pseudo)] += 1
            wph_comp[(j1, 0, p1, j2, dt, p2, n, a, pseudo)] += wph.wph_coeffs[..., i]
        else:
            indices_cnt[(j1, 0, p1, j2, dt, p2, n, a, pseudo)] = 1
            wph_comp[(j1, 0, p1, j2, dt, p2, n, a, pseudo)] = wph.wph_coeffs[..., i]

    # Conversion into numpy arrays and creation of new wph object
    wph_out = deepcopy(wph)
    indices = []
    wph_comp_list = []
    for key in indices_cnt.keys():
        indices.append(key)
        wph_comp_list.append(wph_comp[key] / indices_cnt[key])
    wph_out.wph_coeffs = np.moveaxis(np.array(wph_comp_list), 0, -1)
    wph_out.wph_coeffs_indices = np.array(indices)
    wph_out.reorder()
    
    return wph_out

def WPH_comp_many(list_wph, comp = "isopar"):
    """ Compute "iso" or "isopar" compression of a list of wph object
    
    Input: list of wph objects
    Output: compressed list of wph objects
    
    comp: "iso" or "isopar"
    """   
    list_wph_comp = []
    for i in range(len(list_wph)):
        list_wph_comp.append(WPH_comp(list_wph[i],comp=comp))
    return list_wph_comp

def WPH_ToList(wph, list_model=["S11"], part="real"):
    """ Transform a wph object into a list of coeff array
    Follow the order from a wph_model list
    
    Input: wph object
    Output: list of wph coefficients [[S11_coeffs],[S00_coeffs],...],
    only real, imag, or abs part is kept
    
    list_model: type of coefficients to keep
    part: "real", "imag", or "abs"
    """
    wph_list = []
    for coeff_type in list_model:
        if part == "real":
            coeffs = np.real(wph.get_coeffs(coeff_type)[0])
        elif part == "imag":
            coeffs = np.imag(wph.get_coeffs(coeff_type)[0])
        elif part == "abs":
            coeffs = np.abs(wph.get_coeffs(coeff_type)[0])
        wph_list.append(coeffs)
    return wph_list
        
def WPH_ToList_many(list_wph, list_model=["S11"], part="real"):
    """ Transform a list of wph objects into a list of list of wph coefficients
    Follow the order from a wph_model list
    
    Input: list wph object
    Output: list of list of wph coefficients (see WPH_ToList)
    
    list_model: type of coefficients to keep
    part: "real", "imag", or "abs"
    """
    list_wph_list = []
    for i in range(len(list_wph)):
        list_wph_list.append(WPH_ToList(list_wph[i], list_model, part=part))
    return list_wph_list

def Plot_single(wph, label, wph_model, part='real', out=False, ax=None, dpi=200):
    """ Plot a set of wph coefficients, from a wph object
    Require a wph_model list
    Plots are made by raws of 3 plots """
    
    # wph object to list
    coeffs = WPH_ToList(wph, wph_model, part=part)
    
    # Identify the number of coeffs type and preconfigure plot
    nb_types = len(wph_model)
    nb_lines = int(np.ceil(nb_types/3))
    
    try:
        ax[0].dtype
    except:
        fig, ax = plt.subplots(nb_lines, 3, figsize=(9.5, nb_lines*3), dpi=dpi, sharex=False, sharey=False)
    
    x = [] # abcisse lists
    for i in range(len(wph_model)):
        x.append(np.arange(1,len(coeffs[i]) + 1))

    if nb_lines == 1:
        # Do the plot (one line)
        for j in range(len(wph_model)):
            ax[j%3].plot(x[j],coeffs[j],label=label)
            ax[j%3].grid(False)
            ax[j%3].set_xlabel('$j_1$' if 'S' in wph_model[j] else '($j_1$,$j_2$,$\Delta l$)')
            ax[j%3].set_title(wph_model[j])
        ax[2].legend(loc=3,bbox_to_anchor=(1.05,0.05))
    else:
        # Do the plot (multiple lines)
        for j in range(len(wph_model)):
            ax[j//3,j%3].plot(x[j],coeffs[j],label=label)
            ax[j//3,j%3].grid(False)
            ax[j//3,j%3].set_xlabel('$j_1$' if 'S' in wph_model[j] else '($j_1$,$j_2$,$\Delta l$)')
            ax[j//3,j%3].set_title(wph_model[j])
        ax[0,2].legend(loc=3,bbox_to_anchor=(1.05,0.05))
    
    plt.tight_layout()

    if out:
        return coeffs, ax
    
def Plot_full_single(wph, label, wph_model, part='real', norm='S11', comp="iso", out=False, ax=None, dpi=200):
    """ Plot a set of wph coefficients, from a wph object
    Require a wph_model list
    Plots are made by raws of 3 plots
    Coefficients will be normalized, and compressed """
    
    # Normalize and compress
    wph_loc = normalize_wph(wph, norm=norm)
    wph_loc = WPH_comp(wph_loc, comp=comp)
    
    # Plot 
    if out:
        coeffs, ax = Plot_single(wph_loc, label, wph_model, part=part, out=True, ax=ax, dpi=dpi)
        return coeffs, ax
    else:
        Plot_single(wph_loc, label, wph_model, part=part, out=False, ax=ax, dpi=dpi)
    
def Plot_many(list_wph, list_label, wph_model, part='real', out=False, ax=None, dpi=200):
    """ Plot a set of wph coefficients, from a list of wph object
    Require a wph_model list
    Plots are made by raws of 3 plots """
    
    # wph object to list
    list_coeffs = WPH_ToList_many(list_wph,wph_model,part=part)
    
    # Identify the number of coeffs type and preconfigure plot
    nb_wph = len(list_wph)
    nb_types = len(wph_model)
    nb_lines = int(np.ceil(nb_types/3))
    
    try:
        ax[0].dtype
    except:
        fig, ax = plt.subplots(nb_lines, 3, figsize=(9.5, nb_lines*3), dpi=dpi, sharex=False, sharey=False)
        
    x = [] # abcisse lists
    for i in range(len(wph_model)):
        x.append(np.arange(1,len(list_coeffs[0][i]) + 1))

    if nb_lines == 1:
        # Do the plot (one line)
        for j in range(len(wph_model)):
            for i in range(nb_wph):
                ax[j%3].plot(x[j],list_coeffs[i][j],label=list_label[i])
                ax[j%3].grid(False)
                ax[j%3].set_xlabel('$j_1$' if 'S' in wph_model[j] else '($j_1$,$j_2$,$\Delta l$)')
                ax[j%3].set_title(wph_model[j])
        ax[2].legend(loc=3,bbox_to_anchor=(1.05,0.05))
    else:
        # Do the plot (multiple lines)
        for j in range(len(wph_model)):
            for i in range(nb_wph):
                ax[j//3,j%3].plot(x[j],list_coeffs[i][j],label=list_label[i])
                ax[j//3,j%3].grid(False)
                ax[j//3,j%3].set_xlabel('$j_1$' if 'S' in wph_model[j] else '($j_1$,$j_2$,$\Delta l$)')
                ax[j//3,j%3].set_title(wph_model[j])
        ax[0,2].legend(loc=3,bbox_to_anchor=(1.05,0.05))
    
    plt.tight_layout()
    
    if out:
        return list_coeffs, ax
    
def Plot_full_trio(trio_wph, trio_label, wph_model, part='real', norm='S11', comp = "iso", out=False, ax=None, dpi=200):
    """ Plot a set of wph coefficients, from a trio of wph object
    Require a wph_model list
    Plots are made by raws of 3 plots
    Coefficients will be normalized, and compressed """
    
    # Normalize and compress
    trio_wph_loc = Normalize_trio(trio_wph, norm=norm)
    trio_wph_loc = WPH_comp_many(trio_wph_loc, comp=comp)
            
    # Plot 
    if out:
        list_coeffs, ax = Plot_many(trio_wph_loc, trio_label, wph_model, part=part, out=True, ax=ax, dpi=dpi)
        return list_coeffs, ax
    else:
        Plot_many(trio_wph_loc, trio_label, wph_model, part=part, out=False, ax=ax, dpi=dpi)

def wph_save(name, wph):
    name = name + '.pickle'
    with open(name, 'wb') as file:
        pickle.dump(wph, file) 
        
def wph_load(name):
    name = name + '.pickle'
    with open(name, 'rb') as file:
        wph = pickle.load(file)
    return wph

def C_FullToReduce(coeffs, L=4):
    """ Compute a reduced set of coefficients from 
    C00, C01, C10 real or imag isotropic coefficients.
    The coefficients should be ordered in a standard
    way, where the coefficients of different dt=t2
    is the last varying parameter.
    The reduced coefficients are mean, amplitude of 
    first harmonic, angle of first harmonics.
    
    Input: set of coefficients, float, dim 2*L*Nb
    Ouput: set of reduced coefficient, float, dim 3*Nb

    L: number of angles between zero and pi
    """
    
    # Reshape by factorising the angular dimension
    coeffs = np.reshape(coeffs,(-1,2*L))
    
    # Compute fft and identify the different coefficients
    fft = np.fft.fft(coeffs, axis=-1)
    mean = np.real(fft[:,0]) / (2*L)
    amp = np.abs(fft[:,2]) / (2*L)
    ang = np.angle(fft[:,2])
    rcoeffs = np.concatenate((mean[None,:],amp[None,:],ang[None,:]),axis=0)
    
    return rcoeffs

def C_ReduceToFull(rcoeffs, L=4):
    """ Compute a set of real of image C00, C01, C10
    coefficients from their reduced version
    The reduced coefficients are mean, amplitude of 
    first harmonic, angle of first harmonics.
    
    Input: set of reduced coefficients, float, dim 3*Nb
    Ouput: set of reconstructed coefficients, float, dim 2*L*Nb

    L: number of angles between zero and pi
    """
    
    # Extract mean, amp, and ang
    mean = rcoeffs[0,:][:,None] * np.ones(2*L)[None,:]
    amp = rcoeffs[1,:][:,None] * np.ones(2*L)[None,:]
    ang = rcoeffs[2,:][:,None] * np.ones(2*L)[None,:]

    # Compute a vector of delta_theta
    dt = np.ones(rcoeffs.shape[1])[:,None] * np.arange(2*L)[None,:]
    
    # Compute reconstructed coefficients
    coeffs = mean + 2*amp*np.cos(2*dt*2*np.pi/(2*L) + ang)
    coeffs = coeffs.flatten()
    
    return coeffs

def C_FullToReduceFA(coeffs, L=4):
    """ FullToReduce with Fixed Angle (FA)
  
    Compute a reduced set of coefficients from 
    C00, C01, C10 real or imag isotropic coefficients.
    The coefficients should be ordered in a standard
    way, where the coefficients of different dt=t2
    is the last varying parameter.
    The reduced set is computed from a mean, and an 
    amplitude of first cosine harmonics aligne with
    delta_theta=0.
    
    Input: set of coefficients, float, dim 2*L*Nb
    Ouput: set of reduced coefficient, float, dim 2*Nb

    L: number of angles between zero and pi
    """
    
    # Reshape by factorising the angular dimension
    coeffs = np.reshape(coeffs,(-1,2*L))
    
    # Identify the mean and first harmonic
    mean = np.mean(coeffs, axis = 1)
    cos = np.cos(2 * np.arange(2*L)*np.pi/L)
    amp = coeffs * cos[None,:]
    amp = 2 * np.mean(amp, axis = 1)
    rcoeffs = np.concatenate((mean[None,:],amp[None,:]),axis=0)
    
    return rcoeffs

def C_ReduceToFullFA(rcoeffs, L=4):
    """ ReduceToFull with Fixed Angle (FA)
    
    Compute a set of real of image C00, C01, C10
    coefficients from their reduced version
    The reduced sare a mean, and an amplitude
    of first cosine harmonics aligne with
    delta_theta=0.
    
    Input: set of reduced coefficients, float, dim 2*Nb
    Ouput: set of reconstructed coefficient, dim 2*L*Nb

    L: number of angles between zero and pi
    """
    
    # Extract mean, amp, and ang
    mean = rcoeffs[0,:][:,None] * np.ones(2*L)[None,:]
    cos = rcoeffs[1,:][:,None] * np.cos(2 * np.arange(2*L)*np.pi/L)
    
    # Compute reconstructed coefficients
    coeffs = (mean + cos).flatten()
    
    return coeffs