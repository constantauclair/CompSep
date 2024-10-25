import numpy as np
import numpy.ma as ma
from math import *

def powspec(image, reso=1, apo=1, N_bin=100, a=4, ret_bins=False, nan_frame=False, **kwargs):
	
    # reso have to be in arcmin
    # the output unity is I²/sr
    
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
        
    na=image.shape[1]
    nb=image.shape[0]
    nf=max(na,nb)

    if 'mask' in kwargs:
        Mask = np.fft.ifftshift(kwargs.get('mask'))
    else:
        Mask = np.zeros(np.shape(image)) + 1

    k_crit = nf//2 
    k_crit_phy = 1/(2*reso) # k_max in arcmin-1
    
    def bining(k_crit,a,N):
        indices = np.arange(N+1)
        return k_crit * np.sinh(a*indices/N) / np.sinh(a)
    
    bins = bining(k_crit,a,N_bin)
    
    bins_size = np.zeros(N_bin) # bins size in arcmin-1
    for i in range(N_bin):
        bins_size[i] = (bins[i+1] - bins[i]) / k_crit * k_crit_phy
    
    if apo < 1:
        image = image * apodize(na,nb,apo)
    
	#Fourier transform & 2D power spectrum
	#---------------------------------------------
	
    if(nan_frame == True):
        image = np.nan_to_num(image, copy=False, nan=0)

    imft=np.fft.fft2(image) / (na*nb)
	
    if 'im2' in kwargs:
        im2 = kwargs.get('im2')
        if apo < 1:
            im2 = im2 * apodize(na,nb,apo)
        im2ft = np.fft.fft2(im2) / (na*nb)
        ps2D = (imft * np.conj(im2ft) * (na*nb) + im2ft * np.conj(imft) * (na*nb))/2#imft * np.conj(im2ft) * (na*nb)
    else:
        ps2D = np.abs( imft )**2 * (na*nb)

    del imft

	#Set-up kbins
	#---------------------------------------------

    x=np.arange(na)
    y=np.arange(nb)
    x,y=np.meshgrid(x,y)

    if (na % 2) == 0:
        x = (1.*x - ((na)/2.) ) / na
        shiftx = (na)/2.
    else:
        x = (1.*x - (na-1)/2.)/ na
        shiftx = (na-1.)/2.+1

    if (nb % 2) == 0:
        y = (1.*y - ((nb/2.)) ) / nb
        shifty = (nb)/2.
    else:
        y = (1.*y - (nb-1)/2.)/ nb
        shifty = (nb-1.)/2+1

    k_mat = np.sqrt(x**2 + y**2)
    k_mat = k_mat * nf 
	
    k_mat= np.roll(k_mat,int(shiftx), axis=1)
    k_mat= np.roll(k_mat,int(shifty), axis=0)

    hval, rbin = np.histogram(k_mat,bins=bins,weights=Mask)

	#Average values in same k bin
	#---------------------------------------------

    kval = np.zeros(N_bin-1)
    kpow = np.zeros(N_bin-1)

    for j in range(N_bin-1):
        kval[j] = np.sum(k_mat[np.logical_and(np.logical_and(k_mat >= bins[j], k_mat < bins[j+1]),Mask == 1)]) / hval[j]
        kpow[j] = np.sum(np.real(ps2D[np.logical_and(np.logical_and(k_mat >= bins[j], k_mat < bins[j+1]),Mask == 1)])) / hval[j]
        
    spec_k = kpow[1:np.size(hval)-1] * (reso / 60 / 180 * np.pi)**2 / np.mean(apodize(na,nb,apo)**2)
    
    tab_k = kval[1:np.size(hval)-1] / k_crit * k_crit_phy
    bins = bins / k_crit * k_crit_phy
    
    if ret_bins == False:
        return tab_k, spec_k
    else:
        return tab_k, spec_k, bins
    
# def TEB_powspec(T, E, B, IQU=False, reso=1, apo=1, N_bin=100, a=4, ret_bins=False, nan_frame=False, **kwargs):
# 	
#     # reso have to be in arcmin
#     # the output unity is I²/sr
    
#     def apodize(na, nb, radius):
#         na = int(na)
#         nb = int(nb)
#         ni = int(radius * na)
#         nj = int(radius * nb)
#         dni = na - ni
#         dnj = nb - nj
#         tap1d_x = np.zeros(na) + 1.0
#         tap1d_y = np.zeros(nb) + 1.0
#         tap1d_x[:dni] = (np.cos(3*np.pi/2.+np.pi/2.*(1.*np.linspace(0,dni-1,dni)/(dni-1)) ))
#         tap1d_x[na-dni:] = (np.cos(0.+np.pi/2.*(1.*np.linspace(0,dni-1,dni)/(dni-1)) ))
#         tap1d_y[:dnj] = (np.cos(3*np.pi/2.+np.pi/2.*(1.*np.linspace(0,dnj-1,dnj)/(dnj-1)) ))
#         tap1d_y[nb-dnj:] = (np.cos(0.+np.pi/2.*(1.*np.linspace(0,dnj-1,dnj)/(dnj-1)) ))
#         tapper = np.zeros([na,nb])
#         for i in range(nb):
#             tapper[:,i] = tap1d_x
#         for i in range(na):
#             tapper[i,:] = tapper[i,:] * tap1d_y
#         return tapper
    
#     na=T.shape[1]
#     nb=T.shape[0]
#     nf=max(na,nb)

#     if 'mask' in kwargs:
#         Mask = np.fft.ifftshift(kwargs.get('mask'))
#     else:
#         Mask = np.zeros(np.shape(T)) + 1

#     k_crit = nf//2 
#     k_crit_phy = 1/(2*reso) # k_max in arcmin-1
    
#     def bining(k_crit,a,N):
#         indices = np.arange(N+1)
#         return k_crit * np.sinh(a*indices/N) / np.sinh(a)
    
#     bins = bining(k_crit,a,N_bin)
    
#     bins_size = np.zeros(N_bin) # bins size in arcmin-1
#     for i in range(N_bin):
#         bins_size[i] = (bins[i+1] - bins[i]) / k_crit * k_crit_phy
    
#     if apo < 1:
#         T = T * apodize(na,nb,apo)
#         E = E * apodize(na,nb,apo)
#         B = B * apodize(na,nb,apo)
    
# 	#Fourier transform & 2D power spectrum
# 	#---------------------------------------------
# 	
#     if(nan_frame == True):
#         T = np.nan_to_num(T, copy=False, nan=0)
#         E = np.nan_to_num(E, copy=False, nan=0)
#         B = np.nan_to_num(B, copy=False, nan=0)

#     T_ft = np.fft.fft2(T) / (na*nb)
    
#     if IQU == True:
#         QiU = E+1j*B
#         QiU_ft = np.fft.fft2(QiU)
#         k2d_x, k2d_y = np.meshgrid(np.fft.fftfreq(na), np.fft.fftfreq(nb), indexing='ij')
#         phik = np.arctan2(k2d_y, k2d_x)
#         EiB = np.fft.ifft2(np.exp(-2j*phik)*QiU_ft) / np.sqrt(2)
#         E = np.real(EiB)
#         B = np.imag(EiB)
    
#     E_ft = np.fft.fft2(E) / (na*nb)
#     B_ft = np.fft.fft2(B) / (na*nb)

#     ps2D_TT = np.abs( T_ft )**2 * (na*nb)
#     ps2D_EE = np.abs( E_ft )**2 * (na*nb)
#     ps2D_BB = np.abs( B_ft )**2 * (na*nb)
#     ps2D_TE = (T_ft * np.conj(E_ft) * (na*nb) + E_ft * np.conj(T_ft) * (na*nb)) / 2
#     ps2D_TB = (T_ft * np.conj(B_ft) * (na*nb) + B_ft * np.conj(T_ft) * (na*nb)) / 2
#     ps2D_EB = (E_ft * np.conj(B_ft) * (na*nb) + B_ft * np.conj(E_ft) * (na*nb)) / 2

# 	#Set-up kbins
# 	#---------------------------------------------

#     x=np.arange(na)
#     y=np.arange(nb)
#     x,y=np.meshgrid(x,y)

#     if (na % 2) == 0:
#         x = (1.*x - ((na)/2.) ) / na
#         shiftx = (na)/2.
#     else:
#         x = (1.*x - (na-1)/2.)/ na
#         shiftx = (na-1.)/2.+1

#     if (nb % 2) == 0:
#         y = (1.*y - ((nb/2.)) ) / nb
#         shifty = (nb)/2.
#     else:
#         y = (1.*y - (nb-1)/2.)/ nb
#         shifty = (nb-1.)/2+1

#     k_mat = np.sqrt(x**2 + y**2)
#     k_mat = k_mat * nf 
# 	
#     k_mat= np.roll(k_mat,int(shiftx), axis=1)
#     k_mat= np.roll(k_mat,int(shifty), axis=0)

#     hval, rbin = np.histogram(k_mat,bins=bins,weights=Mask)

# 	#Average values in same k bin
# 	#---------------------------------------------

#     kval = np.zeros(N_bin-1)
#     kpow_TT = np.zeros(N_bin-1)
#     kpow_EE = np.zeros(N_bin-1)
#     kpow_BB = np.zeros(N_bin-1)
#     kpow_TE = np.zeros(N_bin-1)
#     kpow_TB = np.zeros(N_bin-1)
#     kpow_EB = np.zeros(N_bin-1)

#     for j in range(N_bin-1):
#         kval[j] = np.sum(k_mat[np.logical_and(np.logical_and(k_mat >= bins[j], k_mat < bins[j+1]),Mask == 1)]) / hval[j]
#         kpow_TT[j] = np.sum(np.real(ps2D_TT[np.logical_and(np.logical_and(k_mat >= bins[j], k_mat < bins[j+1]),Mask == 1)])) / hval[j]
#         kpow_EE[j] = np.sum(np.real(ps2D_EE[np.logical_and(np.logical_and(k_mat >= bins[j], k_mat < bins[j+1]),Mask == 1)])) / hval[j]
#         kpow_BB[j] = np.sum(np.real(ps2D_BB[np.logical_and(np.logical_and(k_mat >= bins[j], k_mat < bins[j+1]),Mask == 1)])) / hval[j]
#         kpow_TE[j] = np.sum(np.real(ps2D_TE[np.logical_and(np.logical_and(k_mat >= bins[j], k_mat < bins[j+1]),Mask == 1)])) / hval[j]
#         kpow_TB[j] = np.sum(np.real(ps2D_TB[np.logical_and(np.logical_and(k_mat >= bins[j], k_mat < bins[j+1]),Mask == 1)])) / hval[j]
#         kpow_EB[j] = np.sum(np.real(ps2D_EB[np.logical_and(np.logical_and(k_mat >= bins[j], k_mat < bins[j+1]),Mask == 1)])) / hval[j]
        
#     spec_k_TT = kpow_TT[1:np.size(hval)-1] * (reso / 60 / 180 * np.pi)**2
#     spec_k_EE = kpow_EE[1:np.size(hval)-1] * (reso / 60 / 180 * np.pi)**2
#     spec_k_BB = kpow_BB[1:np.size(hval)-1] * (reso / 60 / 180 * np.pi)**2
#     spec_k_TE = kpow_TE[1:np.size(hval)-1] * (reso / 60 / 180 * np.pi)**2
#     spec_k_TB = kpow_TB[1:np.size(hval)-1] * (reso / 60 / 180 * np.pi)**2
#     spec_k_EB = kpow_EB[1:np.size(hval)-1] * (reso / 60 / 180 * np.pi)**2
    
#     spec_k = np.array([spec_k_TT,spec_k_EE,spec_k_BB,spec_k_TE,spec_k_TB,spec_k_EB])
    
#     tab_k = kval[1:np.size(hval)-1] / k_crit * k_crit_phy
#     bins = bins / k_crit * k_crit_phy
    
#     if ret_bins == False:
#         return tab_k, spec_k
#     else:
#         return tab_k, spec_k, bins
