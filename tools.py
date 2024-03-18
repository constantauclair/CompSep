"""
Modules to import.

"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import time
import pywph as pw
import scipy.optimize as opt
from cycler import cycler
import warnings
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

"""
Initial settings.

"""

warnings.filterwarnings("ignore")
plt.rcParams['text.usetex'] = True
color = ['#377EB8', '#FF7F00', '#4DAF4A','#F781BF', '#A65628', '#984EA3','#999999', '#E41A1C', '#DEDE00']
style = ['-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-']
default_cycler = (cycler(color=['#377EB8','#FF7F00','#4DAF4A','#FF7F00','#4DAF4A']))
plt.rc('axes', prop_cycle=default_cycler)

"""
Functions.

"""

def plot(image,cmap='inferno',logscale=False,ampcol=3,size=6,unit=r'$\rm{I \ [MJy/sr]}$',title='',x_title=0.45,y_title=0.85,colorbar=True,no_show=False):
    """
    Quick plot for a single image.
    
    Parameters
    ----------
    image : numpy 2D array
        Image to plot.
    cmap : str, optional
         Colormap to use.
    logscale : bool, optional
        Plot the image in log scale.
        The default is False.
    ampcol : float, optional
        Range of values to plot in std unit, centered on the mean.
        The default is 3.
    size : float, optional
        Size of the figure.
        The default is 6.
    unit : str, optional
        Label of the colorbar.
        The default is r'$\rm{I \ [MJy/sr]}$'.
    title : str, optional
        Title of the figure.
        The default is no title.
    x_title : float, optional
        X position of the title in figure coordinates.
        The default is 0.45.
    y_title : float, optional
        Y position of the title in figure coordinates.
    colorbar : bool, optional
        Set a colorbar.
        The default is True.
    no_show : bool, optional
        Remove the plt.show(), which can cause problems to save the figure.
        The default is False.
        
    Returns
    -------
    None.
    
    """
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
    
def plot_subplot(ax,i,data,label):
    """
    Plot a subplot on a row figure.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes of the figure.
    i : int
        Ax index.
    data : numpy 2D array
        Data to plot.
    label : str
        Title of the subplot.

    Returns
    -------
    matplotlib.image.AxesImage
        Subplot.

    """
    A = ax[i].imshow(data,cmap='inferno',vmin=np.mean(data)-3*np.std(data),vmax=np.mean(data)+3*np.std(data),origin='lower')
    ax[i].set_title(label,fontsize=20,pad=10)
    ax[i].grid(False)
    ax[i].xaxis.set_visible(False)
    ax[i].yaxis.set_visible(False)
    return A

def set_colorbar(ax,A,i,label=r'$\rm{[\mu K_{CMB}]}$'):
    """
    Add a colorbar on a subplot.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes of the figure.
    A : matplotlib.image.AxesImage
        Subplot.
    i : int
        Ax index.
    label : str, optional
        Label of the colorbar.

    Returns
    -------
    None.

    """
    def axins(i):
        return inset_axes(ax[i],width="100%", height="15%",loc='lower left',bbox_to_anchor=(0, -0.1, 1, 0.44),bbox_transform=ax[i].transAxes,borderpad=0,)
    cb=plt.colorbar(A,cax=axins(i),ax=None,orientation='horizontal')
    cb.set_label(label,rotation=0,fontsize=15,labelpad=6)
    return

def powspec(image, im2=None, reso=1, apo=1, N_bin=40, lin_log_trans=3, ret_bins=False):
    """
    Computes the power spectrum.

    Parameters
    ----------
    image : numpy 2D array
        Data of which the power spectrum is computed.
    im2 : numpy 2D array, optional
        If not None, the cross spectrum between image and im2 is computed. 
        The default is None.
    reso : float, optional
        Pixel size in arcminute. 
        The default is 1.
    apo : float, optional
        Width of apodization mask. If the data are not periodic, we recommand to use apo=0.9. 
        The default is 1.
    N_bin : int, optional
        Number of bins. 
        The default is 40.
    lin_log_trans : float, optional
        Define the transition between the linear and logarithmic binning regime. 
        The default is 3.
    ret_bins : bool, optional
        Returns the bins. 
        The default is False.

    Returns
    -------
    numpy 1D array
        Center of each k bin.
    numpy 1D array
        Power spectrum values.
    numpy 1D array
        Bin edges.
        Only returned if ret_bins = True.

    """
      
    na=image.shape[1]
    nb=image.shape[0]
    nf=max(na,nb)

    k_crit = nf//2 
    k_crit_phy = 1/(2*reso) # k_max in arcmin-1
    
    def bining(k_crit,a,N):
        indices = np.arange(N+1)
        return k_crit * np.sinh(a*indices/N) / np.sinh(a)
    
    bins = bining(k_crit,lin_log_trans,N_bin)
    
    bins_size = np.zeros(N_bin) # bins size in arcmin-1
    for i in range(N_bin):
        bins_size[i] = (bins[i+1] - bins[i]) / k_crit * k_crit_phy
    
    if apo < 1:
        image = image * apodize(na,nb,apo)

    imft=np.fft.fft2(image) / (na*nb)
	
    if im2 is not None:
        if apo < 1:
            im2 = im2 * apodize(na,nb,apo)
        im2ft = np.fft.fft2(im2) / (na*nb)
        ps2D = (imft * np.conj(im2ft) * (na*nb) + im2ft * np.conj(imft) * (na*nb))/2#imft * np.conj(im2ft) * (na*nb)
    else:
        ps2D = np.abs( imft )**2 * (na*nb)

    del imft

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

    hval, rbin = np.histogram(k_mat,bins=bins)

	#Average values in same k bin
	#---------------------------------------------

    kval = np.zeros(N_bin-1)
    kpow = np.zeros(N_bin-1)

    for j in range(N_bin-1):
        kval[j] = np.sum(k_mat[np.logical_and(k_mat >= bins[j], k_mat < bins[j+1])]) / hval[j]
        kpow[j] = np.sum(np.real(ps2D[np.logical_and(k_mat >= bins[j], k_mat < bins[j+1])])) / hval[j]
        
    spec_k = kpow[1:np.size(hval)-1] * (reso / 60 / 180 * np.pi)**2 / np.mean(apodize(na,nb,apo)**2)
    
    tab_k = kval[1:np.size(hval)-1] / k_crit * k_crit_phy
    bins = bins / k_crit * k_crit_phy
    
    if ret_bins == False:
        return tab_k, spec_k
    else:
        return tab_k, spec_k, bins

def apodize(na, nb, radius):
    """
    Build a mask to apodize non-periodic data.

    Parameters
    ----------
    na : int
        Size in x.
    nb : int
        Size in y.
    radius : float
        Fraction of kept pixels along each axis.

    Returns
    -------
    numpy 2D array
        Apodization mask.

    """
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
    """
    Computes the phase randomisation of a process. Simple way to construct a close-to-Gaussian map 
    that has the same power spectrum as the initial process.

    Parameters
    ----------
    x : numpy 2D array
        Process to be phase randomized.

    Returns
    -------
    numpy 2D array
        A phase randomization of x.

    """
    [M,N] = np.shape(x) # M and N must be multiples of 2
    ft_x = np.fft.fftshift(np.fft.fft2(x))
    mod_ft_x = np.abs(ft_x)
    gauss = np.random.normal(0,1,size=(M,N))
    ft_gauss = np.fft.fftshift(np.fft.fft2(gauss))
    angle_ft_gauss = np.angle(ft_gauss)
    new_ft_x = mod_ft_x * np.exp(1j*angle_ft_gauss)
    rand_x = np.fft.ifft2(np.fft.ifftshift(new_ft_x))
    pr_x = np.real(rand_x)
    return pr_x

def plot_PS(images,labels=None,colors=None,styles=None,reso=1,apo=0.95,N_bin=40,lin_log_trans=3,cross=None,ret_PS=False,abs_cross=True,axis='k',fontsize=20,plot=True):
    """
    Computes and plots the power spectra of a set of maps.

    Parameters
    ----------
    images : numpy 2D, 3D or 4D array
        Map(s) of which the power spectra are computed.
        If 2D array, the power spectrum of the map is plotted.
        If 3D array, the power spectrum of each map is plotted.
        If 4D array, the mean and std of the power spectra computed on the first axis are plotted.
    labels : list, optional
        List of the labels of each curve.
        The default is None.
    colors : list, optional
        List of the colors of each curve. 
        The default is None.
    styles : list, optional
        List of the linestyles of each curve. 
        The default is None.
    reso : float, optional
        Pixel size in arcminutes. 
        The default is 1.
    apo : float, optional
        Apodization fraction. 
        The default is 0.95.
    N_bin : int, optional
        Bin number. 
        The default is 40.
    lin_log_trans : TYPE, optional
        Define the transition between the linear and logarithmic binning regime. 
        The default is 3.
    cross : list, optional
        Additional cross spectra to compute. 
        It has to be a list of ["index of map 1", "index of map 2", "color", "style"].
        The default is None.
    ret_PS : bool, optional
        Returns the power spectra values. 
        The default is False.
    abs_cross : bool, optional
        Plot the absolute value of the cross spectra. 
        The default is True.
    axis : str, optional
        Unit of the x-axis. Choose 'k' for wavenumbers in arcmin-1 and 'l' for multipoles.
        The default is 'k'.
    fontsize : int, optional
        Fontsize of the plot. 
        The default is 20.
    plot : bool, optional
        Plot the figure. 
        The default is True.

    Returns
    -------
    numpy 1D array
        Center of the bins.
    numpy 2D array
        Vector of the (mean of the) power spectra.
    numpy 2D array
        Vector of the std of the power spectra.

    """
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
    if labels is None:
        labels = np.arange(1,n+1).astype(str).tolist()
    if colors is None:
        colors = color[:n]
    if styles is None:
        styles = ['-']*n
    if t == 1:
        # PS computation
        if n == 1:
            PS = powspec(images,reso=reso,apo=apo,N_bin=N_bin,lin_log_trans=lin_log_trans)
        else:
            PS = []
            for i in range(n):
                PS.append(powspec(images[i],reso=reso,apo=apo,N_bin=N_bin,lin_log_trans=lin_log_trans))
            if cross is not None:
                for i in range(len(cross)):
                    if not abs_cross:
                        PS.append(powspec(images[cross[i][0]],im2=images[cross[i][1]],reso=reso,apo=apo,N_bin=N_bin,lin_log_trans=lin_log_trans))
                    else:
                        PS.append(np.abs(powspec(images[cross[i][0]],im2=images[cross[i][1]],reso=reso,apo=apo,N_bin=N_bin,lin_log_trans=lin_log_trans)))
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
                power_spectra.append(powspec(images[j,0],reso=reso,apo=apo,N_bin=N_bin,lin_log_trans=lin_log_trans))
            else:
                PS = []
                for i in range(n):
                    PS.append(powspec(images[j,i],reso=reso,apo=apo,N_bin=N_bin,lin_log_trans=lin_log_trans))
                if cross is not None:
                    for i in range(len(cross)):
                        if not abs_cross:
                            PS.append(powspec(images[j,cross[i][0]],im2=images[j,cross[i][1]],reso=reso,apo=apo,N_bin=N_bin,lin_log_trans=lin_log_trans))
                        else:
                            PS.append(np.abs(powspec(images[j,cross[i][0]],im2=images[j,cross[i][1]],reso=reso,apo=apo,N_bin=N_bin,lin_log_trans=lin_log_trans)))
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
            plt.xlabel(r'$\rm{\ell}$',fontsize=fontsize)
            plt.ylabel(r'$\rm{C_\ell \ [\mu K^2]}$',fontsize=fontsize)
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
    """
    Plots the modulus of the WPH coefficients of a set of maps.

    Parameters
    ----------
    images : numpy 2D, 3D or 4D array
        Map(s) of which the WPH statistics are computed.
        If 2D array, the WPH statistics of the map are plotted.
        If 3D array, the WPH statistics of each map are plotted.
        If 4D array, the mean and std of the WPH statistics computed on the first axis are plotted.
    labels : list, optional
        List of the labels of each curve.
    colors : list, optional
        List of the colors of each curve. 
        The default is color.
    styles : list, optional
        List of the linestyles of each curve. 
        The default is style.
        DESCRIPTION. The default is None.
    J : int, optional
        Number of dyadic scale to probe. 
        The default is None.
    L : int, optional
        Number of angles to probe. 
        The default is 4.
    dn : int, optional
        Number of translation to compute. 
        The default is 0.
    pbc : bool, optional
        Set it to True if the data are periodic. 
        The default is False.
    ret_coeffs : bool, optional
        Returns the WPH statistics. 
        The default is False.

    Returns
    -------
    numpy 3D array
        Array containing the mean and std of the WPH statistics of each map.

    """
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
    if labels is None:
        labels = np.arange(1,n+1).astype(str).tolist()
    if colors is None:
        colors = color[:n]
    if styles is None:
        styles = ['-']*n
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
    """
    Plots the histograms of the data.

    Parameters
    ----------
    images : numpy 2D, 3D or 4D array
        Map(s) of which the histogram is computed.
        If 2D array, the histogram of the map is plotted.
        If 3D array, the histogram of each map is plotted.
        If 4D array, the mean and std of the histograms computed on the first axis are plotted.
    labels : list, optional
        List of the labels of each curve.
    colors : list, optional
        List of the colors of each curve. 
        The default is color.
    styles : list, optional
        List of the linestyles of each curve. 
        The default is style.
    n_bins : int, optional
        Bin number. 
        The default is 100.
    log : bool, optional
        Puts the y scale in log. 
        The default is False.
    density : bool, optional
        Normalizes the histogram. 
        The default is True.
    value_range : tuple, optional
        Min and max values of the histogram. 
        The default is None.
    fontsize : float, optional
        Fontsize of the plot. 
        The default is 15.
    nature : str, optional
        Unit of the data. 
        The default is r'$\rm{I}$'.

    Returns
    -------
    None.

    """
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
    """
    Generate synthesized map(s) of a given process using WPH statistics.

    Parameters
    ----------
    data : numpy 2D array
        Data of which syntheses will be computed.
    n_syn : int, optional
        Number of syntheses. 
        The default is 1.
    n_step : int, optional
        Number of steps if the synthesis algorithm. 
        The default is 50.
    print_loss : bool, optional
        Prints the loss at each iteration. 
        The default is False.
    pbc : bool, optional
        Set it to True if the data are periodic. 
        The default is False.
    verbose : bool, optional
        Set it to True to print some stuff during the run. 
        The default is False.
    device : str, optional
        Device on which the algorithm is runned. 
        The default is 'cpu'.
    dn : int, optional
        Number of translation in the WPH statistics computation. 
        The default is 2.
    tau_grid : str, optional
        Method for the definition of the translations. The default is 'exp'.

    Returns
    -------
    numpy 2D or 3D array
        Synthesis or array of syntheses of the given process.

    """
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