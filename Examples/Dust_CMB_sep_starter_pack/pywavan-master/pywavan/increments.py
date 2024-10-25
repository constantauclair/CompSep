import numpy as np
import matplotlib.pyplot as plt

def within_pi2(a):
    return (a + 0.5*np.pi) % np.pi - 0.5*np.pi

# Increments of a scalar field over a 2D space
def increments(v_map, lag, norm=2, nbins=None, ivmax=None, randa=False, noper=False):
    # find every relevant shifts and their number
    shifts = []
    # We look for the shifts in a squared area [-lag, lag]^2
    for i in range(-lag, lag + 1):
        for j in range(-lag, lag + 1):
            if lag ** 2 <= i*i + j*j < (lag + 1) ** 2:
                shifts.append([i, j])
    # for each shift, compute absolute or squared difference between shifted and original array
    # then add it to the CVI map
    nx, ny = np.shape(v_map)
    if nbins is None:
        nbins = int(np.sqrt(nx * ny))
    iv_map = np.zeros(np.shape(v_map))
    if ivmax is None:
        ivmax = abs(v_map.max()) + abs(v_map.min())
        if np.isnan(ivmax):
            raise Exception('Pb array has nans... Please provide relevant range.')
    ivhist = np.array([0] * nbins)
    for shift in shifts:
        inc = v_map - np.roll(np.roll(v_map, shift[0], axis=0), shift[1], axis=1)
        # Put nans out of range
        inc[np.isnan(inc)] = ivmax * 2
        if norm == 2:
            # WARNING: this includes discarded values in range...
            iv_map = iv_map + inc ** 2
        else:
            iv_map = iv_map + np.abs(inc)
        # If non periodic, skip edges:
        if noper:
            if shift[0] >= 0:
                if shift[1] >= 0:
                    inc = inc[shift[0]:, shift[1]:]
                else:
                    inc = inc[shift[0]:, :shift[1]]
            else:
                if shift[1] >= 0:
                    inc = inc[:shift[0], shift[1]:]
                else:
                    inc = inc[:shift[0], :shift[1]]
        nbs, bins = np.histogram(inc.flatten(), bins=nbins, range=(-ivmax, ivmax))
        ivhist = ivhist + np.array(nbs)
    cbin = np.array((bins[:-1] + bins[1:]) * 0.5)
    # WARNING: this sigma is now only within range...
    sig = np.sqrt(np.sum(ivhist * cbin ** 2) / np.sum(ivhist))
    iv_map = iv_map / len(shifts)
    if norm == 2:
        iv_map = np.sqrt(iv_map)
    return iv_map, cbin, np.double(ivhist) / len(shifts), sig


def find_gauss_core_hist(x, hin, fac=1, tol=1e-2):
    h = np.array(hin, dtype='float') / np.sum(hin) # normalise input histogram
    mu = np.sum(x * h)
    sig0 = np.sqrt(np.sum((x - mu) ** 2 * h))
    sigold = 0.
    sig = sig0
    while np.abs(sig - sigold) > tol * sig0:
        sigold = sig
        hb = h[np.abs(x - mu) < fac * sig]
        coeff = np.sum(hb) # Keep track of normalisation
        hb = np.array(hb, dtype='float') / coeff
        xb = x[np.abs(x - mu) < fac * sig]
        mu = np.sum(xb * hb)
        p = np.polyfit(xb - mu, np.log(hb + 1e-10), 2)
        sig = np.sqrt(-0.5 / p[0])
        
    # RETURNS  mean, sigma of the core, and proba at mean as in original array
    return mu, sig, np.exp(p[2]) * coeff * np.sum(hin)

def plot_increments(imgs,lag,labels,colors,styles):
    ivmax = abs(imgs[0].min()) + abs(imgs[0].max())
    x_scales = []
    y_scales = []
    increments_data = []
    increments_data_tmp = []
    for img_index, img in enumerate(imgs):
        _, incr_val, incr_hist, _ = increments(img, lag, noper=True, ivmax=ivmax, nbins=256)
        if img_index == 0: # Noisy
            _, sig, _ = find_gauss_core_hist(incr_val, incr_hist, fac=2)
            x_scales.append(1 / sig)
        if img_index == 2: # Truth
            y_scales.append(1.0)
        increments_data_tmp.append([incr_val, incr_hist / incr_hist.sum() / (incr_val[1] - incr_val[0])])
    increments_data.append(increments_data_tmp)
    x_scales = np.array(x_scales)
    y_scales = np.array(y_scales)
    increments_data = np.array(increments_data)
    increments_data[:, :, 0, :] *= np.expand_dims(np.expand_dims(x_scales, axis=-1), axis=-1)
    increments_data[:, :, 1, :] *= np.expand_dims(np.expand_dims(y_scales, axis=-1), axis=-1)
    
    frac = 0.7
    fig = plt.figure(figsize=(10, 10))
    for img_index, img in enumerate(imgs):
        incr_val = increments_data[img_index, 0]
        incr_hist = increments_data[img_index, 1]
        z=len(incr_val)
        plt.plot(incr_val[int(0.5*z*frac):int(-0.5*z*frac)], incr_hist[int(0.5*z*frac):int(-0.5*z*frac)], label=labels[img_index], color=colors[img_index], linestyle=styles[img_index])
    plt.title(f'Increments PDF at $l = {lag}$ px',fontsize=30)
    plt.legend(loc=3,bbox_to_anchor=(-0.3,0.6),fontsize=20)
    plt.ylim(10**(-5),10**1.3)
    plt.yscale('log')
    fig.text(0.5, 0.02, r'$\delta I / \sigma_l$', ha='center',fontsize=30)
    fig.text(0.01, 0.5, r'$p(\delta I)$', va='center', rotation='vertical',fontsize=30)
    plt.tight_layout()
    return