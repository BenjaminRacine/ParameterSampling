import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse        
import ploter_parameters as plp
import camber as cb
import healpy as hp
import CG_functions as CG
import MH_module as MH
import PS2param_module as PS2P # defines w() Pi() priors etc.
import sys

try:
    from local_paths import *
except:
    print "you need to define local_paths.py, that defines, for example: \ncamb_dir = '/Users/benjar/Travail/camb/' \n and the output path for the temporary ini files: \noutput_camb = '../MH_MCMC/camb_ini/test1'"
    sys.exit()

# random number added to outputs for security (anti-overwrite)
# carefull though, most files aren't removed, whatchout memory
random_id = np.random.randint(0,100000)

# loads initial param file
dd  = cb.ini2dic(camb_dir+"Planck_params_params.ini")

# defines the strings of the variable we will use (as named in camb)
strings=np.array(['ombh2','omch2','re_optical_depth','scalar_amp(1)','scalar_spectral_index(1)','hubble'])

# defines the strings for the titles of the plors
titles = np.array(["$\Omega_b h^2$","$\Omega_c h^2$",r"$\tau$","$A_s$","$n_s$","$H_0$"])

#inital values for dataset generation: rest is planck 2015 
# also defining the output filename

dd["output_root"] = output_camb+'_%d'%random_id

dd['ombh2'] =  0.02222
dd['omch2'] =  0.1197 
dd['re_optical_depth'] = 0.078
dd['scalar_amp(1)'] = np.exp(3.089)*1e-10
dd['scalar_spectral_index(1)'] = 0.9655
dd['hubble'] = 67.31


########### Here we simulate the data set ############

# White noise spectrum, (Commander level, so low)
nl = 1.7504523623688016e-16*1e12 * np.ones(2500)

# Gaussian beam fwhm 5 arcmin 
bl = CG.gaussian_beam(2500,5)

# Spectrum according to parameter defined above
Cl = cb.generate_spectrum(dd)
lmax = Cl.shape[0]-1
alm = hp.synalm(Cl[:,1])
dlm = hp.almxfl(alm,bl[:lmax+1])
nlm = hp.synalm(nl[:lmax+1])
dlm = dlm+nlm



# Could be used for asymetric proposal, but now only for first guess
x_mean = np.array([0.02222,0.1197,0.078,3.089,0.9655,67.31])


#cov_mat from tableTT_lowEB downloaded from PLA, used in proposal
cov_new = np.load("cov_tableTT_lowEB_2_3_5_6_7_23.npy")


# priors parameters here central value and 1/sigma**2
# if invvar = 0, equivalent to uniform prior
priors_central = np.array([0,0,0.07,0,0,0])
priors_invvar = np.array([0,0,1/0.02**2,0,0,0])



def plot_input(dlm):
    hp.mollview(hp.alm2map(dlm,1024),title="Input data",unit="$\mu K_{\mathrm{CMB}}$")
    plt.figure()
    plt.loglog(Cl[:,1],"r")





def run_MCMC(which_par,niter,save_title, renorm_var):
    """
    Functions that runs the MCMC for a given set of parameters

    Keyword Arguments:
    which_par -- a list of indices, corresponding to the order defined above, exemple [0,2] means ombh2,tau if order is [ombh2,omch2,tau,As,ns,H0]
    niter -- number of iterations in MCMC
    renorm_var -- factor multipling the variance, to play around for better acceptance rate.
    """
    cov_new_temp = cov_new[which_par,:][:,which_par] * renorm_var
    string_temp = strings[which_par]
    titles_temp = titles[which_par]
    x_mean_temp = x_mean[which_par]
    priors_central_temp = priors_central[which_par]
    priors_invvar_temp = priors_invvar[which_par]
    print titles_temp
    guess_param = PS2P.prop_dist_form_params(x_mean_temp,cov_new_temp)
    testss = np.array(MH.MCMC_log(guess_param, PS2P.functional_form_params_n,PS2P.prop_dist_form_params, PS2P.prop_func_form_params,niter,PS2P.Gaussian_priors_func,[dlm,string_temp,dd,nl,bl],[x_mean_temp*0,np.matrix(cov_new_temp)],[priors_central_temp,priors_invvar_temp]))
    #print "%.2f rejected; %.2f accepted; %.2f Lucky accepted"%((flag==0).mean(),(flag==1).mean(),(flag==2).mean())
    np.save("chain_%s_%s_%d_%d.npy"%(save_title,str(which_par).replace(',','').replace('[','').replace(']','').replace(' ',''),np.random.randint(0,100000),niter),testss)
    return testss





