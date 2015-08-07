import healpy as hp
import numpy as np
from matplotlib import pyplot as plt
import CG_functions as CG
import time
import sys
import camber as cb
import MH_module as MH
import PS2param_module as PS2P
import Jeff_idea as JJi

nside = 2048
lmax=2200
generate_new_data = 0

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
nl = 1.7504523623688016e-16*1e12 * np.ones(2500) *2


# Gaussian beam fwhm 5 arcmin 
#bl = CG.gaussian_beam(2500,5)
#bl = CG.gaussian_beam(2500,5*np.sqrt(hp.nside2pixarea(nside,degrees=True))*60)
bl = CG.gaussian_beam(2500,15)

# Spectrum according to parameter defined above
if generate_new_data==1:
    Cl = cb.generate_spectrum(dd)
    lmax_temp = Cl.shape[0]-1
    alm = hp.synalm(Cl[:,1])
    dlm = hp.almxfl(alm,bl[:lmax_temp+1])
    nlm = hp.synalm(nl[:lmax_temp+1])
    dlm = dlm+nlm
    dlm_filt = CG.filter_alm(dlm,lmax+1)
    print "dataset generated"

else:
    dlm =np.load("Dataset_planck2015_35009eminus4_whitenoise.npy")
    print "dataset read"
#################################################


# Could be used for asymetric proposal, but now only for first guess
x_mean = np.array([0.02222,0.1197,0.078,3.089,0.9655,67.31])


#cov_mat from tableTT_lowEB downloaded from PLA, used in proposal
cov_new = np.load("cov_tableTT_lowEB_2_3_5_6_7_23.npy")


# priors parameters here central value and 1/sigma**2
# if invvar = 0, equivalent to uniform prior
priors_central = np.array([0,0,0.07,0,0,0])
priors_invvar = np.array([0,0,1/0.02**2,0,0,0])




#################################################



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
    # generate first guess parameters
    guess_param = PS2P.prop_dist_form_params(x_mean_temp,cov_new_temp)
    # generate first fluctuation map
    dd2 = cb.update_dic(dd,guess_param,string_temp)
    cl = cb.generate_spectrum(dd2)[:,1]
    cl[:2] = 1.e-35
    renorm = CG.renorm_term(cl,bl,nl)
    fluc = hp.almxfl(CG.generate_w1term(cl[:lmax+1],bl[:lmax+1],nl[:lmax+1]) + CG.generate_w0term(cl[:lmax+1]),renorm)
    # the core of the MCMC
    testss = np.array(MH.MCMC_log_Jeff_test(guess_param, JJi.target_distrib_newrescale,PS2P.prop_dist_form_params, PS2P.prop_func_form_params,niter,PS2P.Gaussian_priors_func,[[dlm,string_temp,dd,nl[:lmax+1],bl[:lmax+1]],[cl[:lmax+1],fluc]],[x_mean_temp*0,np.matrix(cov_new_temp)],[priors_central_temp,priors_invvar_temp]))
    np.save("chain_%s_%s_%d_%d.npy"%(save_title,str(which_par).replace(',','').replace('[','').replace(']','').replace(' ',''),np.random.randint(0,100000),niter),testss)
    return testss





def run_MCMC_new(which_par,niter,save_title, renorm_var):
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
    # generate first guess parameters
    guess_param = PS2P.prop_dist_form_params(x_mean_temp,cov_new_temp)
    print "initial guess = ", guess_param
    # generate first fluctuation map
    dd2 = cb.update_dic(dd,guess_param,string_temp)
    cl = cb.generate_spectrum(dd2)[:,1]
    cl[:2] = 1.e-35
    renorm = CG.renorm_term(cl,bl,nl)
    fluc = hp.almxfl(CG.generate_w1term(cl[:lmax+1],bl[:lmax+1],nl[:lmax+1]) + CG.generate_w0term(cl[:lmax+1]),renorm)
    mf = hp.almxfl(CG.generate_mfterm(dlm,cl[:lmax+1],bl[:lmax+1],nl[:lmax+1]),renorm)
    # the core of the MCMC
    testss = np.array(MH.MCMC_log_Jeff_new(guess_param, JJi.target_new,PS2P.prop_dist_form_params, PS2P.prop_func_form_params,niter,PS2P.Gaussian_priors_func,[[dlm,string_temp,dd,nl[:lmax+1],bl[:lmax+1]],[cl[:lmax+1],fluc,mf]],[x_mean_temp*0,np.matrix(cov_new_temp)],[priors_central_temp,priors_invvar_temp]))
    np.save("chain_%s_%s_%d_%d.npy"%(save_title,str(which_par).replace(',','').replace('[','').replace(']','').replace(' ',''),np.random.randint(0,100000),niter),testss)
    return testss



############################### test functions ##############

def plot_tests():
    spec_mf_b = hp.alm2cl(hp.almxfl(mf_lm_new,bl))
    spec_mf = hp.alm2cl((mf_lm_new))
    spec = hp.alm2cl(dlm)
    diff = hp.alm2cl(dlm - (hp.almxfl(mf_lm_new,bl)))
    fluc_l = hp.alm2cl(fluc)
    nnn = hp.alm2cl(dlm-hp.almxfl(mf_lm_new,bl))
    plt.figure()
    plt.plot(spec_mf,label='mean field ($\hat{s}$)')
    plt.plot(spec_mf_b,label='beamed mean field ($A\hat{s}$)')
    plt.plot(spec,label='data (d)')
    plt.plot(spec - spec_mf_b,label='($d_\ell - (A\hat{s})_\ell$)')
    plt.plot(diff,label='($d - (A\hat{s})$)$_\ell$')
    plt.plot(fluc_l,"-.",label='($\hat{f}$)$_\ell$')
    plt.plot(nl,"-.",label="noise")
    plt.plot(cl,"-.",label="trial spectrum")
    plt.yscale("log")
    plt.legend(loc = 'best')




def solve_CG(params_i,dd):
    dd = cb.update_dic(dd,params_i,strings)
    cl = cb.generate_spectrum(dd)[:,1]
    # define the first guess, the b from eq 25 since the code was design with this at first (might not be best idea ever)
    b_mf = CG.generate_mfterm(dlm_filt,cl[:lmax+1],bl[:lmax+1],nl[:lmax+1])
    b_w1 = CG.generate_w1term(cl[:lmax+1],bl[:lmax+1],nl[:lmax+1])
    b_w0 = CG.generate_w0term(cl[:lmax+1])
    b = b_mf + b_w1 + b_w0
    ###### left hand side factor
    renorm= np.sqrt(cl[:lmax+1])/(1+cl[:lmax+1]*bl[:lmax+1]**2/(nl[:lmax+1]))
    out = hp.almxfl(b,renorm)
    out_mf = hp.almxfl(b_mf,renorm)
    out_w1 = hp.almxfl(b_w1,renorm)
    out_w0 = hp.almxfl(b_w0,renorm)
    return out,out_mf,out_w0,out_w1


def test_3rd_Term(list_guess):
    #initial guess
    guess_param = PS2P.prop_dist_form_params(x_mean,cov_new)
    # generate first fluctuation map
    dd2 = cb.update_dic(dd,guess_param,strings)
    cl = cb.generate_spectrum(dd2)[:,1]
    cl[:2] = 1.e-35
    renorm = CG.renorm_term(cl,bl,nl)
    fluc = hp.almxfl(CG.generate_w1term(cl[:lmax+1],bl[:lmax+1],nl[:lmax+1]) + CG.generate_w0term(cl[:lmax+1]),renorm)
    Cl_i,fluc_i = [cl,fluc]
    list_save=[]
    for i in range(len(list_guess)):
        print i
        test_param = list_guess[i]
        Like_3,Cl_ip1,fluc_ip1,fluc_i_rescaled = test_fluctuation_term(test_param,[dlm,strings,dd2,nl[:lmax+1],bl[:lmax+1]],[Cl_i, fluc_i])
        Cl_i,fluc_i = [Cl_ip1,fluc_ip1]
        list_save.append([Like_3,Cl_ip1,fluc_ip1,fluc_i_rescaled])
    return list_save

def ren(fluc):
    return -1./2 * hp.almxfl(abs(ll[i][2])**2,1./nl*bl**2)
