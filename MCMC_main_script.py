mport healpy as hp
import numpy as np
from matplotlib import pyplot as plt
import CG_functions as CG
import time
import sys
import camber as cb
import MH_module as MH
import PS2param_module as PS2P
import Jeff_idea as JJi
import multiprocessing
import itertools
from logger import *
import psutil
import logging
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',filename='log/%s_%d.log'%(__file__.replace(".py",""),os.getpid()),level=logging.DEBUG,stream=sys.stdout)
#open_log(1, "MCMC_main_script_posteriorcov_1")

logging.warning(psutil.virtual_memory())
#print "test"
#sys.stdout.flush()

#close_log(1)



N_proc_tot = multiprocessing.cpu_count()
print "number of available proc: %d"%N_proc_tot

Method = 1
N_iter = 30000
print "N_iter = ",N_iter
title_save = "renopt_sym_900SNR1_final_run"

logging.warning('is when this event was logged.')
#sys.stdout.flush()

nside = 2048#128
lmax= 1500#2200
generate_new_data = 0

#method = sys.argv[1]
#define global variable that will be used in the complex2real function: Nasty... 
global index_pos
index_pos = np.array(list(itertools.chain.from_iterable([[hp.Alm.getidx(lmax, l, m) for m in range(1,l+1)] for l in range(lmax+1)])))

try:
    from local_paths import *
except:
    print "you need to define local_paths.py, that defines, for example: \ncamb_dir = '/Users/benjar/Travail/camb/' \n and the output path for the temporary ini files: \noutput_camb = '../MH_MCMC/camb_ini/test1'"
    sys.exit()

# random number added to outputs for security (anti-overwrite)
# carefull though, most files aren't removed, whatchout memory
random_id = np.random.randint(0,100000)

# loads initial param file
dd  = cb.ini2dic(camb_dir+"Planck_params_params.ini")#_nolens.ini")

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
#nl = 1.7504523623688016e-16*1e12 * np.ones(2500)
#nl = 1.7504523623688016e-16*1e12 * np.ones(2500) *2


# Gaussian beam fwhm 5 arcmin 
#bl = CG.gaussian_beam(2500,5)
#bl = CG.gaussian_beam(2500,5*np.sqrt(hp.nside2pixarea(nside,degrees=True))*60)
bl = CG.gaussian_beam(2500,13)

# Spectrum according to parameter defined above
if generate_new_data==1:
    Cl = cb.generate_spectrum(dd)
    # White noise level defined so that SNR=1 at \ell of 1700
    nl = Cl[900,1]*bl[900]**2*np.ones(2500)
    lmax_temp = Cl.shape[0]-1
    alm = hp.synalm(Cl[:,1])
    dlm = hp.almxfl(alm,bl[:lmax_temp+1])
    nlm = hp.synalm(nl[:lmax_temp+1])
    dlm = dlm+nlm
    #np.save("Dataset_planck2015_900SNR1_13arcmin.npy",dlm)
    plt.figure()
    ell = np.arange(lmax)*np.arange(1,lmax+1)
    plt.plot(ell*(Cl[:lmax,1]*bl[:lmax]**2),label = "$C_\ell b_\ell^2$")
    plt.plot(ell*(nl[:lmax]),label="$n_\ell$")
    plt.plot(ell*(Cl[:lmax,1]*bl[:lmax]**2+nl[:lmax]),"--",label = "$C_\ell b_\ell^2 + n_\ell$")
    plt.ylabel("$\ell (\ell +1) C_\ell$")
    plt.xlabel("$\ell$")
    plt.axvline(900)
    plt.yscale("log")
    plt.legend(loc="best")
    plt.savefig("plots/powerspectrum_WMAPtype.png")
    print "dataset generated"

else:
    dlm =np.load("Dataset_planck2015_900SNR1_13arcmin.npy")
    #dlm =np.load("Dataset_planck2015_175eminus4_whitenoise_7arcmin.npy")
    #dlm =np.load("Dataset_planck2015__175eminus4_7arcmin_128_200.npy")
    print "dataset read %s"%"Dataset_planck2015_900SNR1_13arcmin.npy"


Cl = cb.generate_spectrum(dd)
nl = Cl[900,1]*bl[900]**2*np.ones(2500)

dlm = CG.filter_alm(dlm,lmax)
#################################################
dlm[[hp.Alm.getidx(lmax,0,0),hp.Alm.getidx(lmax,1,0),hp.Alm.getidx(lmax,1,1)]]=0
nl = nl[:lmax+1]
bl=bl[:lmax+1]


print nl[5]
    


# Could be used for asymetric proposal, but now only for first guess
x_mean = np.array([0.02222,0.1197,0.078,3.089,0.9655,67.31])
#np.load("mean_from_posteriors_highres.npy")#np.array([0.02222,0.1197,0.078,3.089,0.9655,67.31])                            

#cov_mat from tableTT_lowEB downloaded from PLA, used in proposal
cov_new = np.load("covar_new_900_40000samples_burnin1000_forpaper.npy")
#np.load("cov_tableTT_lowEB_2_3_5_6_7_23.npy") This is for the first run, to produce the chains for estimating the posterior variance.



# priors parameters here central value and 1/sigma**2
# if invvar = 0, equivalent to uniform prior
priors_central = np.array([0,0,0.07,0,0,0])
priors_invvar = np.array([0,0,1/0.02**2,0,0,0])




#################################################



def run_MCMC_new(which_par,niter,save_title, renorm_var,firstiter=0,seed="none",guess = "random"):
    """
    Functions that runs the MCMC for a given set of parameters

    Keyword Arguments:
    which_par -- a list of indices, corresponding to the order defined above, exemple [0,2] means ombh2,tau if order is [ombh2,omch2,t
au,As,ns,H0]
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
    if guess=="random":
        guess_param = PS2P.prop_dist_form_params(x_mean_temp,cov_new_temp)
    else :
        guess_param = guess
    print "initial guess = ", guess_param
    # generate first fluctuation map
    dd2 = cb.update_dic(dd,x_mean_temp,string_temp)
    cl = cb.generate_spectrum(dd2)[:lmax+1,1]
    cl[:2] = 1.e-35
    renorm = CG.renorm_term(cl,bl,nl)
    fluc = hp.almxfl(CG.generate_w1term(cl[:lmax+1],bl[:lmax+1],nl[:lmax+1]) + CG.generate_w0term(cl[:lmax+1]),renorm)
    mf = hp.almxfl(CG.generate_mfterm(dlm,cl[:lmax+1],bl[:lmax+1],nl[:lmax+1]),renorm)
    # the core of the MCMC
    tt1 = time.time()
    save_string = "outputs/chain_new_%s_%s_%d_%d"%(save_title,str(which_par).replace(',','').replace('[','').replace(']','').replace(' ',''),np.random.randint(0,100000),niter)
    print save_string
    sys.stdout.flush()
    testss = np.array(MH.MCMC_log_Jeff_new(guess_param, JJi.target_new,PS2P.prop_dist_form_params, PS2P.prop_func_form_params,niter,PS2P.Gaussian_priors_func,firstiter,seed,save_string,[[dlm,string_temp,dd,nl[:lmax+1],bl[:lmax+1]],[cl[:lmax+1],fluc,mf]],[x_mean_temp*0,np.matrix(cov_new_temp)],[priors_central_temp,priors_invvar_temp]))
    print time.time() - tt1
    np.save(save_string+".npy",testss)
    return testss




def run_MCMC_ex(which_par,niter,save_title, renorm_var,firstiter=0,seed="none",guess="random"):
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
    sys.stdout.flush()
    dd2 = cb.update_dic(dd,x_mean_temp,string_temp)
    cl = cb.generate_spectrum(dd2)[:lmax+1,1]
    cl[:2] = 1.e-35
    if guess=="random":
        guess_param = PS2P.prop_dist_form_params(x_mean_temp,cov_new_temp)
    else :
        guess_param = guess
    tt1 = time.time()
    save_string = "outputs/chain_ex_%s_%s_%d_%d"%(save_title,str(which_par).replace(',','').replace('[','').replace(']','').replace(' ',''),np.random.randint(0,100000),niter)
    print save_string
    sys.stdout.flush()
    testss = np.array(MH.MCMC_log(guess_param, JJi.functional_form_params_n,PS2P.prop_dist_form_params, PS2P.prop_func_form_params,niter,PS2P.Gaussian_priors_func,firstiter,seed,save_string,[dlm,string_temp,dd,nl,bl,cl],[x_mean_temp*0,np.matrix(cov_new_temp)],[priors_central_temp,priors_invvar_temp]))
    print time.time() - tt1
    np.save(save_string+".npy",testss)
    return testss





if Method==0:
    tt = time.time()
    test_ex_REAL = run_MCMC_ex([0,1,2,3,4,5],N_iter,title_save,2.4**2/6)
    print time.time()-tt
elif Method==1:
    tt = time.time()
    test_new_REAL = run_MCMC_new([0,1,2,3,4,5],N_iter,title_save,2.4**2/6)
    print time.time()-tt

sys.stdout.flush()
logging.warning(psutil.virtual_memory())


#close_log(1)  
