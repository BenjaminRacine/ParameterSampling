import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse        
import ploter_parameters as plp
import camber as cb
import healpy as hp
import CG_functions as CG
import MH_module as MH


def prop_func_form_params(param1,param2,*arg):
    """
    Returns w(theta_i|theta_i+1), which is here a gaussian distribution with a given covariance and mean.
    Keyword Arguments:
    *args are:
    x_mean -- the mean vector (np.array)
    Cov -- covariance Matrix (np.matrix)
    """
    return np.log(MH.simple_2D_Gauss(param1-param2,arg[0],arg[1]))


def prop_dist_form_params(*arg):
    """
    Draw random numbers from a gaussian distribution with a given covariance and mean.
    Keyword Arguments:
    *args are:
    x_mean -- the mean vector (np.array)
    Cov -- covariance Matrix (np.matrix)
    """
    return np.random.multivariate_normal(*arg)

def test_loglike(dlm,Cl,noise,beam):
    """
    returns the log likelihood for a given Cl, beam, noise, and data.
    """
    lmax = Cl.shape[0]
    tt_exp = -1./2 * np.real(np.vdot(dlm.T,hp.almxfl(dlm,1/(beam[:lmax]**2*Cl[:,1]+noise[:lmax]))))
    #plt.plot(Cl[:,1])
    tt_det = - 1./2 *(np.arange(1,lmax+1)*np.log((noise[:lmax]+Cl[:,1]*beam[:lmax]**2))).sum() 
    tt_f = tt_exp  + tt_det
    return tt_exp,tt_det,tt_f#,Cl[:,1]





def functional_form_params_n(x,*arg):
    """
    Keyword Arguments:
    x -- the vector of params

    *args are:
    dlm -- input map
    x_str -- the dictionary strings corresponding to x
    params -- a camber dictionnary
    noise -- a noise power spectrum
    beam -- a beam power spectrum
    """
    dlm = arg[0]
    strings = arg[1]
    params = arg[2].copy()
    noise = arg[3]
    beam = arg[4]
    #params["output_root"] = '../Codes/CG_git/MH_MCMC/camb_ini/test%d'%np.random.randint(100)
    for i in range(np.size(x)):
        #print strings[i]
        if strings[i]=='scalar_amp(1)':
            #print params[strings[i]]
            params[strings[i]]=np.exp(x[i])*1e-10
            #print params[strings[i]]
        else:
            params[strings[i]]=x[i]
    Cl = cb.generate_spectrum(params)
    lmax = Cl.shape[0]
    tt = np.real(np.vdot(dlm.T,hp.almxfl(dlm,1/(beam[:lmax]**2*Cl[:,1]+noise[:lmax]))))
    #determinant is the product of the diagonal element: in log:
    tt = -1/2. * tt  - 1./2 *(np.arange(1,lmax+1)*np.log(noise[:lmax]+Cl[:,1]*beam[:lmax]**2)).sum()
    return tt,Cl[:,1]



def Gaussian_priors_func(guesses,central,invvar):
    """
    Returns the priors given an array of guesses. 
    Keyword Arguments:
    guesses -- the input guesses from the MCMC (np.array)
    central --  the central value of the prior (np.array)
    invsigmas -- inverse variance (here only diagonal)
    
    Note: the returned value is in "log space"
    """
    return [-0.5 * np.dot(np.dot((guesses[i]-central[i]).T,invvar[i]),guesses[i]-central[i]) for i in range(len(central))]

