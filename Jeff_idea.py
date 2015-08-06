import numpy as np
from matplotlib import pyplot as plt
import camber as cb
import healpy as hp
import CG_functions as CG
import MH_module as MH




class target_obj:
    def __init__(self,dlm,x_str,params,noise,beam):
        """
        dlm -- input map
        x_str -- the dictionary strings corresponding to x
        params -- a camber dictionnary
        noise -- a noise power spectrum
        beam -- a beam power spectrum
        """
        target_obj.dlm = dlm
        target_obj.strings = x_str
        target_obj.params = params
        target_obj.nl = noise
        target_obj.bl = beam
    
    def get_spec(self,x):
        parameters = cb.update_dic(self.params,x,self.strings)
        return cb.generate_spectrum(parameters)
        
def target_distrib(guess, *arg):
    """
    Keyword Arguments:
    x -- the vector of params

    *args are:
    arg[0] -- a target class object
    arg[1] -- Cl_old, fluc_lm_old
    """
    #print guess, arg
    dlm,strings,params,nl,bl = arg[0]
    Cl_old, fluc_lm_old = arg[1]
    dd = cb.update_dic(params,guess,strings)
    Cl_new = cb.generate_spectrum(dd)[:,1]
    Cl_new[:2] = 1.e-35
    print "new = ",Cl_new[50]
    print "old = ",Cl_old[50]
    renorm = CG.renorm_term(Cl_new,bl,nl)
    mf_lm_new = hp.almxfl(CG.generate_mfterm(dlm,Cl_new,bl,nl),renorm)
    fluc_lm_type2 = hp.almxfl(CG.generate_w1term(Cl_new,bl,nl)+CG.generate_w0term(Cl_new),renorm)
    print "new = ",fluc_lm_type2[50]
    print "old = ",fluc_lm_old[50]
    fluc_lm_determ = hp.almxfl(fluc_lm_old,np.sqrt(Cl_new/Cl_old))
    tt1 = -1/2.*np.real(np.vdot((dlm-hp.almxfl(mf_lm_new,bl)).T,hp.almxfl((dlm-hp.almxfl(mf_lm_new,bl)),1/nl)))
    print tt1
    tt2 = -1/2. *np.real(np.vdot((mf_lm_new).T,hp.almxfl((mf_lm_new),1./Cl_new)))
    print tt2
    tt3 = -1/2. *np.real(np.vdot((fluc_lm_determ).T,hp.almxfl((fluc_lm_determ),1/nl*bl**2)))
    print tt3
    #tt4 = - 1./2 *(np.arange(1,np.size(Cl_new)+1)*np.log(Cl_new)).sum()
    #print tt4
    return [tt1,tt2,tt3],Cl_new,fluc_lm_type2

def test_loglike_Ji(guess, *arg):
    """
    Keyword Arguments:
    x -- the vector of params

    *args are:
    arg[0] -- a target class object
    arg[1] -- Cl_old, fluc_lm_old
    """
    #print guess, arg
    dlm,strings,params,nl,bl,Cl_new = arg[0]
    Cl_old, fluc_lm_old = arg[1]
    Cl_new[:2] = 1.e-35
    Cl_old[:2] = 1.e-35
    renorm = CG.renorm_term(Cl_new,bl,nl)
    mf_lm_new = hp.almxfl(CG.generate_mfterm(dlm,Cl_new,bl,nl),renorm)
    fluc_lm_type2 = hp.almxfl(CG.generate_w1term(Cl_new,bl,nl)+CG.generate_w0term(Cl_new),renorm)
    fluc_lm_determ = hp.almxfl(fluc_lm_old,np.sqrt(Cl_new/Cl_old))
    tt = -1/2.*np.real(np.vdot((dlm-hp.almxfl(mf_lm_new,bl)).T,hp.almxfl((dlm-hp.almxfl(mf_lm_new,bl)),1./nl)))
    tt2 = -1/2. *np.real(np.vdot((mf_lm_new).T,hp.almxfl((mf_lm_new),1./Cl_new)))
    tt3 = -1/2. *np.real(np.vdot((fluc_lm_determ).T,hp.almxfl((fluc_lm_determ),1/nl*bl**2)))

    tt4 = - 1./2 *(np.arange(1,np.size(Cl_new)+1)*np.log(Cl_new)).sum()
    return tt,tt2,tt3,tt4,Cl_new,fluc_lm_type2



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



def target_distrib_newrescale(guess, *arg):
    """
    Keyword Arguments:
    x -- the vector of params

    *args are:
    arg[0] -- a target class object
    arg[1] -- Cl_old, fluc_lm_old
    """
    #print guess, arg
    dlm,strings,params,nl,bl = arg[0]
    Cl_old, fluc_lm_old = arg[1]
    dd = cb.update_dic(params,guess,strings)
    #generate new spectrum from new params
    lmax = len(Cl_old)
    Cl_new = cb.generate_spectrum(dd)[:lmax,1]
    # avoid dividing by 0
    Cl_new[:2] = 1.e-35
    print "new = ",Cl_new[50]
    print "old = ",Cl_old[50]
    # renormalization, i.e. lhs part of eq (24) and (25)
    renorm = CG.renorm_term(Cl_new,bl,nl)
    # generate mean field map using new PS
    mf_lm_new = hp.almxfl(CG.generate_mfterm(dlm,Cl_new,bl,nl),renorm)
    # generate fluctuation map using new PS, this is actually the 'step 2', with acceptance 1, won't be used anymore in this function
    fluc_lm_type2 = hp.almxfl(CG.generate_w1term(Cl_new,bl,nl)+CG.generate_w0term(Cl_new),renorm)
    print "new = ",fluc_lm_type2[50]
    print "old = ",fluc_lm_old[50]
    # get deterministic fluctuation map (new rescaling with noise)
    fluc_lm_determ = hp.almxfl(fluc_lm_old,np.sqrt((1./Cl_old+bl**2/nl))/np.sqrt((1./Cl_new+bl**2/nl)))
    # Chi2 part of the likelihood
    tt1 = -1/2.*np.real(np.vdot((dlm-hp.almxfl(mf_lm_new,bl)).T,hp.almxfl((dlm-hp.almxfl(mf_lm_new,bl)),1/nl)))
    print tt1
    # "mean field part" of the likelihood
    tt2 = -1/2. *np.real(np.vdot((mf_lm_new).T,hp.almxfl((mf_lm_new),1./Cl_new)))
    print tt2
    # "fluctuation part" of the likelihood
    tt3 = -1/2. *np.real(np.vdot((fluc_lm_determ).T,hp.almxfl((fluc_lm_determ),1/nl*bl**2+1/Cl_new)))
    print tt3
    # we return Cl_new and fluc_lm_type2 for next iteration.
    return [tt1,tt2,tt3],Cl_new,fluc_lm_type2



def test_fluctuation_term(guess, *arg):
    """
    Keyword Arguments:
    x -- the vector of params

    *args are:
    arg[0] -- a target class object
    arg[1] -- Cl_old, fluc_lm_old
    """
    #print guess, arg
    dlm,strings,params,nl,bl = arg[0]
    Cl_old, fluc_lm_old = arg[1]
    dd = cb.update_dic(params,guess,strings)
    #generate new spectrum from new params
    lmax = len(Cl_old)
    Cl_new = cb.generate_spectrum(dd)[:lmax,1]
    # avoid dividing by 0
    Cl_new[:2] = 1.e-35
    print "new = ",Cl_new[50]
    print "old = ",Cl_old[50]
    # renormalization, i.e. lhs part of eq (24) and (25)
    renorm = CG.renorm_term(Cl_new,bl,nl)

    # generate fluctuation map using new PS, this is actually the 'step 2', with acceptance 1, won't be used anymore in this function
    fluc_lm_type2 = hp.almxfl(CG.generate_w1term(Cl_new,bl,nl)+CG.generate_w0term(Cl_new),renorm)
    print "new = ",fluc_lm_type2[50]
    print "old = ",fluc_lm_old[50]
    # get deterministic fluctuation map (new rescaling with noise)
    fluc_lm_determ = hp.almxfl(fluc_lm_old,np.sqrt((1./Cl_old+bl**2/nl))/np.sqrt((1./Cl_new+bl**2/nl)))
    # "fluctuation part" of the likelihood
    tt3 = -1/2. *np.real(np.vdot((fluc_lm_determ).T,hp.almxfl((fluc_lm_determ),1/nl*bl**2)))
    print tt3
    # we return Cl_new and fluc_lm_type2 for next iteration.
    return tt3,Cl_new,fluc_lm_type2,fluc_lm_determ








def Step_MC_withnoise(guess, *arg):
    """
    Keyword Arguments:
    x -- the vector of params

    *args are:
    arg[0] -- a target class object
    arg[1] -- Cl_old, fluc_lm_old
    """
    #print guess, arg
    dlm,strings,params,nl,bl = arg[0]
    Cl_old, fluc_lm_GS,mf_old = arg[1]
    dd = cb.update_dic(params,guess,strings)
    #generate new spectrum from new params
    lmax = len(Cl_old)
    Cl_new = cb.generate_spectrum(dd)[:lmax,1]
    # avoid dividing by 0
    Cl_new[:2] = 1.e-35
    print "new = ",Cl_new[50]
    print "old = ",Cl_old[50]
    # renormalization, i.e. lhs part of eq (24) and (25)
    renorm = CG.renorm_term(Cl_new,bl,nl)
    # generate mean field map using new PS
    mf_lm_new = hp.almxfl(CG.generate_mfterm(dlm,Cl_new,bl,nl),renorm)
    # get deterministic fluctuation map (new rescaling with noise)
    fluc_lm_determ = hp.almxfl(fluc_lm_GS,np.sqrt((1./Cl_old+bl**2/nl))/np.sqrt((1./Cl_new+bl**2/nl)))
    # get GS fluctuation map "for next iteration"
    fluc_lm_GS_next = hp.almxfl(CG.generate_w1term(Cl_new,bl,nl)+CG.generate_w0term(Cl_new),renorm)
    # Chi2 part of the likelihood
    return fluc_lm_determ,mf_lm_new, fluc_lm_GS_next, Cl_new


def diff_fixed_like_withnoise(fluc_lm_determ,fluc_lm_GS,mf_lm_new,mf_lm_old,Cl_new,Cl_old, *arg):
    """
    Keyword Arguments:
    x -- the vector of params

    *args are:
    arg[0] -- a target class object
    arg[1] -- Cl_old, fluc_lm_old
    """
    #print guess, arg
    dlm,strings,params,nl,bl = arg[0]
    tt1_new = -1/2.*np.real(np.vdot((dlm-hp.almxfl(mf_lm_new,bl)).T,hp.almxfl((dlm-hp.almxfl(mf_lm_new,bl)),1/nl)))
    tt1_old = -1/2.*np.real(np.vdot((dlm-hp.almxfl(mf_lm_old,bl)).T,hp.almxfl((dlm-hp.almxfl(mf_lm_old,bl)),1/nl)))
    print "chi**2 = ",tt1_new-tt1_old
    # "mean field part" of the likelihood
    tt2_new = -1/2. *np.real(np.vdot((mf_lm_new).T,hp.almxfl((mf_lm_new),1./Cl_new)))
    tt2_old = -1/2. *np.real(np.vdot((mf_lm_old).T,hp.almxfl((mf_lm_old),1./Cl_old)))
    print "'mf like' = ",tt2_new-tt2_old
    # "fluctuation part" of the likelihood
    tt3_new = -1/2. *np.real(np.vdot((fluc_lm_determ).T,hp.almxfl((fluc_lm_determ),1/nl*bl**2+1/Cl_new)))
    tt3_old = -1/2. *np.real(np.vdot((fluc_lm_GS).T,hp.almxfl((fluc_lm_GS),1/nl*bl**2+1/Cl_old)))
    print "'fluc like' = ",tt3_new-tt3_old
    tt_new = tt1_new + tt2_new + tt3_new
    tt_old = tt1_old + tt2_old + tt3_old
    return tt_new,tt_old



def Step_MC(guess, *arg):
    """
    Keyword Arguments:
    x -- the vector of params

    *args are:
    arg[0] -- a target class object
    arg[1] -- Cl_old, fluc_lm_old
    """
    #print guess, arg
    dlm,strings,params,nl,bl = arg[0]
    Cl_old, fluc_lm_GS,mf_old = arg[1]
    dd = cb.update_dic(params,guess,strings)
    #generate new spectrum from new params
    lmax = len(Cl_old)
    Cl_new = cb.generate_spectrum(dd)[:lmax,1]
    # avoid dividing by 0
    Cl_new[:2] = 1.e-35
    print "new = ",Cl_new[50]
    print "old = ",Cl_old[50]
    # renormalization, i.e. lhs part of eq (24) and (25)
    renorm = CG.renorm_term(Cl_new,bl,nl)
    # generate mean field map using new PS
    mf_lm_new = hp.almxfl(CG.generate_mfterm(dlm,Cl_new,bl,nl),renorm)
    # get deterministic fluctuation map (new rescaling with noise)
    fluc_lm_determ = hp.almxfl(fluc_lm_GS,np.sqrt((1./Cl_old))/np.sqrt((1./Cl_new)))
    # get GS fluctuation map "for next iteration"
    fluc_lm_GS_next = hp.almxfl(CG.generate_w1term(Cl_new,bl,nl)+CG.generate_w0term(Cl_new),renorm)
    # Chi2 part of the likelihood
    return fluc_lm_determ,mf_lm_new, fluc_lm_GS_next, Cl_new

def diff_fixed_like(fluc_lm_determ,fluc_lm_GS,mf_lm_new,mf_lm_old,Cl_new,Cl_old, *arg):
    """
    Keyword Arguments:
    x -- the vector of params

    *args are:
    arg[0] -- a target class object
    arg[1] -- Cl_old, fluc_lm_old
    """
    #print guess, arg
    dlm,strings,params,nl,bl = arg[0]
    tt1_new = -1/2.*np.real(np.vdot((dlm-hp.almxfl(mf_lm_new,bl)).T,hp.almxfl((dlm-hp.almxfl(mf_lm_new,bl)),1/nl)))
    tt1_old = -1/2.*np.real(np.vdot((dlm-hp.almxfl(mf_lm_old,bl)).T,hp.almxfl((dlm-hp.almxfl(mf_lm_old,bl)),1/nl)))
    print "chi**2 = ",tt1_new-tt1_old
    # "mean field part" of the likelihood
    tt2_new = -1/2. *np.real(np.vdot((mf_lm_new).T,hp.almxfl((mf_lm_new),1./Cl_new)))
    tt2_old = -1/2. *np.real(np.vdot((mf_lm_old).T,hp.almxfl((mf_lm_old),1./Cl_old)))
    print "'mf like' = ",tt2_new-tt2_old
    # "fluctuation part" of the likelihood
    tt3_new = -1/2. *np.real(np.vdot((fluc_lm_determ).T,hp.almxfl((fluc_lm_determ),1/nl*bl**2)))
    tt3_old = -1/2. *np.real(np.vdot((fluc_lm_GS).T,hp.almxfl((fluc_lm_GS),1/nl*bl**2)))
    print "'fluc like' = ",tt3_new-tt3_old
    tt_new = tt1_new + tt2_new + tt3_new
    tt_old = tt1_old + tt2_old + tt3_old
    return tt_new,tt_old


def target_new(guess, *arg):
    """
    Keyword Arguments:
    guess -- the vector of params

    *args are:
    arg[0] -- dlm,strings,params,nl,bl
    arg[1] -- Cl_old, fluc_lm_GS, mf_old
    """
    dlm,strings,params,nl,bl = arg[0]
    Cl_old, fluc_lm_GS, mf_lm_old = arg[1]
    fluc_lm_determ, mf_lm_new, fluc_lm_GS_next, Cl_new = Step_MC(guess, *arg)
    likes = diff_fixed_like(fluc_lm_determ,fluc_lm_GS,mf_lm_new,mf_lm_old,Cl_new,Cl_old,*arg)
    return likes, Cl_new, fluc_lm_GS_next, mf_lm_new



    
