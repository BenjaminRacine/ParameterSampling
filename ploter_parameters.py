from matplotlib import pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter
from matplotlib.patches import Ellipse  
import MH_module as MH


def Triangle_plot_Cov(Cov,x_mean,**kwargs):
    """
    """
    
    nullfmt   = NullFormatter()
    nb_param = Cov.shape[0]
    # definitions for the axes (left width and left_h is also bottom height and bottom_h)
    left = 0.1/(nb_param)
    width = 0.9/(nb_param) 
    axHistx=[]
    axScatter=[]
    for i in range(nb_param):
        rect_histx = [left+i*width, left+(nb_param-1-i)*width, width, width]
        ax_temp = plt.axes(rect_histx)
        axHistx.append(ax_temp)
        ax_temp.xaxis.tick_top()
        ax_temp.yaxis.set_visible(False)#tick_right()
        ax_temp.locator_params(nbins=4)
        x1 = np.linspace(x_mean[i]- 3*np.sqrt(Cov[i,i]),x_mean[i] + 3*np.sqrt(Cov[i,i]),200)
        #sUtahtop
        ax_temp.plot(x1,np.exp(-0.5*(x1-x_mean[i])**2/Cov[i,i])/np.sqrt(2*np.pi*Cov[i,i]),**kwargs)
        for j in range(i+1,nb_param):
            rect_scatter = [left+(i)*width, left+(nb_param-j-1)*width, width, width]
            #print 1
            ax_temp=plt.axes(rect_scatter)
            ell = plot_ellipse(Cov[[i,j],:][:,[i,j]],x_mean[[i,j]],1,plot=1,axe=ax_temp,fill=False)
#            ax_temp.add_artist(ell)
            #ax_temp.plot(np.arange(12))
            ax_temp.xaxis.set_major_formatter(nullfmt)
            ax_temp.xaxis.set_visible(False)
            ax_temp.yaxis.set_visible(False)
            axScatter.append(ax_temp)
            pass
    return axScatter, axHistx


def plot_ellipse(cov,x_mean,nsig,axe,plot=0,**kwargs):
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]
    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
    width, height = 2 * nsig * np.sqrt(vals)
    ell = Ellipse(xy=x_mean, width=width, height=height, angle=theta,**kwargs)
    #plt.figure()
    #ax = plt.gca()
    if plot == 1:
        axe.add_artist(ell)
        #print ell.get_extents()
        plt.xlim(ell.center[0]-3*np.sqrt(cov[0,0]),ell.center[0]+3*np.sqrt(cov[0,0]))
        plt.ylim(ell.center[1]-3*np.sqrt(cov[1,1]),ell.center[1]+3*np.sqrt(cov[1,1]))
    #stop
    return ell


def cor2cov(cov_diag,Correlation_matrix):
    """
    Computes the covariance matrix, given the correlation matrix and the diagonal elements of the covariance
    """
    dim = cov_diag.shape[0]
    cov_new = np.zeros((dim,dim))
    for i in range(dim):
        #print i
        for j in range(i,dim):
            #print j
            cov_new[i,j] = Correlation_matrix[i,j]/100.*np.sqrt(cov_diag[i]*cov_diag[j])
            cov_new[j,i] = cov_new[i,j]
    return cov_new




def Triangle_plot_Cov_dat(guesses,flag,x_mean,Cov,titles,which_par,**kwargs):
    """
    Returns the triangle plots for a given chain, and compare to a given prediction

    Keyword Arguments:
    guesses -- the full list of generated guesses from the MCMC
    flag -- a list giving 0 for rejected, 1 for accepted, 2 for accepted even if likelihood is lower, and -1 for forbidden values (most probably negative ones)
    x_mean -- mean value corresponding to input data
    Cov -- covariance matrix (probably extracted from paper of other chains)
    titles -- a list of titles for the plots
    which_par -- a list of indices, corresponding to the order defined above, exemple [0,2] means ombh2,tau if order is [ombh2,omch2,tau,As,ns,H0]
    **kwargs are a set of arguments for the ellipses (to be tested)
    """
    nullfmt   = NullFormatter()
    nb_param = guesses.shape[1]
    # definitions for the axes (left width and left_h is also bottom height and bottom_h)
    left = 0.1/(nb_param)
    width = 0.9/(nb_param) 
    axHistx=[]
    axScatter=[]
    x_mean = x_mean[which_par]
    Cov = Cov[which_par,:][:,which_par]
    for i in range(nb_param):
        rect_histx = [left+i*width, left+(nb_param-1-i)*width, width, width]
        ax_temp = plt.axes(rect_histx)
        axHistx.append(ax_temp)
        ax_temp.xaxis.tick_top()
        ax_temp.yaxis.set_visible(False)#tick_right()
        ax_temp.locator_params(nbins=4)
        x1 = np.linspace(x_mean[i]- 5*np.sqrt(Cov[i,i]),x_mean[i] + 5*np.sqrt(Cov[i,i]),200)
        #sUtahtop
        ax_temp.plot(x1,np.exp(-0.5*(x1-x_mean[i])**2/Cov[i,i])/np.sqrt(2*np.pi*Cov[i,i]),**kwargs)
        ax_temp.hist(guesses[:,i][flag>0],np.sqrt(sum(flag>0)),histtype="step",normed=True)
        ax_temp.set_title(titles[which_par[i]])
        ax_temp.set_xlim(x1.min(),x1.max())
        for j in range(i+1,nb_param):
            rect_scatter = [left+(i)*width, left+(nb_param-j-1)*width, width, width]
            ax_temp=plt.axes(rect_scatter)
            ell = plot_ellipse(Cov[[i,j],:][:,[i,j]],x_mean[[i,j]],1,plot=1,axe=ax_temp,fill=False)
            ax_temp.scatter(guesses[:,i][flag==0],guesses[:,j][flag==0],color="k",alpha=0.05)
            ax_temp.scatter(guesses[:,i][flag==1],guesses[:,j][flag==1],color="g",alpha=0.3)
            ax_temp.scatter(guesses[:,i][flag==2],guesses[:,j][flag==2],color="r",alpha=0.3)
#            ax_temp.add_artist(ell)
            #ax_temp.plot(np.arange(12))
            ax_temp.xaxis.set_major_formatter(nullfmt)
            ax_temp.xaxis.set_visible(False)
            ax_temp.yaxis.set_visible(False)
            axScatter.append(ax_temp)
            ax_temp.set_xlim(x1.min(),x1.max())
            pass
    return axScatter, axHistx


def plot_like_profile(guesses,like,flag,titles,which_par,save=0):
    """
    plots the 1D likelihood profiles, ie the log likelihood as a function of the parameters.

    Keyword Arguments:
    guesses -- the full list of generated guesses from the MCMC
    like -- list of likelihood values from the MCMC
    flag -- a list giving 0 for rejected, 1 for accepted, 2 for accepted even if likelihood is lower, and -1 for forbidden values (most probably negative ones)
    titles -- a list of titles for the plots
    which_par -- a list of indices, corresponding to the order defined above, exemple [0,2] means ombh2,tau if order is [ombh2,omch2,tau,As,ns,H0]
    """
    j=0
    for i in which_par:
        plt.figure()
        plt.plot(guesses[flag==1,j],like[flag==1],".g",label="Accepted")
        plt.plot(guesses[flag==2,j],like[flag==2],".r",label="Lucky accepted")
        j+=1 
        plt.title(titles[i])
        plt.ylabel("Log Likelihood")
        plt.xlabel(titles[i])
        plt.legend(loc="best")
        if save!=0:
            plt.savefig("plots/log_like_profile_%s_%s_%d.png"%(save,str(which_par).replace(',','').replace('[','').replace(']','').replace(' ',''),j))#,SafeID))

def plot_like(guesses,like,flag,titles,which_par,save=0):
    """
    Plots the likelihood values, ie the log likelihood as a function of iteration.

    Keyword Arguments:
    guesses -- the full list of generated guesses from the MCMC
    like -- list of likelihood values from the MCMC
    flag -- a list giving 0 for rejected, 1 for accepted, 2 for accepted even if likelihood is lower, and -1 for forbidden values (most probably negative ones)
    titles -- a list of titles for the plots
    which_par -- a list of indices, corresponding to the order defined above, exemple [0,2] means ombh2,tau if order is [ombh2,omch2,tau,As,ns,H0]
    """
    plt.figure()
    plt.plot(np.arange(len(guesses))[flag==1],like[flag==1],".g",label="Accepted")
    plt.plot(np.arange(len(guesses))[flag==2],like[flag==2],".r",label="Lucky accepted")
    plt.ylabel("Log Likelihood")
    plt.xlabel("Iteration")
    plt.legend(loc="best")
    if save!=0:
        plt.savefig("plots/log_like_%s_%s.png"%(save,str(which_par).replace(',','').replace('[','').replace(']','').replace(' ','')))#,SafeID))



def plot_chains(guesses,flag,titles,which_par,x_mean,Cov,save=0):
    """
    Plots the chains for all parameters, and the priors

    Keyword Arguments:
    guesses -- the full list of generated guesses from the MCMC
    flag -- a list giving 0 for rejected, 1 for accepted, 2 for accepted even if likelihood is lower, and -1 for forbidden values (most probably negative ones)
    titles -- a list of titles for the plots
    which_par -- a list of indices, corresponding to the order defined above, exemple [0,2] means ombh2,tau if order is [ombh2,omch2,tau,As,ns,H0]
    x_mean -- mean value corresponding to input data
    Cov -- covariance matrix (probably extracted from paper of other chains)
    """
    niter = len(flag)
    j=0
    for i in which_par:
        plt.figure()
        plt.plot(np.arange(niter)[flag==0],guesses[flag==0,j],'k.',alpha = 0.2,label='Rejected')
        plt.plot(np.arange(niter)[flag==1],guesses[flag==1,j],'g.',label="Accepted")
        plt.plot(np.arange(niter)[flag==2],guesses[flag==2,j],'r.',label='Lucky accepted')
        plt.title(titles[i]+"MC chains")
        plt.xlabel("Iterations")
        plt.ylabel(titles[i])
        plt.plot(np.arange(niter),x_mean[i]*np.ones(niter),color='b',label = "Planck prior")
        plt.fill_between(np.arange(niter),x_mean[i]-np.sqrt(Cov[i,i]),x_mean[i]+np.sqrt(Cov[i,i]),color='b',alpha=0.2)
        plt.legend(loc="best")
        tot = len(flag)
        print titles[i],": %.2f rejected; %.2f accepted; %.2f Lucky accepted"%((flag==0).mean(),(flag==1).mean(),(flag==2).mean())
        j+=1
        if save!=0:
            plt.savefig("plots/chain_%s_%s_%d.png"%(save,str(which_par).replace(',','').replace('[','').replace(']','').replace(' ',''),j))#,SafeID))




def plot_autocorr(guesses,flag,titles,which_par,burnin_cut,save=0):
    """
    Plots the chains for all parameters, and the priors

    Keyword Arguments:
    guesses -- the full list of generated guesses from the MCMC
    flag -- a list giving 0 for rejected, 1 for accepted, 2 for accepted even if likelihood is lower, and -1 for forbidden values (most probably negative ones)
    titles -- a list of titles for the plots
    which_par -- a list of indices, corresponding to the order defined above, exemple [0,2] means ombh2,tau if order is [ombh2,omch2,tau,As,ns,H0]
    burnin_cut -- cut the first few iterations for computation of the autocorr

    """
    j=0
    for i in which_par:
        print j
        plt.figure()
        plt.plot(MH.autocorr(guesses[flag>0,j][burnin_cut:]))
        plt.title("%s autocorrelation"%titles[i])
        plt.ylabel(titles[i])
        plt.xlabel("Lag")
        if save!=0:
            plt.savefig("plots/Autocorrelation_%s_%s_%d.png"%(save,str(which_par).replace(',','').replace('[','').replace(']','').replace(' ',''),j))#,SafeID))
        j+=1
            
def plot_all(chain,titles,which_par,x_mean,Cov,burnin_cut=50,save=0,plot_int = 0):
    """
    chain -- output of the MCMC, should contain guesses,flag,like,Cls
    titles -- a list of titles for the plots
    which_par -- a list of indices, corresponding to the order defined above, exemple [0,2] means ombh2,tau if order is [ombh2,omch2,tau,As,ns,H0]
    burnin_cut -- cut the first few iterations for computation of the autocorr
    x_mean -- mean value corresponding to input data
    Cov -- covariance matrix (probably extracted from paper of other chains)
    """
    if plot_int==0:
        plt.ioff()
    else:
        plt.ion()
    guesses,flag,like,Cls = chain
    guesses = np.concatenate(guesses)
    guesses = guesses.reshape(len(guesses)/len(which_par),len(which_par))
    #Cls = np.concatenate(Cls)
    #Cls = Cls.reshape(len(Cls)/2,2)
    Cls=Cls[flag!=-2]
    like=like[flag!=-2]
    flag=flag[flag!=-2]
    plot_autocorr(guesses,flag,titles,which_par,burnin_cut,save)
    plot_chains(guesses,flag,titles,which_par,x_mean,Cov,save)
    plt.figure()
    Triangle_plot_Cov_dat(guesses,flag,x_mean,Cov,titles,which_par)
    if save!=0:
        plt.savefig("plots/Triangle_%s.png"%save)
    plot_like(guesses,like,flag,titles,which_par,save)
    plot_like_profile(guesses,like,flag,titles,which_par,save)
    
