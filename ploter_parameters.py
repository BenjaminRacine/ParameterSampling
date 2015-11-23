from matplotlib import pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter
from matplotlib.patches import Ellipse  
import MH_module as MH
from matplotlib.legend_handler import HandlerPatch
import tabulate as T
import scipy.stats
from scipy.stats import kde

list_rules = list(T.LATEX_ESCAPE_RULES)
for i in list_rules :
    del(T.LATEX_ESCAPE_RULES[u'%s'%i])




def determine_FD_binning(array_in):
    '''
    returns the optimal binning, according to Freedman-Diaconis rule: see http://en.wikipedia.org/wiki/Freedman%E2%80%93Diaconis_rule                           
    '''
    sorted_arr = np.sort(array_in)
    Q3 = scipy.stats.scoreatpercentile(sorted_arr,3/4.)
    Q1 = scipy.stats.scoreatpercentile(sorted_arr,1/4.)
    IQR = Q3-Q1
    bin_size = 2.*IQR*np.size(array_in)**(-1/3.)
    return min(np.round(np.sqrt(np.size(array_in))),np.round((sorted_arr[-1] - sorted_arr[0])/bin_size))

class HandlerEllipse(HandlerPatch):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        center = 0.5 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent
        p = mpatches.Ellipse(xy=center, width=width + xdescent,
                             height=height + ydescent)
        self.update_prop(p, orig_handle, legend)
        p.set_transform(trans)
        return [p]

def make_legend_ellipse(legend, orig_handle,
                        xdescent, ydescent,
                        width, height, fontsize):
    """
    stolen from http://matplotlib.org/examples/pylab_examples/legend_demo_custom_handler.html
    """
    p = mpatches.Ellipse(xy=(0.5*width-0.5*xdescent, 0.5*height-0.5*ydescent),
                         width = width+xdescent, height=(height+ydescent))

    return p




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
        ax_temp.locator_params(tight=True,nbins=4)
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
    width, height = 2 * nsig * np.sqrt(vals) * np.sqrt(2.278)#5.991)
    #print vals
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

def compute_sigma_level(trace1, trace2, nbins=20):
    """From a set of traces, bin by number of standard deviations, stolen from http://jakevdp.github.io/"""
    L, xbins, ybins = np.histogram2d(trace1, trace2, nbins)
    L[L == 0] = 1E-16
    logL = np.log(L)

    shape = L.shape
    L = L.ravel()

    # obtain the indices to sort and unsort the flattened array
    i_sort = np.argsort(L)[::-1]
    i_unsort = np.argsort(i_sort)

    L_cumsum = L[i_sort].cumsum()
    L_cumsum /= L_cumsum[-1]
    
    xbins = 0.5 * (xbins[1:] + xbins[:-1])
    ybins = 0.5 * (ybins[1:] + ybins[:-1])

    return xbins, ybins, L_cumsum[i_unsort].reshape(shape)


def Triangle_plot_Cov_dat(chains,x_mean,Cov,titles,which_par,save,title_plot,**kwargs):
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
    nb_param = chains.shape[1]
    # definitions for the axes (left width and left_h is also bottom height and bottom_h)
    left = 0.06#/(nb_param)
    bottom = 0.1/(nb_param)
    width = 0.9/(nb_param) 
    axHistx=[]
    axScatter=[]
    x_mean = x_mean[which_par]
    Cov = Cov[which_par,:][:,which_par]
    plt.figure(figsize=(12,9))
    for i in range(nb_param):
        rect_histx = [left+i*width, bottom+(nb_param-1-i)*width, width, width]
        ax_temp = plt.axes(rect_histx)
        axHistx.append(ax_temp)
        ax_temp.xaxis.tick_top()
        ax_temp.yaxis.set_visible(False)#tick_right()
        ax_temp.locator_params(tight=True,nbins=4)
        x1 = np.linspace(x_mean[i]- 5*np.sqrt(Cov[i,i]),x_mean[i] + 5*np.sqrt(Cov[i,i]),200)
        #sUtahtop
        ax_temp.plot(x1,np.exp(-0.5*(x1-x_mean[i])**2/Cov[i,i])/np.sqrt(2*np.pi*Cov[i,i]),"b--")
        ax_temp.plot(x1,np.exp(-0.5*(x1-chains[:,i].mean())**2/chains[:,i].std()**2)/np.sqrt(2*np.pi*chains[:,i].std()**2),"r")
        ax_temp.hist(chains[:,i],np.round(np.sqrt(chains.shape[0])),histtype="step",normed=True,color='g')
        ax_temp.set_title(titles[which_par[i]],y=1.25)
        ax_temp.set_xlim(np.min((chains[:,i].mean()-5*chains[:,i].std(),x1.min())),np.max((chains[:,i].mean()+5*chains[:,i].std(),x1.max())))
        #ax_temp.set_xlim(x1.min(),x1.max())
        for j in range(i+1,nb_param):
            covar = np.cov([chains[:,i],chains[:,j]])
            means = [chains[:,i].mean(),chains[:,j].mean()]
            rect_scatter = [left+(i)*width, bottom+(nb_param-j-1)*width, width, width]
            y1 = np.linspace(x_mean[j]- 5*np.sqrt(Cov[j,j]),x_mean[j] + 5*np.sqrt(Cov[j,j]),200)
            ax_temp=plt.axes(rect_scatter)
            ell = plot_ellipse(Cov[[i,j],:][:,[i,j]],x_mean[[i,j]],1,plot=1,axe=ax_temp,fill=False,color="b")
            ell2 = plot_ellipse(covar,means,1,plot=1,axe=ax_temp,fill=False,color="r")
            xbins, ybins, sigma = compute_sigma_level(chains[:,i], chains[:,j],nbins = 30)
            ax_temp.contour(xbins, ybins, sigma.T, levels=[0.683, 0.955])
            #ax_temp.scatter(guesses[:,i][flag==0],guesses[:,j][flag==0],color="g",alpha=0.05)
            #scat = ax_temp.scatter(guesses[:,i][flag==1],guesses[:,j][flag==1],color="g",alpha=0.05)
            #ax_temp.scatter(guesses[:,i][flag==2],guesses[:,j][flag==2],color="g",alpha=0.05)
            scat = ax_temp.scatter(chains[:,i],chains[:,j],color="g",alpha=0.05)
#            ax_temp.add_artist(ell)
            #ax_temp.plot(np.arange(12))
            if i==0:
                ax_temp.yaxis.set_ticks_position('left')
            else:
                ax_temp.yaxis.set_visible(False)
            ax_temp.xaxis.set_major_formatter(nullfmt)
            ax_temp.xaxis.set_visible(False)
            ax_temp.locator_params(tight=True,nbins=4)
            axScatter.append(ax_temp)
            ax_temp.set_xlim(np.min((means[0]-5*np.sqrt(covar[0,0]),x1.min())),np.max((means[0]+5*np.sqrt(covar[0,0]),x1.max())))
            ax_temp.set_ylim(np.min((means[1]-5*np.sqrt(covar[1,1]),y1.min())),np.max((means[1]+5*np.sqrt(covar[1,1]),y1.max())))
            #ax_temp.set_ylim(y1.min(),y1.max())
            #pass
    plt.suptitle(title_plot,size=16)
    plt.figtext(0.65,0.75,"number of samples: %d"%len(chains[:,0]),size=16)
    plt.figtext(0.65,0.70,"aceptance rate = %.2f"%np.divide(len(np.unique(chains[:,0])),len(chains[:,0]),dtype=float),size =16)
    fig = plt.gcf()
    fig.legend([ell, scat, ell2], ['Proposal Covariance', 'MCMC steps','Posterior Covariance'], "upper right")
    if save!=0:
        plt.savefig("plots/Triangle_%s"%save)
    return axScatter, axHistx


def Triangle_plot_Cov_density(chains,x_mean,Cov,titles,which_par,save,title_plot,**kwargs):
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
    nb_param = chains.shape[1]
    # definitions for the axes (left width and left_h is also bottom height and bottom_h)
    left = 0.06#/(nb_param)
    bottom = 0.1/(nb_param)
    width = 0.9/(nb_param) 
    axHistx=[]
    axScatter=[]
    x_mean = x_mean[which_par]
    Cov = Cov[which_par,:][:,which_par]
    plt.figure(figsize=(12,9))
    for i in range(nb_param):
        rect_histx = [left+i*width, bottom+(nb_param-1-i)*width, width, width]
        ax_temp = plt.axes(rect_histx)
        axHistx.append(ax_temp)
        ax_temp.xaxis.tick_top()
        ax_temp.yaxis.set_visible(False)#tick_right()
        ax_temp.locator_params(tight=True,nbins=4)
        x1 = np.linspace(x_mean[i]- 5*np.sqrt(Cov[i,i]),x_mean[i] + 5*np.sqrt(Cov[i,i]),200)
        #sUtahtop
        ax_temp.plot(x1,np.exp(-0.5*(x1-x_mean[i])**2/Cov[i,i])/np.sqrt(2*np.pi*Cov[i,i]),"b--")
        ax_temp.plot(x1,np.exp(-0.5*(x1-chains[:,i].mean())**2/chains[:,i].std()**2)/np.sqrt(2*np.pi*chains[:,i].std()**2),"r")
        ax_temp.hist(chains[:,i],np.round(np.sqrt(chains.shape[0])),histtype="step",normed=True,color='g')
        ax_temp.set_title(titles[which_par[i]],y=1.25)
        ax_temp.set_xlim(np.min((chains[:,i].mean()-5*chains[:,i].std(),x1.min())),np.max((chains[:,i].mean()+5*chains[:,i].std(),x1.max())))
        print i
        #ax_temp.set_xlim(x1.min(),x1.max())
        for j in range(i+1,nb_param):
            covar = np.cov([chains[:,i],chains[:,j]])
            means = [chains[:,i].mean(),chains[:,j].mean()]
            k = kde.gaussian_kde(chains[:,[i,j]].T)
            nbins = 30
            x, y = chains[:,[i,j]].T
            xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
            zi = k(np.vstack([xi.flatten(), yi.flatten()]))
            rect_scatter = [left+(i)*width, bottom+(nb_param-j-1)*width, width, width]
            y1 = np.linspace(x_mean[j]- 5*np.sqrt(Cov[j,j]),x_mean[j] + 5*np.sqrt(Cov[j,j]),200)
            ax_temp=plt.axes(rect_scatter)
            ax_temp.pcolormesh(xi, yi, zi.reshape(xi.shape),cmap = "gnuplot2_r")
            ell = plot_ellipse(Cov[[i,j],:][:,[i,j]],x_mean[[i,j]],1,plot=1,axe=ax_temp,fill=False,color="b",ls="dashed")
            ell2 = plot_ellipse(covar,means,1,plot=1,axe=ax_temp,fill=False,color="r",ls = "dashed")
            xbins, ybins, sigma = compute_sigma_level(chains[:,i], chains[:,j],nbins = 30)
            ax_temp.contour(xbins, ybins, sigma.T, levels=[0.683, 0.955],colors=["c","k"])
            #ax_temp.scatter(guesses[:,i][flag==0],guesses[:,j][flag==0],color="g",alpha=0.05)
            #scat = ax_temp.scatter(guesses[:,i][flag==1],guesses[:,j][flag==1],color="g",alpha=0.05)
            #ax_temp.scatter(guesses[:,i][flag==2],guesses[:,j][flag==2],color="g",alpha=0.05)
            #scat = ax_temp.scatter(chains[:,i],chains[:,j],color="g",alpha=0.05)
#            ax_temp.add_artist(ell)
            #ax_temp.plot(np.arange(12))
            if i==0:
                ax_temp.yaxis.set_ticks_position('left')
            else:
                ax_temp.yaxis.set_visible(False)
            ax_temp.xaxis.set_major_formatter(nullfmt)
            ax_temp.xaxis.set_visible(False)
            ax_temp.locator_params(tight=True,nbins=4)
            axScatter.append(ax_temp)
            ax_temp.set_xlim(np.min((means[0]-5*np.sqrt(covar[0,0]),x1.min())),np.max((means[0]+5*np.sqrt(covar[0,0]),x1.max())))
            ax_temp.set_ylim(np.min((means[1]-5*np.sqrt(covar[1,1]),y1.min())),np.max((means[1]+5*np.sqrt(covar[1,1]),y1.max())))
            #ax_temp.set_ylim(y1.min(),y1.max())
            #pass
    plt.suptitle(title_plot,size=16)
    plt.figtext(0.65,0.75,"number of samples: %d"%len(chains[:,0]),size=16)
    plt.figtext(0.65,0.70,"aceptance rate = %.2f"%np.divide(len(np.unique(chains[:,0])),len(chains[:,0]),dtype=float),size =16)
    fig = plt.gcf()
    fig.legend([ell, ell2], ['Proposal Covariance','Posterior Covariance'], "upper right")
    if save!=0:
        plt.savefig("plots/Triangle_density_%s"%save)
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
    
def plot_like(guesses,like,flag,titles,which_par,save=0,title_plot=""):
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



def plot_detailed_chains(guesses,flag,chains,titles,which_par,x_mean,Cov,save=0,title_plot=""):
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
        plt.plot(np.arange(np.where(flag>0)[0][0],niter),chains[:,j],'b.',alpha = 0.5,label='MC chain')
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
        plt.title(title_plot)
        if save!=0:
            plt.savefig("plots/chain_%s_%s_%d.png"%(save,str(which_par).replace(',','').replace('[','').replace(']','').replace(' ',''),j))#,SafeID))


def plot_chains(outputs,titles,which_par,x_mean,Cov,save=0):
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
    guesses,flag,like,Cls = create_cleaned_outputs(outputs[:4])
    niter = len(flag)
    j=0
    for i in which_par:
        plt.figure()
        plt.plot(np.arange(np.where(flag>0)[0][0],niter),chains[:,j],'b.',alpha = 0.5,label='MC chain')
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




def plot_autocorr(chains,titles,which_par,burnin_cut,max_plot,save=0,title_plot=""):
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
    plt.figure(figsize=(12,9))
    for i in which_par:
        print i
        plt.plot(MH.autocorr(chains[:,j][burnin_cut:])[:max_plot],label=titles[i])
        j+=1
    plt.legend(loc="best")
    plt.ylabel("autocorrelation")
    plt.xlabel("Lag")
    plt.hlines(0,0,max_plot,linestyle = '--',alpha=0.5)
    plt.title(title_plot)
    plt.ylim(-0.2,1)
    if save!=0:
        plt.savefig("plots/Autocorrelation_%s_%s.png"%(save,str(which_par).replace(',','').replace('[','').replace(']','').replace(' ','')))#,SafeID))

            
def plot_all(MCMC_output,titles,which_par,x_mean,Cov,burnin_cut=50,save=0,plot_title="",plot_int = 0):
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
    guesses,flag,like,Cls = create_real_chain(MCMC_output)
    chains = real_chain(guesses,flag)
    plot_autocorr(chains,titles,which_par,burnin_cut,1000,save,plot_title)
    plot_detailed_chains(guesses,flag,chains,titles,which_par,x_mean,Cov,save,plot_title)
    Triangle_plot_Cov_dat(chains,x_mean,Cov,titles,which_par,save,plot_title)
    plot_like(guesses,like,flag,titles,which_par,save)
    plot_like_profile(guesses,like,flag,titles,which_par,save)
    
def real_chain(guesses,flag):
    """
    fills the guesses where rejected with previous accepted value
    """
    guesse_new = guesses.copy()
    accep_idx = np.where(flag>0)[0]
    for i in range(len(accep_idx)-1):
        for j in range(accep_idx[i],accep_idx[i+1]):
            guesse_new[j] = guesse_new[accep_idx[i]]
    for k in range(accep_idx[-1],len(guesses)):
        guesse_new[k] = guesse_new[accep_idx[-1]]
    #print accep_idx[0]
    #print guesse_new[accep_idx[0]:].shape[0]
    return guesse_new[accep_idx[0]:]

def create_real_chain(MCMC_output):
    """
    Return a clean chain, ie: the guesses where rejected with previous accepted value, from the output of the code
    """
    guesses,flag,like,Cls = create_cleaned_outputs(MCMC_output)
    chains = real_chain(guesses,flag)
    return chains

def create_cleaned_outputs(MCMC_output):
    """
    Generate arrays of guesses, Cls, flags, A values, cleaned from failed samples 
    """
    guesses,flag,like,Cls = MCMC_output[:4]
    N_par = len(guesses[0])
    guesses = np.concatenate(guesses)
    guesses = guesses.reshape(len(guesses)/N_par,N_par)
    Cls=np.array(Cls)[flag>-1]
    like=like[flag>-1]
    guesses = guesses[flag>-1]
    flag=flag[flag>-1]
    return guesses,flag,like,Cls



def compare_chains(chain1,chain2,save = 0,burnin_cut = [200,200],titles = np.array(["$\Omega_b h^2$","$\Omega_c h^2$",r"$\tau$","$A_s$","$n_s$","$H_0$"]),lab = ["chain 1","chain 2"],format_table = "latex"):
    """
    Return different statistics to compare chains from different outputs

    Keyword Arguments:
    chains1, chains2 -- the 2 outputs of the MCMC to be compared
    titles    -- usual names of parameters
    burnin    -- burnin to be applied to the differente chains
    """
    chain_1 = create_real_chain(chain1)[burnin_cut[0]:,:]
    chain_2 = create_real_chain(chain2)[burnin_cut[1]:,:]
    N_par = np.shape(chain_1)[1]
    bin_1 = 200#np.round(np.sqrt(np.shape(chain_1)[0]))
    bin_2 = 200#np.round(np.sqrt(np.shape(chain_2)[0]))
    plt.figure(figsize = (12,9))
    for i in range(N_par):
        plt.subplot(3,2,i+1)
        his_1 = plt.hist(chain_1[:,i],bin_1,histtype='step',normed=True,label = lab[0],alpha=0.5,color="g")
        his_2 = plt.hist(chain_2[:,i],bin_2,histtype='step',normed=True,label = lab[1],alpha=0.5,color="b")
        plt.title(titles[i])
        plt.tight_layout()
        if save!=0:
            plt.savefig("plots/Marginal_1D_lin_%s.png"%(save))
    plt.figure(figsize = (12,9))
    for i in range(N_par):
        plt.subplot(3,2,i+1)
        his_1 = plt.hist(chain_1[:,i],bin_1,histtype='step',normed=True,label = lab[0],alpha=0.5,log=True,color='g')
        his_2 = plt.hist(chain_2[:,i],bin_2,histtype='step',normed=True,label = lab[1],alpha=0.5,log=True,color='b')
        plt.title(titles[i])
        plt.tight_layout()
        if save!=0:
            plt.savefig("plots/Marginal_1D_log_%s.png"%(save))
    plt.figure(figsize = (12,9))
    for i in range(N_par):
        plt.subplot(3,2,i+1)
        plt.plot(chain_1[:,i],label = lab[0],alpha=0.5,color="g")
        plt.plot(chain_2[:,i],label = lab[1],alpha=0.5,color="b")
        plt.title(titles[i])
        plt.tight_layout()
        plt.locator_params(nbins=4)
        plt.xlim(0,100000)
        if save!=0:
            plt.savefig("plots/Trace_plot_%s.png"%(save))
    plt.figure(figsize = (12,9))
    for i in range(N_par):
        plt.subplot(3,2,i+1)
        plt.plot(chain_1[:1000,i],label = lab[0],alpha=0.5,color="g")
        plt.plot(chain_2[:1000,i],label = lab[1],alpha=0.5,color="b")
        plt.title(titles[i])
        plt.tight_layout()
        if save!=0:
            plt.savefig("plots/Trace_plot_zoom_%s.png"%(save))
    plt.figure()
    handle_1, = plt.plot(chain_1[:,i],label = lab[0],alpha=0.5,color="g")
    handle_2, = plt.plot(chain_2[:,i],label = lab[1],alpha=0.5,color="b")
    figlegend = plt.figure(figsize=(3,2))
    figlegend.legend([handle_1,handle_2],[lab[0],lab[1]],"center")
    figlegend.show()
    if save!=0:
        figlegend.savefig('plots/legend_%s.png'%save)
    header = [" "]
    mean_1 = ["mean %s"%lab[0]]
    mean_2 = ["mean %s"%lab[1]]
    std_1 = ["std %s"%lab[0]]
    std_2 = ["std %s"%lab[1]]
    skewness_1 = ["skew : %s"%lab[0]]
    skewness_2 = ["skew : %s"%lab[1]]
    kurt_1 = ["kurt : %s"%lab[0]]
    kurt_2 = ["kurt : %s"%lab[1]]
    ratios = ["std : %s/%s"%(lab[0],lab[1])]
    for i in range(N_par):
        header.append(titles[i])
        mean_1.append(chain_1[:,i].mean())
        mean_2.append(chain_2[:,i].mean())
        std_1.append(chain_1[:,i].std())
        std_2.append(chain_2[:,i].std())
        skewness_1.append(scipy.stats.skew(chain_1[:,i]))
        skewness_2.append(scipy.stats.skew(chain_2[:,i]))
        kurt_1.append(scipy.stats.kurtosis(chain_1[:,i]))
        kurt_2.append(scipy.stats.kurtosis(chain_2[:,i]))
        ratios.append(chain_1[:,i].std()/chain_2[:,i].std())
        table = [mean_1,mean_2,std_1,std_2,ratios,skewness_1,skewness_2,kurt_1,kurt_2]
    print T.tabulate(table, header,tablefmt=format_table)






def merge(list_file, cut_burn):
    '''
    list_file : list of temporary or full files eg : ['outputs/test1.npy','test2.npy']
    cut_burn  : burn-in cut applied to each file
    '''
    G = []
    F = []
    L = []
    C = []
    for file_name in list_file : 
        file_ = np.load(file_name)
        G.append(np.array(file_[0])[cut_burn:,:])
        F.append(np.array(file_[1])[cut_burn:])
        L.append(np.array(file_[2])[cut_burn:])
        C.append(np.zeros(len(np.array(file_[2])))[cut_burn:])#np.array(file_[3])[cut_burn:])
    G = np.concatenate(G)
    F = np.concatenate(F)
    L= np.concatenate(L)
    C = np.concatenate(C)
    return [G,F,L,C]



def Gel_Rub(chain_list,n,burnin):
    chain_list = np.array(map(lambda arr:arr[burnin:n],chain_list))
    n= chain_list.shape[1]
    m= chain_list.shape[0]
    Vars = np.var(chain_list,axis=1)
    Means = np.mean(chain_list,axis=1)
    W = np.mean(Vars,axis=0)
    B = n*np.var(Means,axis=0)
    #W = np.mean(var(chain_list,axis=1))
    #theta_m_m = np.mean(np.mean(chain_list,axis=1))
    #B = n*np.var(cc_list.mean(axis=1)-theta_m_m)
    vaar = (1-1./n)*W+1./n*B
    return np.sqrt(vaar/W)

def plot_Gel_rub(file_list,titles,N_max,burnin,save=0,title_plot="",thining_fact=1):
    """
    Return the Gelman-Rubin statistics for a given list of similar chains, as a function of iteration

    Keyword Arguments:
    file_list -- list of files generated by the MCMC
    titles    -- usual names of parameters
    N_max     -- maximum number of iteration for the plot
    burnin    -- burnin to be applied to the differente chains
    """
    def prepare_chains(file_list):
        outputs = map(np.load,file_list)
        tt = map(lambda outs:create_real_chain(outs[:4])[::thining_fact],outputs)
        return tt
    print "Gelman Rubin computed from %d chains"%len(file_list)
    tt = prepare_chains(file_list)
    gel_dep = []
    plt.figure(figsize=(12,9))
    burnin = burnin/thining_fact
    N_max = N_max/thining_fact
    incr=100/thining_fact
    for i in range(burnin+incr,N_max,incr):
        gel_dep.append(Gel_Rub(tt,i,burnin))
    for i in range(6):
        plt.plot(np.arange(burnin+incr,N_max,incr),np.array(gel_dep)[:,i],label=titles[i])
    plt.xlabel("number of samples (including burnin)")
    plt.fill_betweenx(np.linspace(0.95,1.4),0,burnin,alpha=0.3)
    plt.plot([], [], color='blue',alpha=0.3,label="burned_in",linewidth=10)
    plt.ylabel("R (Gelman-Rubin)")
    plt.hlines(1.01,0,N_max,linestyle = '--',alpha=0.5)
    plt.legend(loc='best')
    plt.title(title_plot)
    plt.ylim(0.95,1.2)
    if save!=0:
        plt.savefig("plots/Gelman_Rubin_%s.png"%save)





def plot_abel(list_len,x_mean,cov_new,titles,which_par,save=0,title_plot="",N_max=15000,burnin=500,thining_fact=1):
    test_merge = merge(list_len,burnin)
    chains = create_real_chain(test_merge)[::thining_fact]
    Triangle_plot_Cov_density(chains,x_mean,cov_new,titles,which_par,save,title_plot)
    plot_autocorr(chains,titles,which_par,0,1000,save,title_plot)
    plot_Gel_rub(list_len,titles,N_max,burnin,save,title_plot,thining_fact)
    plot_parameter_stats(list_len,titles,burnin,save,title_plot,thining_fact)


def plot_parameter_stats(file_list,titles,burnin,save=0,title_plot="",thining_fact=1):
    """
    Return the parameter estimated from the subsamples, and compare to total chain.

    Keyword Arguments:
    file_list -- list of files generated by the MCMC
    titles    -- usual names of parameters
    burnin    -- burnin to be applied to the differente chains
    """
    test_merge = merge(file_list,burnin)
    chains = create_real_chain(test_merge)[::thining_fact]
    nullfmt   = NullFormatter()
    def prepare_chains(file_list):
        outputs = map(np.load,file_list)
        tt = map(lambda outs:create_real_chain(outs[:4])[::thining_fact],outputs)
        return tt
    N_list = len(file_list)
    print "parameter stats computed from %d chains"%len(file_list)
    tt = prepare_chains(file_list)
    N_par = np.shape(tt[0])[1]
    plt.figure(figsize=(12,9))
    for i in range(N_par):
        plt.subplot(3,2,i+1)
        for j in range(N_list):
            tt_arr = np.array(tt[j])
            plt.errorbar(j, tt_arr[:,i].mean(),yerr = tt_arr[:,i].std()/np.sqrt(len(tt_arr[:,i])),marker='.',mfc='red')
        plt.title(titles[i])
        plt.tight_layout()
        ax_temp=plt.gca()
        plt.plot(np.arange(N_list),chains[:,i].mean()*np.ones(N_list),color='b',label = "Total chain")
        plt.fill_between(np.arange(N_list),chains[:,i].mean()-chains[:,i].std()/np.sqrt(len(chains[:,i])),chains[:,i].mean()+chains[:,i].std()/np.sqrt(len(chains[:,i])),color='b',alpha=0.2)
        ax_temp.xaxis.set_major_formatter(nullfmt)
        ax_temp.xaxis.set_visible(False)
        ax_temp.locator_params(tight=True,nbins=4)
        plt.xlim(-1,N_list)
