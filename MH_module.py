import numpy as np
from matplotlib import pyplot as plt
import sys



def Equation_ap(x,A,b):
    """
    returns Ax-b
    Keyword Arguments:
    A -- Matrix
    x -- guess vector
    b -- vector
    """
    return np.dot(A,x) - b

def quadratic_form(x,A,b):
    """
    Keyword Arguments:
    A -- 
    x -- 
    b -- 
    """
    return 1/2. * np.dot(np.dot(x.T,A),x) - np.dot(b.T,x) 


def grid_test(minmax,dim,func,*args):
    grid = np.zeros((dim,dim))
    for i,xi in enumerate(np.arange(minmax[0],minmax[1],(minmax[1]-minmax[0])/float(dim))):
        for j,xj in enumerate(np.arange(minmax[0],minmax[1],(minmax[1]-minmax[0])/float(dim))):
            grid[i,j] = func(np.array([xi,xj]),*args)
    return grid
    




def simple_2D_Gauss(x,*arg):
    """
    Keyword Arguments:
    x -- vector (np.array)
    *arg are:
    x_mean -- the mean vector (np.array)
    Cov -- covariance Matrix (np.matrix)
    
    faster than scipy.stats.multivariate_normal.pdf(x,mean,cov) for some reason
    """
    
#    return float(np.exp(-0.5 * np.dot(np.dot((x-x_mean).T,Cov.I),x-x_mean)))
    return float(np.exp(-0.5 * np.dot(np.dot((x-arg[0]).T,arg[1].I),x-arg[0])))





def autocorr(x):
    '''
    return the autocorrelation of a given array (much faster than computing the actual function)
    '''
    aa = np.correlate(x-x.mean(), x-x.mean(), mode='full')
    maxcorr = np.argmax(aa)
    result = aa / aa[maxcorr]
    return result[np.argmax(result):]




def MCMC_log(guess,functional_form,proposal,proposal_fun,niter,priors_func,firstiter,seed, *arg):
    """
    Same as previous, but for log likelihood

    Keyword Arguments:
    guess -- initial guess vector (np.array)
    functional_form -- form you are trying to sample
    proposal -- proposal random generator for next step
    proposal_fun -- proposal function to calculate ratios (next version should have both proposal and proposal_fun in one go)
    niter -- number of iterations
    *arg -- arguments that could be used by the functional form, and the proposal: 
    *arg[0] for the functional, 
    *arg[1] for the proposal. 
    *arg[2] for the priors
    It can be for example *arg = [A,x],[],[0,1/0.5**2]
    """
    Pid = np.random.randint(0,100000)
    #print "Pid = %d"%Pid
    #print "priors= ",arg[2]
    failed = []
    f_old,Cl = functional_form(guess,*arg[0])
    f_old += sum(priors_func(guess,*arg[2]))
    print f_old
    guesses = []
    guesses.append(guess)
    flag = []
    flag.append(-3)
    Cls=[]
    Cls.append(Cl)
    like=[]
    like.append(f_old)
    failed = 0
    state_rand=[]
    if firstiter==0:
        pass
    else:
        np.random.set_state(seed)
        print seed[0]
    for i in xrange(firstiter+1, niter+1):
        #print i
        try:#if 0==0:  
            #if i%500==0:
            #    print i, np.random.random()   
            #print i
            guess_new = guess + proposal(*arg[1])
            guesses.append(guess_new)
            if np.min(guess_new)<0:
                #print "Negative param ! ",guess_new
                flag.append(-1)
                like.append(-1)
                Cls.append(0)
            else:
                f_new,Cl = functional_form(guess_new,*arg[0])
                like.append(f_new)
                Cls.append(Cl)
                f_new += sum(priors_func(guess_new,*arg[2]))
                A = min(0,f_new-f_old+proposal_fun(guess,guess_new,*arg[1])-proposal_fun(guess_new,guess,*arg[1]))
                #print A,"f_new = ",f_new,"f_old = ",f_old, "guess_new = ", guess_new, "guess_old = ",guess
                if A==0:
                    guess=guess_new
                    flag.append(1)
                    f_old = f_new
                elif A<0:
                    u = np.log(np.random.rand(1))
                    #print "u = ",u
                    if u <= A:
                        guess=guess_new
                        flag.append(2)
                        f_old = f_new
                        #print "Lucky choice ! f_old = ",f_old
                    else:
                        flag.append(0)
                        pass
            if i%10000==0:
                flag_temp = np.array(flag)
                print i, " %.2f rejected; %.2f accepted; %.2f Lucky accepted; %d negative: try removed"%(len(np.where(flag_temp==0)[0])/float(i),len(np.where(flag_temp==1)[0])/float(i),len(np.where(flag_temp==2)[0])/float(i),len(np.where(flag_temp==-1)[0]))
                #print guess,f_old
                if i%10000==0:
                    state_rand.append(np.random.get_state())
                    #print np.random.random()
                    np.save("outputs/tempo_MC_chain_%d.npy"%Pid,[guesses,np.array(flag),np.array(like),Cls,state_rand])
                    print "temporary file saved: %d"%Pid
        except:
            failed+=1
            #print "error: %s on line %s"%(sys.exc_info()[0],sys.exc_info()[-1].tb_lineno)
            flag.append(-2)
            like.append(-2)
            Cls.append(np.zeros(len(Cl)))
            #plt.draw()
    print "%d fails"%failed
    return guesses,np.array(flag),np.array(like),Cls,state_rand







def MCMC_log_Jeff_new(guess,functional_form,proposal,proposal_fun,niter,priors_func,firstiter,seed,*arg):
    """
    Same as previous, but for log likelihood

    Keyword Arguments:
    guess -- initial guess vector (np.array)
    functional_form -- form you are trying to sample
    proposal -- proposal random generator for next step
    proposal_fun -- proposal function to calculate ratios (next version should have both proposal and proposal_fun in one go)
    niter -- number of iterations
    *arg -- arguments that could be used by the functional form, and the proposal: 
    *arg[0] for the functional, arg[0][0] = [dlm,strings,dd,nl[:lmax+1],bl[:lmax+1]],arg[0][1] = [fluc,cl]
    *arg[1] for the proposal. 
    *arg[2] for the priors
    It can be for example *arg = [A,x],[],[0,1/0.5**2]
    """
    Pid = np.random.randint(0,1e6)
    #print "Pid = %d"%Pid
    #print "priors= ",arg[2]
    failed = []
    guesses = []
    #guesses.append(guess)
    priors_old = sum(priors_func(guess,*arg[2]))
    flag = []
    #flag.append(-3)
    Cls=[]
    #Cls.append(Cl_old)
    like=[]
    #like.append(f_old)
    failed = 0
    state_rand=[]
    if firstiter==0:
        pass
    else:
        np.random.set_state(seed)
    for i in xrange(firstiter,niter):
        #print i
        try:#if 0==0:
            #print i
            guess_new = guess + proposal(*arg[1])
            guesses.append(guess_new)  
            if np.min(guess_new)<0:
                #print "Negative param ! ",guess_new
                flag.append(-1)
                like.append(-1)
                Cls.append(np.zeros(len(Cl_new)))
            else:
                likes, Cl_new, fluc_lm_GS_next, mf_lm_new = functional_form(guess_new,*arg[0])
                Cls.append(Cl_new)
                priors_new = sum(priors_func(guess_new,*arg[2]))
                like.append(likes[0] - likes[1] + priors_new - priors_old)#likes[0])
                A = min(0,likes[0] - likes[1] + priors_new - priors_old)#+proposal_fun(guess,guess_new,*arg[1])-proposal_fun(guess_new,guess,*arg[1]))
                #print A,"new = ",likes[0],"old = ",likes[1], "guess_new = ", guess_new, "guess_old = ",guess
                if A==0:
                    guess=guess_new
                    flag.append(1)
                    arg[0][1] =  [Cl_new, fluc_lm_GS_next,mf_lm_new]
                    priors_old = priors_new
                elif A<0:
                    u = np.log(np.random.rand(1))
                    #print "u = ",u
                    if u <= A:
                        guess=guess_new
                        flag.append(2)
                        arg[0][1] =  [Cl_new, fluc_lm_GS_next,mf_lm_new]
                        priors_old = priors_new                        
                    else:
                        flag.append(0)
                        pass
            if i%10000==0:
                state_rand.append(np.random.get_state())                    
                flag_temp = np.array(flag)
                #print "%.2f rejected; %.2f accepted; %.2f Lucky accepted; %d negative: try removed"%(len(np.where(flag_temp==0)[0])/float(i),len(np.where(flag_temp==1)[0])/float(i),len(np.where(flag_temp==2)[0])/float(i),len(np.where(flag_temp==-1)[0]))
                if i%10000==0:
                    np.save("outputs/tempo_MC_chain_%d.npy"%Pid,[guesses,np.array(flag),np.array(like),Cls,state_rand])
                    print i, " temporary file saved: %d"%Pid
        except:
            failed+=1
            print i," error: %s on line %s"%(sys.exc_info()[0],sys.exc_info()[-1].tb_lineno)
            flag.append(-2)
            like.append(-2)
            Cls.append(np.zeros(len(Cl_new)))
            #plt.draw()
    print "%d fails"%failed
    return guesses,np.array(flag),np.array(like),Cls,state_rand










def MCMC_log_Jeff_new_1D(guess,functional_form,proposal,proposal_fun,niter,priors_func,firstiter,seed,*arg):
    """
    Same as previous, but for log likelihood

    Keyword Arguments:
    guess -- initial guess vector (np.array)
    functional_form -- form you are trying to sample
    proposal -- proposal random generator for next step
    proposal_fun -- proposal function to calculate ratios (next version should have both proposal and proposal_fun in one go)
    niter -- number of iterations
    *arg -- arguments that could be used by the functional form, and the proposal: 
    *arg[0] for the functional, arg[0][0] = [dlm,strings,dd,nl[:lmax+1],bl[:lmax+1]],arg[0][1] = [fluc,cl]
    *arg[1] for the proposal. 
    *arg[2] for the priors
    It can be for example *arg = [A,x],[],[0,1/0.5**2]
    """
    Pid = np.random.randint(0,1e6)
    #print "Pid = %d"%Pid
    #print "priors= ",arg[2]
    failed = []
    guesses = []
    #guesses.append(guess)
    priors_old = sum(priors_func(guess,*arg[2]))
    flag = []
    #flag.append(-3)
    Cls=[]
    #Cls.append(Cl_old)
    like=[]
    #like.append(f_old)
    failed = 0
    state_rand=[]
    grid_1D = np.linspace(3.0,3.6,100)
    if firstiter==0:
        pass
    else:
        np.random.set_state(seed)
    for i in xrange(firstiter,niter):
        #print i
        if 0==0:
            guess_new = [grid_1D[i]]
            guesses.append(guess_new)  
            if np.min(guess_new)<0:
                #print "Negative param ! ",guess_new
                flag.append(-1)
                like.append(-1)
                Cls.append(np.zeros(len(Cl_new)))
            else:
                likes, Cl_new, fluc_lm_GS_next, mf_lm_new = functional_form(guess_new,*arg[0])
                Cls.append(Cl_new)
                priors_new = sum(priors_func(guess_new,*arg[2]))
                like.append(likes[0] - likes[1] + priors_new - priors_old)#likes[0])
                A = min(0,likes[0] - likes[1] + priors_new - priors_old)#+proposal_fun(guess,guess_new,*arg[1])-proposal_fun(guess_new,guess,*arg[1]))
                #print A,"new = ",likes[0],"old = ",likes[1], "guess_new = ", guess_new, "guess_old = ",guess
                if A==0:
                    guess=guess_new
                    flag.append(1)
                    arg[0][1] =  [Cl_new, fluc_lm_GS_next,mf_lm_new]
                    priors_old = priors_new
                elif A<0:
                    u = np.log(np.random.rand(1))
                    #print "u = ",u
                    if u <= A:
                        guess=guess_new
                        flag.append(2)
                        arg[0][1] =  [Cl_new, fluc_lm_GS_next,mf_lm_new]
                        priors_old = priors_new                        
                    else:
                        flag.append(0)
                        guess=guess_new
                        #flag.append(2)
                        arg[0][1] =  [Cl_new, fluc_lm_GS_next,mf_lm_new]
                        priors_old = priors_new
                        pass
            if i%10000==0:
                state_rand.append(np.random.get_state())                    
                flag_temp = np.array(flag)
                #print "%.2f rejected; %.2f accepted; %.2f Lucky accepted; %d negative: try removed"%(len(np.where(flag_temp==0)[0])/float(i),len(np.where(flag_temp==1)[0])/float(i),len(np.where(flag_temp==2)[0])/float(i),len(np.where(flag_temp==-1)[0]))
                if i%10000==0:
                    np.save("outputs/tempo_MC_chain_1D_%d.npy"%Pid,[guesses,np.array(flag),np.array(like),Cls,state_rand])
                    print i, " temporary file saved: %d"%Pid
        #except:
        #    failed+=1
        #    print i," error: %s on line %s"%(sys.exc_info()[0],sys.exc_info()[-1].tb_lineno)
        #    flag.append(-2)
        #    like.append(-2)
        #    Cls.append(np.zeros(len(Cl_new)))
            #plt.draw()
    #print "%d fails"%failed
    return guesses,np.array(flag),np.array(like),Cls,state_rand
