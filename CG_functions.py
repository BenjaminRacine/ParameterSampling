import numpy as np
import healpy as hp
import itertools
from matplotlib import pyplot as plt

global index_pos
#index_pos = np.array(list(itertools.chain.from_iterable([[hp.Alm.getidx(lmax, l, m) for m in range(1,l+1)] for l in range(lmax+1)])))

def gaussian_beam(lmax, fwhm):
   """
   Returns the spherical transform of a gaussian beam
   from l=0 to lmax with a fwhm in arcmin.
   """
   l_arr = np.arange(lmax+1)
   sigma = np.radians(fwhm/60.) / np.sqrt(8.*np.log(2.))
   beam = np.exp(-0.5*l_arr*(l_arr+1)*sigma*sigma)
   return beam



def filter_alm(alm,new_lmax):
        '''                                                                                                                             
        For a given alm array, returns this array, where all useless multipoles have been filtered out                                  
        '''
        alm_temp = np.zeros(hp.Alm.getsize(new_lmax))+0*1j#alm.size)+0*1j                                                                    
        lmax = hp.Alm.getlmax(alm.size)
        index = np.concatenate([hp.Alm.getidx(lmax,ell,np.arange(ell+1)) for ell in range(lmax+1)])
        index_temp = np.concatenate([hp.Alm.getidx(new_lmax,ell,np.arange(ell+1)) for ell in range(new_lmax+1)])
        alm_temp[index_temp] = alm[index]
        return alm_temp


def complex2real_alm(alm):
    """
    To be finished: compute the real a_lm
    """
    ##### The bottleneck here is to compute the double for loop for the indices. We define it as global here so that it's defined once and for all.
    global index_pos
    lmax = hp.Alm.getlmax(alm.size)
    alm_temp = np.zeros((lmax+1)**2)
    try: 
       index_pos
       #print "yes"
    except: 
       print lmax
       index_pos = np.array(list(itertools.chain.from_iterable([[hp.Alm.getidx(lmax, l, m) for m in range(1,l+1)] for l in range(lmax+1)])))
       #print "no"
    index_0 = [hp.Alm.getidx(lmax, l, 0) for l in range(lmax+1)]
    alm_temp[index_0] = np.real(alm[index_0])
    alm_temp[lmax+1+2*(index_pos-lmax-1)] = np.sqrt(2)*np.real(alm[index_pos])
    alm_temp[lmax+2+2*(index_pos-lmax-1)] = np.sqrt(2)*np.imag(alm[index_pos])
    return alm_temp


def real2complex_alm(alm):
    """
    To be finished: compute the real a_lm
    """
    lmax = int(np.sqrt(alm.size)-1)
    alm_temp = np.zeros(hp.Alm.getsize(lmax))+1j*0
    index_pos = np.array(list(itertools.chain.from_iterable([[hp.Alm.getidx(lmax, l, m) for m in range(1,l+1)] for l in range(lmax+1)])))
    index_0 = [hp.Alm.getidx(lmax, l, 0) for l in range(lmax+1)]
    alm_temp[index_0] = alm[index_0]+1j*0
    alm_temp[index_pos] = 1/np.sqrt(2)*(alm[lmax+1+2*(index_pos-lmax-1)]+1j*alm[lmax+2+2*(index_pos-lmax-1)])
    #alm_temp[lmax+1+2*(index_pos-lmax-1)] = np.sqrt(2)*np.real(alm[index_pos])
    #alm_temp[lmax+2+2*(index_pos-lmax-1)] = np.sqrt(2)*np.imag(alm[index_pos])
    return alm_temp

    

def A_matrix_func(data):
    """
    cf eq 25 of eriksen 2004
    """
    Chalf = np.sqrt(data.cl_th[:data.lmax+1])*data.beam[:data.lmax+1]
    map2 = hp.alm2map(hp.almxfl(real2complex_alm(data.alm),Chalf),data.nside)*data.invvar
    alm2 = hp.almxfl(hp.map2alm(map2,data.lmax,use_weights=False)*hp.nside2npix(data.nside)/4./np.pi,Chalf)
    return data.alm+complex2real_alm(alm2)

def rs_data_matrix_func(data,d):
    """
    cf eq 25 of eriksen 2004
    """
    Chalf = np.sqrt(data.cl_th[:data.lmax+1])*data.beam[:data.lmax+1]
    map2 = d*data.invvar
    #map2 = hp.alm2map(real2complex_alm(data.alm),data.nside)*data.invvar
    alm2 = hp.almxfl(hp.map2alm(map2,data.lmax,use_weights=False)*hp.nside2npix(data.nside)/4./np.pi,Chalf)
    return complex2real_alm(alm2)

def rs_w1_matrix_func(data,w1):
    """
    cf eq 25 of eriksen 2004
    """
    Chalf = np.sqrt(data.cl_th[:data.lmax+1])*data.beam[:data.lmax+1]
    map2 = w1*np.sqrt(data.invvar)
    #map2 = hp.alm2map(real2complex_alm(data.alm),data.nside)*np.sqrt(data.invvar)
    alm2 = hp.almxfl(hp.map2alm(map2,data.lmax,use_weights=False)*hp.nside2npix(data.nside)/4./np.pi,Chalf)
    return complex2real_alm(alm2)

def return_map(map_class):
    """
    We solve for C^{-1/2}x, here is to recover x
    """
    Shalf = np.sqrt(map_class.cl_th[:map_class.lmax+1])
    alm_out = hp.almxfl(real2complex_alm(map_class.alm),Shalf)
    cl_out = hp.alm2cl(alm_out)
    map_out = hp.alm2map(alm_out,map_class.nside)
    return cl_out,map_out

class data_class:
    """
    data_class has attributes: alm, cl_th, beam, sigma, lmax, nside
    """
    alm = 0
    cl_th = 0
    beam = 0
    sigma = 0
    invvar = 0
    lmax = 0
    nside = 0
    def __init__(self, vec):
        self.alm = vec[0]
        self.cl_th = vec[1]
        self.beam = vec[2]
        self.sigma = vec[3]
        self.invvar = vec[4]
        self.lmax = vec[5]
        self.nside = vec[6]

def CG_algo_dirty(Matrix,b,data_start,i_max,eps):
    """
    Matrix is the function to apply the matrix on a vector
    data_start is a data_class class, with real alms
    """
    i = 0
    x = data_start.alm.copy() 
    cl_th = data_start.cl_th
    beam = data_start.beam
    sigma = data_start.sigma
    invvar = data_start.invvar
    lmax = data_start.lmax
    nside = data_start.nside
    out = data_class([0,cl_th,beam,sigma,invvar,lmax,nside])
    #print b
    #print data_start.alm
    r = b-Matrix(data_start)#np.dot(A,x)
    print "r[10] = ",r[10]
    d = data_class([0,cl_th,beam,sigma,invvar,lmax,nside])
    d.alm = r.copy()
    print "ini ","start = ",data_start.alm[10],"x = ",x[10],"d = ",d.alm[10]
    delt_n = np.dot(r.T,r)
    delt_0 = delt_n.copy()
    iter_out_map=[]    
    iter_out_cl=[]
    while (i<i_max and delt_n > (eps**2 * delt_0)):
        q = Matrix(d)#np.dot(A,d)
        #hp.mollview(hp.alm2map(real2complex_alm(q),1024),title = "q loop %d"%i)
        alph = np.float(delt_n) / np.dot(d.alm.T,q)
        x = x + alph*d.alm
        #hp.mollview(hp.alm2map(real2complex_alm(x),1024),title = "x loop %d"%i)
        #print "test : r[50] = ",r[50],"d[50] = ",d.alm[50],"x[50] = ",x[50],"b[50] = ",b[50]
        if i%10==0:
            #print "in here"
            dat_temp = data_class([0,cl_th,beam,sigma,invvar,lmax,nside])
            dat_temp.alm = x
            r = b - Matrix(dat_temp)
        else:
            r = r - alph*q
        #hp.mollview(hp.alm2map(real2complex_alm(r),1024),title = "r loop %d"%i)
        delt_old = delt_n.copy()
        delt_n = np.dot(r.T,r)
        #print "delt_n = ",delt_n,"delt_old = ",delt_old
        bet = delt_n / delt_old
        d.alm = r + bet*d.alm
        #hp.mollview(hp.alm2map(real2complex_alm(d.alm),1024),title = "d loop %d"%i)
        i += 1
        #print i,"start = ",data_start.alm[10],"alpha = ",alph,"bet = ",bet
        out.alm = x
        #plt.figure(10)
        #plt.plot(np.log10(hp.alm2cl(real2complex_alm(x))))
        iter_out_map.append(return_map(out)[1])
        iter_out_cl.append(return_map(out)[0])
        #plt.figure(100),plt.plot(np.log10(ret[0]),label='%d'%i)
        #hp.mollview(ret[1],title = "%d"%i)
        #print i, np.sqrt(delt_n/delt_0),alph,bet
        print i,np.sqrt(delt_n/delt_0),"r[50] = ",r[50],"q[50] = ",q[50],"x[50] = ",x[50],"d[50] = ",d.alm[50]#,"out.alm[50] = ",out.alm[50]
    return iter_out_map,iter_out_cl


def CG_algo(Matrix,b,data_start,i_max,eps):
    """
    Cf. algorithm B2 from Shewchuk 94 (page 50)
    Matrix is the function to apply the matrix on a vector (eg A_matrix_func)
    data_start is a data_class class, with real alms
    """
    i = 0
    x = data_start.alm.copy()
    cl_th = data_start.cl_th
    beam = data_start.beam
    sigma = data_start.sigma
    invvar = data_start.invvar
    lmax = data_start.lmax
    nside = data_start.nside
    out = data_class([0,cl_th,beam,sigma,invvar,lmax,nside])
    r = b-Matrix(data_start)
    d = data_class([0,cl_th,beam,sigma,invvar,lmax,nside])
    d.alm = r.copy()
    delt_n = np.dot(r.T,r)
    delt_0 = delt_n.copy()
    iter_out_map=[]
    iter_out_cl=[]
    res = []
    x_list =[]
    while (i<i_max and delt_n > (eps**2 * delt_0)):
        q = Matrix(d)
        alph = np.float(delt_n) / np.dot(d.alm.T,q)
        x = x + alph*d.alm
        if i%10==0:
            dat_temp = data_class([0,cl_th,beam,sigma,invvar,lmax,nside])
            dat_temp.alm = x
            r = b - Matrix(dat_temp)
        else:
            r = r - alph*q
        delt_old = delt_n.copy()
        delt_n = np.dot(r.T,r)
        bet = delt_n / delt_old
        d.alm = r + bet*d.alm
        i += 1
        out.alm = x
        res.append(hp.alm2cl(real2complex_alm(r-(b-Matrix(data_start))))/hp.alm2cl(real2complex_alm(b-Matrix(data_start))))
        #x_list.append(real2complex_alm(x))
        iter_out_map.append(return_map(out)[1])
        iter_out_cl.append(return_map(out)[0])
    return iter_out_map,iter_out_cl,res#x_list



def CG_algo_precond_diag(Matrix,Precond_diag,b,data_start,i_max,eps):
    """
    Matrix is the function to apply the matrix on a vector 
    data_start is a data_class class, with real alms 
    """
    i = 0
    x = data_start.alm.copy()
    cl_th = data_start.cl_th
    beam = data_start.beam
    sigma = data_start.sigma
    lmax = data_start.lmax
    nside = data_start.nside
    invvar = data_start.invvar
    out = data_class([0,cl_th,beam,sigma,invvar,lmax,nside])
    r = b-Matrix(data_start)
    d = data_class([0,cl_th,beam,sigma,invvar,lmax,nside])
    d.alm = r.copy()
    d.alm = complex2real_alm(hp.almxfl(real2complex_alm(r),Precond_diag))
    delt_n = np.dot(r.T,d.alm)
    delt_0 = delt_n.copy()
    iter_out_map=[]
    iter_out_cl=[]
    while (i<i_max and delt_n > (eps**2 * delt_0)):
        q = Matrix(d)
        alph = np.float(delt_n) / np.dot(d.alm.T,q)
        x = x + alph*d.alm
        if i%10==0:
            dat_temp = data_class([0,cl_th,beam,sigma,invvar,lmax,nside])
            dat_temp.alm = x
            r = b - Matrix(dat_temp)
        else:
            r = r - alph*q
        s = complex2real_alm(hp.almxfl(real2complex_alm(r),Precond_diag))
        #s = hp.almxfl(r,Precond_diag)
        delt_old = delt_n.copy()
        delt_n = np.dot(r.T,s)
        bet = delt_n / delt_old
        d.alm = s + bet*d.alm
        i += 1
        out.alm = x
        iter_out_map.append(return_map(out)[1])
        iter_out_cl.append(return_map(out)[0])
    return iter_out_map,iter_out_cl





def generate_w1term(Cl,bl,nl):
    """
    
    """
    lmax = np.size(Cl) -1
    onesl = np.ones(lmax+1)
    onesl[:2]=0
    w1 = hp.synalm(onesl)
    #### make sure no mono- and dipole
    #w1 = 1/np.sqrt(2)*(np.random.randn(hp.Alm.getsize(lmax)) + 1j*np.random.randn(hp.Alm.getsize(lmax)))
    out = hp.almxfl(w1,np.sqrt(Cl[:lmax+1]/nl[:lmax+1])*bl[:lmax+1])
    return out

def generate_w0term(Cl):
    """
    
    """
    lmax = np.size(Cl)-1
    onesl = np.ones(lmax+1)
    onesl[:2]=0
    #### make sure no mono- and dipole
    w0 = hp.synalm(onesl)
    #w0 = hp.synalm(np.ones(lmax))#1/np.sqrt(2)*(np.random.randn(hp.Alm.getsize(lmax)) + 1j*np.random.randn(hp.Alm.getsize(lmax)))
    return w0


def generate_mfterm(dlm,Cl,bl,nl):
    """
    
    """
    lmax = np.size(Cl)-1
    out = hp.almxfl(dlm,np.sqrt(Cl[:lmax+1])*bl[:lmax+1]/nl[:lmax+1])
    return out

def renorm_term(Cl,bl,nl):
    """
    
    """
    lmax = np.size(Cl)-1 
    return np.sqrt(Cl[:lmax+1])/(1+Cl[:lmax+1]*bl[:lmax+1]**2/(nl[:lmax+1]))
    
   
