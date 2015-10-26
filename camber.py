import numpy as np
import os
import subprocess
import sys

try:
    from local_paths import *
except:
    print "you need to define local_paths.py, that defines, for example: \ncamb_dir = '/Users/benjar/Travail/camb/' \n and the output path for the temporary ini files: \noutput_camb = '../MH_MCMC/camb_ini/test1'"
    sys.exit()
#camb_dir -- path where camb should be called



def ini2dic(filename):
    """
    Creates a dictionnary from a camb .ini file

    Keyword Arguments:
    filename -- filename.ini where the parameters will be read
    """
    dic = {}
    file = open(filename,'r')
    for line in file.readlines():
        ind = line.find("=")
        try:
            aa = int(line[ind+2:-1])
        except:
            try:
                aa = float(line[ind+2:-1])
            except:
                aa = line[ind+2:-1]
        dic[line[:ind-1]]=aa
    return dic
    
    
def dic2ini(dic,filename):
    """
    Writes a .ini file from a dictionnary

    Keyword Arguments:
    filename -- filename.ini where the dictionary will be written
    dic -- input parameter dictionary
    """
    with open(filename, "w") as f:
        for (name, mobile) in dic.items():
            f.write("%s = %s\n"%(name,mobile))
    f.close()


def run_camb(dic,hide=1,filename = "temporary_ini_file.ini"):
    """
    Call Camb for a given dictionary of parameters

    Keyword Arguments:
    filename -- filename.ini where the temporary .ini file will be written
    """
    dic2ini(dic,filename)
    current_dir = os.getcwd()
    os.chdir(camb_dir)
    if hide==1:
        with open(os.devnull, "w") as f:
            subprocess.call("./camb %s/%s"%(current_dir,filename),shell = True, stdout=f)
    else:
        subprocess.call("./camb %s/%s"%(current_dir,filename),shell = True)
    os.chdir(current_dir)

    
def generate_spectrum(dic):
    """
    Generates power spectra from a given param dictionnary
    !!!!! for now, only works for unlensed scalar Cl, easy to modify though
 
    Keyword Arguments:
    dic -- parameter dictionnary
    """
    Pid = np.random.randint(0,1000000)
    run_camb(dic,1,"tempopo%s.ini"%Pid)
    #try:
    Cl = np.loadtxt("%s/%s_scalCls.dat"%(camb_dir,dic["output_root"]))
    # camb generates l(l+1)Cl/2/pi from lmin of 2 to lmax,
    # we want Cl from lmin = 0 for healpix : 
    l1l2 = np.zeros((2,Cl.shape[-1]))
    l1l2[:,0] = [0,1]
    Cl = np.concatenate((l1l2,Cl))
    renorm = np.arange(Cl.shape[0])*np.arange(1,Cl.shape[0]+1)/2/np.pi
    Cl[1:,1:]/=renorm[1:,np.newaxis]
    os.system("rm -rf tempopo%s.ini"%Pid)
    os.system("rm -rf %s/%s_scalCls.dat"%(camb_dir,dic["output_root"]))
    return Cl


def update_dic(dic,new_par,strings):
    for i in range(np.size(new_par)):
        if strings[i]=='scalar_amp(1)':
            #print params[strings[i]]
            dic[strings[i]]=np.exp(new_par[i])*1e-10
            #print params[strings[i]]
        else:
            dic[strings[i]]=new_par[i]
    return dic
