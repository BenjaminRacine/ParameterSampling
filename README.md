# ParameterSampling
Version of the code used for https://arxiv.org/abs/1512.06619. 

Improvement TBD: 
- Needs cleaning and documentation for public usage, but I can try my best if anyone wants to use it. 
- The notebooks are quite old and issues mentioned there have been fixed.
- MPI instead of this multiple job submission !!
- camber is a simple hacky way to call camb from python, the python module now available on the camb website is probably much more efficient. 



1) git pull.

2) Then, local_paths.py should be changed for to your camb directory.

3) "mkdir log", "mkdir outputs" and "mkdir plots"

4) run "python MCMC_main_script.py”
- with Method = 0
- with Method = 1

