#
# # NAMD.py
# # Written by Chris Lockhart in Python3
#
# import numpy as np
# import pandas as pd
# import subprocess
#
# # NAMD class, which launches simulations. Eventually this will do more error
# # checking; however, for today it's bare minimum. This will break if there
# # are duplicate parameters in the config file.
# class NAMD:
#     def __init__(self,name='config',exe='./namd2',
#                  num_core=None,background=False):
#         # Name of this configuration (without extension)
#         self.name=name
#
#         # NAMD executable
#         self.exe=exe
#
#         # Number of cores
#         self.num_core=num_core
#         if num_core is not None and not isinstance(num_core,int):
#             raise AttributeError('num_core must be None or integer')
#
#         # Should job be run on background?
#         self.background=background
#
#     def execute(self):
#         # Run
#         command=[self.exe,self.name+'.namd']
#         if isinstance(self.num_core,int):
#             command=[self.exe,'+p'+str(self.num_core),self.name+'.namd']
#         with open(self.name+'.out','w') as file:
#             proc=subprocess.Popen(command,stdout=file)
#
#         # Wait?
#         if not self.background:
#             proc.wait()
#             if proc.poll() != 0:
#                 raise RuntimeError('NAMD job did not finish successfully')
#
#         # If the job should be run in the background, return process
#         #if self.background: return proc
#         return proc
#
#     def read_config(self,filename=None):
#         filename=self.name+'.namd' if filename is None else str(filename)
#
#         file=open(filename,'r')
#         self.parameters=file.read()
#         file.close()
#
#     def read_output(self):
#         return read_NAMD_output(self.name+'.out')
#
#     def write_config(self):
#         file=open(self.name+'.namd','w')
#         file.write(self.parameters)
#         file.close()
#
# # Function for reading NAMD output
# def read_NAMD_output(fname,ignore_first=False):
#     file=open(str(fname),'r')
#     lines=file.read().split('\n')
#     file.close()
#     keys=None
#     values=[]
#     for line in lines:
#         if line[:6] == 'ETITLE':
#             keys_=line.split()[1:]
#             if keys is None: keys=keys_
#             if keys != keys_:
#                 raise ValueError('keys change')
#         elif line[:6] == 'ENERGY':
#             values_=line.split()[1:]
#             values.append(values_)
#     if ignore_first: values=values[1:]
#     values=np.array(values,dtype=float)
#     df=pd.DataFrame()
#     for i,key in enumerate(keys):
#         df[key]=pd.Series(values[:,i])
#     return df