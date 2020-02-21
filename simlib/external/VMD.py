#
# # VMD.py
# # Written by Chris Lockhart in Python3
#
# # Import packages
# import LLTK.external
# import os
# import socket
# import subprocess
# import tempfile
#
# # The class VMD starts up an instance of VMD that Python can connect
# # with. We can send commands, send structures, and close VMD.
# class VMD:
#     def __init__(self,port=45000,exe='C:\\Program Files (x86)\\University of Illinois\\VMD\\vmd.exe'):
#         self.port=int(port) # Port to open server on
#         self.exe=str(exe) # VMD executable
#         self.connected=False # Is VMD currently connected? Start at False
#
#     # Close VMD and the socket
#     def close(self):
#         # Mark as no longer connected
#         self.connected=False
#
#         # If the VMD process is no longer available, return error
#         if self.vmd_process.poll() is not None:
#             raise RuntimeError('VMD already terminated')
#
#         # Exit VMD
#         self.socket.sendall(b'exit\n')
#
#     # Connect to VMD
#     def connect(self):
#         # Launch VMD
#         vmd_server_script=\
#             (LLTK.external.__path__[0]+'/vmd_server.tcl').replace('\\','/')
#         with tempfile.TemporaryFile() as file:
#             self.vmd_process=\
#                 subprocess.Popen([self.exe,'-e',vmd_server_script,'-args',
#                                   str(self.port)],stdout=subprocess.PIPE,
#                                   stdin=file,stderr=subprocess.PIPE)
#
#         # Set up the socket
#         self.socket=socket.socket()
#
#         # Wait for VMD to load and then connect
#         # There is probably a circumstance where this results in an
#         # infinite loop --> should there be a timeout?
#         while not self.connected:
#             try:
#                 self.socket.connect(('localhost',self.port))
#                 self.connected=True
#             except ConnectionRefusedError:
#                 pass
#
#         # Mark as connected
#         self.connected=True
#
#     # Load a structure into VMD
#     def load(self,structure):
#         # Create temporary file
#         file=tempfile.NamedTemporaryFile(delete=False)
#         file.close()
#
#         # Get temporary file name
#         fname=file.name.replace('\\','/')
#
#         # Write out structure (just PDB for now; eventually PSF/DCD?)
#         structure.write_pdb(fname)
#
#         # Execute load command on VMD
#         self.execute('mol new '+fname+' waitfor all')
#
#         # Delete temporary file
#         os.remove(fname)
#
#     # Execute command
#     def execute(self,command):
#         # If the socket isn't open, don't do anything
#         if not self.connected:
#             raise RuntimeError('connection already terminated')
#
#         # If the VMD process is no longer available, return error
#         if self.vmd_process.poll() is not None:
#             raise RuntimeError('VMD already terminated')
#
#         # Execute the command
#         command=str(command)+'\n' # Need newline
#         command=command.replace('$','$::') # Need to use global namespace
#         command=command.replace('[','[uplevel #0 ') # For global namespace
#         self.socket.sendall(command.encode())
#
#         # Receive results (should buffer be user-specified?)
#         result=self.socket.recv(4096).decode('ascii').strip()
#
#         # Split retcode from rest of results
#         retcode=int(result[0])
#
#         # Raise error
#         if retcode:
#             raise RuntimeError('VMD command error')
#
#         # Get remaining results and return
#         result=result[1:]
#         if len(result) != 0: return result
#
