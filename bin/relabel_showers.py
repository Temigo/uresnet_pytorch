#!/usr/bin/python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from scipy import spatial
import scipy as sp
import numpy as np
import argparse
import tempfile
import time
from sklearn.cluster import DBSCAN

def prepare(input_files,output_file):
    """
    Prepares larcv IO manager.
    input_files is a string list of input data files
    output_file is a string name of output file
    return is larcv.IOManager instance pointer
    """
    outfile = output_file
    infile="["
    for fname in input_files:
        infile += '"%s",' % fname
    infile=infile.rstrip(',')
    infile+=']'

    cfg = '''
    IOManager: {
      IOMode: 2
      OutFileName: "%s"
      InputFiles: %s
    }
    '''
    cfg = cfg % (outfile,str(infile))
    cfg_file = tempfile.NamedTemporaryFile('w')
    cfg_file.write(cfg)
    cfg_file.flush()
    from larcv import larcv
    io=larcv.IOManager(cfg_file.name)
    io.initialize()
    return io

def dbscan(voxels,values,
    find_val=None,find_valmax=None,eps=2.8284271247461903,min_samples=1):
    """
    voxels is a list of ALL points' coordinate, dtype float32 np array shape (N,3)
    values is a list of ALL points' label value, dtype float32 np array shape (N,1)
    return is a (DBSCAN) clustered set of voxels and their indices in the original voxels and values array.
    """
    
    if find_val is not None:
        selection = np.where(values == find_val)[0]
    elif find_valmax is not None:
        selection = np.where(values <= find_valmax)[0]
    if len(selection) == 0:
        return [],[]
    sel_vox = voxels[selection]
    res=DBSCAN(eps=eps+1.e-6,min_samples=min_samples,metric='euclidean').fit(sel_vox)
    cls_idx = [ selection[np.where(res.labels_ == i)] for i in range(np.max(res.labels_)+1) ]
    cls_vox = [ voxels[cls_idx[i]] for i in range(len(cls_idx)) ]

    return cls_idx, cls_vox

def correlate(trunks,branches,eps=2.8284271247461903):
    """
    trunks represent a list of HIP/MIP pixel clusters DBSCAN-ed. 
    branches represent a list of michel or delta pixel clusters DBSCAN-ed.
    Both trunks and branches should be a list of clusters where a cluster is a group of point represented in np array of shape (N,3).
    return is an integer array with the length same as branches filled with a value either 0 or 1.
    Return value 0 means a branch should be relabed as Shower, and 1 means a branch should not be relabeled.
    """
    
    result=np.zeros(shape=[len(branches)],dtype=np.int32)
    for i, branch in enumerate(branches):
        correlated = False
        for trunk in trunks:
            cdist = np.min(sp.spatial.distance.cdist(trunk,branch,'euclidean'))
            if cdist < eps:
                correlated=True
                break
        if correlated: result[i]=1
    return result

def process(input_data,output_data):
    """
    input_data and output_data types should be larcv.SparseTensor3D.
    input_data holds 3D segmentation label values per voxel, (0,1,2,3,4) = (HIP,MIP,Shower,Delta,Michel)
    output_data should be empty (or will be overwritten)
    return is None
    This function will change some voxels of (Delta,Michel) into Shower.
    Those remain unchanged are the primary ionization trajectories attached to (HIP,MIP) pixels.
    """
    if input_data.as_vector().size() < 1:
        output_data.set(input_data,input_data.meta())
        return

    #t0=time.time()
    voxels = np.zeros(shape=[input_data.as_vector().size(),3],dtype=np.int32)
    values = np.zeros(shape=[input_data.as_vector().size(),1],dtype=np.float32)
    #if debug>0: print('c0', time.time()-t0)

    #t0=time.time()
    from larcv import larcv
    larcv.fill_3d_voxels(input_data,voxels)
    larcv.fill_3d_pcloud(input_data,values)
    #if debug>0: print('c1', time.time()-t0)
    values=values.squeeze(axis=1)

    #t0=time.time()
    no_correction = len(np.where(values>2)[0]) == 0
    trivial_correction = len(np.where(values<3)[0]) == 0
    #if debug>0: print('c2', time.time()-t0)

    # Nothing to correct, return
    if no_correction:
        output_data.set(input_data,input_data.meta())
        return

    # Only michel/delta ray, make them all shower
    if trivial_correction:
        values[:] = 2.
        vs=larcv.as_tensor3d(voxels,values,input_data.meta(),-1.)
        output_data.set(vs,input_data.meta())
        return

    # Reaching here means something to correct.
    # DBSCAN 
    #t0=time.time()
    others_idx,  others_vox  = dbscan(voxels,values,find_valmax=1.)
    deltas_idx,  deltas_vox  = dbscan(voxels,values,find_val=3.,min_samples=1)
    michels_idx, michels_vox = dbscan(voxels,values,find_val=4.,min_samples=1)
    #if debug>0: print('c3', time.time()-t0)

    #t0=time.time()
    correlated_deltas = correlate(others_vox,deltas_vox)
    #if debug>0: print('c4', time.time()-t0)

    #t0=time.time()
    correlated_michels = correlate(others_vox,michels_vox)
    #if debug>0: print('c5', time.time()-t0)

    #t0=time.time()
    for i, correlation in enumerate(correlated_deltas):
        if correlation > 0: continue
        values[deltas_idx[i]] = 2.
    for i, correlation in enumerate(correlated_michels):
        if correlation > 0: continue
        values[michels_idx[i]] = 2.
    #if debug>0: print('c6', time.time()-t0)

    vs=larcv.as_tensor3d(voxels,values,input_data.meta(),-1.)
    output_data.set(vs,input_data.meta())

    return

class timing:
    """
    A silly class to hold timing information
    """
    def __init__(self):
        self.read,self.write,self.proc=[0.,0.,0.]
        self.read_range  = [1.e6,-1.]
        self.write_range = [1.e6,-1.]
        self.proc_range  = [1.e6,-1.]
        self.ctr = 0.

    def set_tread(self,t):
        t *= 1000.
        self.read += t
        if t < self.read_range[0]: self.read_range[0]=t
        if t > self.read_range[1]: self.read_range[1]=t

    def set_twrite(self,t):
        t *= 1000.
        self.write += t
        if t < self.write_range[0]: self.write_range[0]=t
        if t > self.write_range[1]: self.write_range[1]=t

    def set_tproc(self,t):
        t *= 1000.
        self.proc += t
        if t < self.proc_range[0]: self.proc_range[0]=t
        if t > self.proc_range[1]: self.proc_range[1]=t

    def report(self):
        msg  = 'Processed %d entries\n' % int(self.ctr)
        msg += 'Average'
        msg += ' read %g [ms] ...' % ( int(100.*self.read/self.ctr)/100. )
        msg += ' write %g [ms] ...' % ( int(100.*self.write/self.ctr)/100. )
        msg += ' proc %g [ms]\n' % ( int(100.*self.proc/self.ctr)/100. )
        msg += '  read  range: %g => %g [ms]\n' % ( int(100.*self.read_range[0])/100.,  int(100.*self.read_range[1])/100.)
        msg += '  write range: %g => %g [ms]\n' % ( int(100.*self.write_range[0])/100., int(100.*self.write_range[1])/100.)
        msg += '  proc  range: %g => %g [ms]\n' % ( int(100.*self.proc_range[0])/100.,  int(100.*self.proc_range[1])/100.)
        return msg

def main():
    """
    A main function to be executed
    """

    parser = argparse.ArgumentParser(description='Michel/Delta-ray label correction script')
    parser.add_argument('-if','--input_file',type=str,help='Input files (comma separated)')
    parser.add_argument('-of','--output_file',type=str,help='Output file name')
    parser.add_argument('-il','--input_label',type=str,default='label',help='Input data product label [default: label')
    parser.add_argument('-ol','--output_label',type=str,default='label',help='Output data product label [default: label]')
    parser.add_argument('-s','--start',type=int,default=0,help='Start entry [default: 0]')
    parser.add_argument('-n','--num',type=int,default=-1,help='Number of entries to process [default: -1]')
    parser.add_argument('-r','--report',type=int,default=100,help='Number of steps to print out process record [default: 100]')
    #parser.add_argument('-d','--debug',type=int,default=0,help='Enable debug mode [default: 0]')
    args = parser.parse_args()

    if args.input_file is None or args.output_file is None:
        print('Error: --input_file/-if and --output_file/-of need to be provided!')
        print('Try --help.')
        return 1

    input_files = args.input_file.split(',')
    output_file = args.output_file
    input_label = args.input_label
    output_label = args.output_label

    io = prepare(input_files,output_file)

    total_entries = io.get_n_entries() - args.start
    if args.num > 0 and args.num < total_entries:
        total_entries = args.num

    tspent=timing()

    current_entry = args.start
    while tspent.ctr < total_entries:

        t0=time.time()
        io.read_entry(current_entry)
        data_input  = io.get_data('sparse3d',input_label)
        data_output = io.get_data('sparse3d',output_label)
        tspent.set_tread(time.time() - t0)

        t0=time.time()
        process(data_input,data_output)
        tspent.set_tproc(time.time() - t0)

        t0=time.time()
        io.save_entry()
        tspent.set_twrite(time.time() - t0)

        tspent.ctr += 1.

        if int(tspent.ctr) % args.report == 0:
            print(tspent.report())

        current_entry += 1

    io.finalize()

main()
