from larcv import larcv
from ROOT import TChain
import sys

file1 = sys.argv[1]
file2 = sys.argv[2]

ch_blob = {}
br_blob = {}

ch_blob['sparse3d_data'] = TChain('sparse3d_data_tree')
ch_blob['sparse3d_label'] = TChain('sparse3d_label_tree')
ch_blob['sparse3d_predictions'] = TChain('sparse3d_prediction_tree')

ch_blob['particle_mcst'] = TChain('particle_mcst_tree')
ch_blob['cluster3d_mcst'] = TChain('cluster3d_mcst_tree')

ch_blob['sparse3d_data'].AddFile(file1)
ch_blob['sparse3d_label'].AddFile(file1)
ch_blob['sparse3d_predictions'].AddFile(file1)

ch_blob['particle_mcst'].AddFile(file2)
ch_blob['cluster3d_mcst'].AddFile(file2)

ach = ch_blob.values()[0]
for i in range(ach.GetEntries()):
    for key, ch in ch_blob.iteritems():
        ch.GetEntry(i)
        if i == 0:
            br_blob[key] = getattr(ch, '%s_branch' % key)
            
        print(dir(ch))