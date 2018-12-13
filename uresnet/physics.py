from larcv import larcv
from ROOT import TChain
import sys
import numpy as np
import utils

file1 = sys.argv[1]
file2 = sys.argv[2]

ch_blob = {}
br_blob = {}

ch_blob['sparse3d_data'] = TChain('sparse3d_data_tree')
ch_blob['sparse3d_label'] = TChain('sparse3d_label_tree')
ch_blob['sparse3d_prediction'] = TChain('sparse3d_prediction_tree')

ch_blob['particle_mcst'] = TChain('particle_mcst_tree')
ch_blob['cluster3d_mcst'] = TChain('cluster3d_mcst_tree')

ch_blob['sparse3d_data'].AddFile(file1)
ch_blob['sparse3d_label'].AddFile(file1)
ch_blob['sparse3d_prediction'].AddFile(file1)

ch_blob['particle_mcst'].AddFile(file2)
ch_blob['cluster3d_mcst'].AddFile(file2)

num_entries = ch_blob['sparse3d_data'].GetEntries()
# print(ch_blob['sparse3d_data'].GetEventList().Pop())
# print(ch_blob['sparse3d_data'].GetEntryList())
# nu_current_type = {}
# nu_interaction_type = {}
creation_process = {}
print(num_entries)
for i in range(num_entries):
    print(i)
    ch_blob['sparse3d_data'].GetEntry(i)
    event_id = ch_blob['sparse3d_data'].sparse3d_data_branch.event()
    print(event_id)
    for key, ch in ch_blob.iteritems():
        ch.GetEntry(i)
        # print(dir(ch))
        #ch.GetEntryWithIndex(event_id)
        # ch.GetEvent(event_id)
        print(ch.GetBranch('%s_branch' % key))
        # print(key, dir(ch))
        if i == 0:
            br_blob[key] = getattr(ch, '%s_branch' % key)
            print(key, ch, br_blob[key], br_blob[key].event(), dir(br_blob[key]))

    num_point = br_blob['sparse3d_data'].as_vector().size()

    # Data
    np_data = np.zeros(shape=(num_point, 3), dtype=np.int32)
    larcv.fill_3d_voxels(br_blob['sparse3d_data'], np_data)

    # Labels
    labels = larcv.as_ndarray(br_blob['sparse3d_label'])

    # Predictions
    predictions = larcv.as_ndarray(br_blob['sparse3d_prediction'])

    # Particles
    particles = br_blob['particle_mcst'].as_vector()
    clusters = br_blob['cluster3d_mcst'].as_vector()

    metrics = {
        'acc': []
    }
    metrics_logger = utils.CSVData(sys.argv[3])

    for i, particle in enumerate(particles):
        cluster = clusters[i].as_vector()
        num_voxels = cluster.size()
        if num_voxels > 0:
            x = np.zeros((num_voxels,), dtype=np.int32)
            y = np.zeros((num_voxels,), dtype=np.int32)
            z = np.zeros((num_voxels,), dtype=np.int32)
            value = np.zeros((num_voxels,), dtype=np.float32)
            larcv.as_flat_arrays(clusters[i], br_blob['sparse3d_data'].meta(), x, y, z, value)
            cluster = np.stack([x, y, z, value], axis=1)
            # print(cluster.shape, particle.pdg_code(), particle.energy_deposit(),
            #       particle.nu_interaction_type(), particle.creation_process(),
            #       particle.distance_travel(), particle.energy_init(),
            #       # particle.momentum(),
            #       particle.nu_current_type())
            # for voxel in cluster:
            #     print(voxel.id())
            # if particle.nu_interaction_type() not in nu_interaction_type:
            #     nu_interaction_type[particle.nu_interaction_type()] = len(nu_interaction_type)
            # if particle.nu_current_type() not in nu_current_type:
            #     nu_current_type[particle.nu_current_type()] = len(nu_current_type)
            if particle.creation_process() not in creation_process:
                creation_process[particle.creation_process()] = len(creation_process)

            nonzero_index = labels[x, y, z] > 0
            print(nonzero_index.sum(), np.unique(labels))
            particle_acc = (labels[x, y, z] == predictions[x, y, z]).astype(np.int32).sum() / float(num_voxels)
            metrics_logger.record(('acc', 'num_voxels', 'id'),
                                  (particle_acc, num_voxels, particle.id()))
            metrics_logger.record(('pdg_code', 'energy_deposit', 'energy_init'),
                                  (particle.pdg_code(), particle.energy_deposit(), particle.energy_init()))
            metrics_logger.record(('nu_current_type', 'nu_interaction_type'),
                                  (particle.nu_current_type(), particle.nu_interaction_type()))
            metrics_logger.record(('creation_process', 'distance_travel'),
                                  (creation_process[particle.creation_process()], particle.distance_travel()))
            metrics_logger.write()

    metrics_logger.close()

print(creation_process)
