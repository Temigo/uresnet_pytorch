from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import time
import datetime
import sys
import numpy as np
import nu_net
import torch


def iotest(flags):
    # IO configuration
    io = nu_net.io_factory(flags)
    io.initialize()
    num_entries = io.num_entries()
    ctr = 0
    data_key = flags.DATA_KEYS[0]
    while ctr < num_entries:
        idx,blob=io.next()
        msg = str(ctr) + '/' + str(num_entries) + ' ... '  + str(idx) + ' ' + str(blob[data_key][0].shape)
        for key in flags.DATA_KEYS:
            if key == data_key: continue
            msg += str(blob[key][0].shape)
        print(msg)
        ctr += len(data)
    io.finalize()


class Handlers:
    sess         = None
    data_io      = None
    csv_logger   = None
    weight_io    = None
    train_logger = None
    iteration    = 0


def train(flags):
    flags.TRAIN = True
    handlers = prepare(flags)
    train_loop(flags, handlers)


def inference(flags):
    flags.TRAIN = False
    handlers = prepare(flags)
    inference_loop(flags, handlers)


def prepare(flags):
    torch.cuda.set_device(flags.GPUS[0])
    handlers = Handlers()

    # IO configuration
    handlers.data_io = nu_net.io_factory(flags)
    handlers.data_io.initialize()
    if 'sparse' in flags.IO_TYPE:
        handlers.data_io.start_threads()
        handlers.data_io.next()
    if 'sparse' in flags.MODEL_NAME and 'sparse' not in flags.IO_TYPE:
        sys.stderr.write('UResNet needs sparse IO.')
        sys.exit(1)

    # Trainer configuration
    flags.NUM_CHANNEL = handlers.data_io.num_channels()
    handlers.trainer = nu_net.trainval(flags)

    # Restore weights if necessary
    handlers.iteration = 0
    loaded_iteration   = handlers.trainer.initialize()
    if flags.TRAIN: handlers.iteration = loaded_iteration

    # Weight save directory
    if flags.WEIGHT_PREFIX:
        save_dir = flags.WEIGHT_PREFIX[0:flags.WEIGHT_PREFIX.rfind('/')]
        if save_dir and not os.path.isdir(save_dir): os.makedirs(save_dir)

    # Log save directory
    if flags.LOG_DIR:
        if not os.path.exists(flags.LOG_DIR): os.mkdir(flags.LOG_DIR)
        logname = '%s/train_log-%07d.csv' % (flags.LOG_DIR, loaded_iteration)
        if not flags.TRAIN:
            logname = '%s/inference_log-%07d.csv' % (flags.LOG_DIR, loaded_iteration)
        handlers.csv_logger = nu_net.utils.CSVData(logname)

    return handlers


def get_keys(flags):
    data_key = flags.DATA_KEYS[0]
    grp_key, pdg_key, label_key, weight_key = (None, None, None, None)
    if flags.TRAIN:
        data_key, grp_key, pdg_key = flags.DATA_KEYS[0:3]
        if len(flags.DATA_KEYS) > 3:
            label_key = flags.DATA_KEYS[3]
        if len(flags.DATA_KEYS) > 4:
            weight_key = flags.DATA_KEYS[4]
    else:
        if len(flags.DATA_KEYS) > 1:
            grp_key = flags.DATA_KEYS[1]
        if len(flags.DATA_KEYS) > 2:
            pdg_key = flags.DATA_KEYS[2]
        if len(flags.DATA_KEYS) > 3:
            label_key = flags.DATA_KEYS[3]
        if len(flags.DATA_KEYS) > 4:
            weight_key = flags.DATA_KEYS[4]
    return data_key, grp_key, pdg_key, label_key, weight_key


def log(handlers, tstart_iteration, tsum, label, res, alpha,
        flags, idx, epoch, tstamp_iteration):
    report_step  = flags.REPORT_STEP and ((handlers.iteration+1) % flags.REPORT_STEP == 0)

    pred_seg = res['segmentation']
    loss_seg = res['loss_seg']
    acc_seg = 0.
    if label is not None:
        acc_seg = np.mean(nu_net.utils.compute_accuracy(handlers.data_io,
                                                        [idx],
                                                        label,
                                                        pred_seg))
    # Report (logger)
    tspent_iteration = time.time() - tstart_iteration
    if handlers.csv_logger:
        tsum += tspent_iteration
        handlers.csv_logger.record(('iter', 'epoch', 'titer', 'tsumiter'),
                                   (handlers.iteration,epoch,tspent_iteration,tsum))
        handlers.csv_logger.record(('tio', 'tsumio'),
                                   (handlers.data_io.tspent_io,handlers.data_io.tspent_sum_io))
        tmap, tsum_map = handlers.trainer.tspent, handlers.trainer.tspent_sum
        if flags.TRAIN:
            handlers.csv_logger.record(('ttrain','tsave','tsumtrain','tsumsave'),
                                       (tmap['train'],tmap['save'],tsum_map['train'],tsum_map['save']))
        else:
            handlers.csv_logger.record(('tforward','tsave','tsumforward','tsumsave'),
                                       (tmap['forward'],tmap['save'],tsum_map['forward'],tsum_map['save']))

        handlers.csv_logger.record(('loss_seg','acc_seg'),(loss_seg,acc_seg))
        handlers.csv_logger.write()

    # Report (stdout)
    if report_step:
        loss_seg   = nu_net.utils.round_decimals(loss_seg,   4)
        tmap  = handlers.trainer.tspent
        tfrac = nu_net.utils.round_decimals(tmap['train']/tspent_iteration*100., 2)
        tabs  = nu_net.utils.round_decimals(tmap['train'], 3)
        epoch = nu_net.utils.round_decimals(epoch, 2)
        mem = nu_net.utils.round_decimals(torch.cuda.max_memory_allocated()/1.e9, 3)
        msg = 'Iter. %d (epoch %g) @ %s ... ttrain %g%% (%g [s]) mem. %g GB \n'
        msg = msg % (handlers.iteration, epoch, tstamp_iteration, tfrac, tabs, mem)
        msg += '   Segmentation: loss %g accuracy %g\n' % (loss_seg, acc_seg)
        print(msg)
        sys.stdout.flush()
        if handlers.csv_logger: handlers.csv_logger.flush()
        if handlers.train_logger: handlers.train_logger.flush()


def train_loop(flags, handlers):
    data_key, grp_key, pdg_key, label_key, weight_key = get_keys(flags)
    tsum = 0.
    handlers.data_io.next()
    handlers.data_io.next()
    while handlers.iteration < flags.ITERATION:
        epoch = handlers.iteration * float(flags.BATCH_SIZE) * float(flags.MINIBATCH_SIZE) / handlers.data_io.num_entries()
        tstamp_iteration = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
        tstart_iteration = time.time()

        checkpt_step = flags.CHECKPOINT_STEP and flags.WEIGHT_PREFIX and ((handlers.iteration+1) % flags.CHECKPOINT_STEP == 0)

        idx, blob = handlers.data_io.next()

        data_blob = {}
        data_blob['data'] = blob[data_key]
        data_blob['grp']  = blob[grp_key]
        data_blob['pdg']  = blob[pdg_key]
        if label_key is not None:
            data_blob['label'] = blob[label_key]
        if weight_key is not None:
            data_blob['weight'] = blob[weight_key]

        # Train step
        alpha = min(flags.ALPHA_LIMIT, 1 + (float(epoch) / flags.ALPHA_DECAY) * (flags.ALPHA_LIMIT-1.) + 1.)
        data_blob['alpha'] = alpha
        res = handlers.trainer.train_step(data_blob, epoch=float(epoch))

        # Save snapshot
        tspent_save = 0.
        if checkpt_step:
            handlers.trainer.save_state(handlers.iteration)

        log(handlers, tstart_iteration, tsum,
            data_blob['label'], res, alpha, flags,
            idx, epoch, tstamp_iteration)

        # Increment iteration counter
        handlers.iteration += 1

    # handlers.train_logger.close()
    if handlers.csv_logger:
        handlers.csv_logger.close()
    handlers.data_io.finalize()


def inference_loop(flags, handlers):
    data_key, grp_key, pdg_key, label_key, weight_key = get_keys(flags)
    tsum = 0.
    while handlers.iteration < flags.ITERATION:
        tstamp_iteration = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
        tstart_iteration = time.time()

        idx, blob = handlers.data_io.next()

        data_blob = {}
        data_blob['data'] = blob[data_key]
        if grp_key is not None:
            data_blob['grp'] = blob[grp_key]
        if pdg_key is not None:
            data_blob['pdg'] = blob[pdg_key]
        if label_key is not None:
            data_blob['label'] = blob[label_key]
        if weight_key is not None:
            data_blob['weight'] = blob[weight_key]

        # Run inference
        alpha = flags.ALPHA_LIMIT
        data_blob['alpha'] = alpha
        res = handlers.trainer.forward(data_blob)

        preds = res.get('predictions', None)
        scores = res.get('scores', None)
        segmentations = res.get('segmentation', None)
        segmentation_pred = res.get('segmentation_pred', None)
        if 'nunetC' in flags.MODEL_NAME:
            preds = res['predictions'][0]
        # Store output if requested
        if flags.OUTPUT_FILE:
            if grp_key is not None:
                #if grp_key is not None:
                grp_pred = nu_net.utils.DBScan(preds, 0.05)  # Kazu algorithm KILLME
                grp_pred2 = nu_net.utils.DBScan2(preds, 0.05)  # Scikit learn nu_net.utils.DBSCAN
                grp_pred3 = nu_net.utils.DBScan3(preds, scores)  # SGPN grouping
                groups = [(grp_pred[0].reshape([-1, 1]), 'prediction'),
                          (data_blob['grp'][0], 'grp'), (data_blob['pdg'][0], 'pdg'),
                          (grp_pred2[0].reshape([-1, 1]), 'prediction2'),
                          (grp_pred3[0].reshape([-1, 1]), 'prediction3'),
                          (scores[0].reshape([-1, 1]), 'final_scores'),
                          (segmentation_pred[0].reshape([-1, 1]).astype(np.float32), 'segmentation'),
                          (data_blob['label'][0].reshape([-1, 1]).astype(np.float32), 'segmentation_label')]
                for i, (tensor, name) in enumerate(res['tensors']):
                    groups.append((tensor[0, 0, :].reshape([-1, 1]), name))
                handlers.data_io.store_cluster(idx[0], groups)
                purity, efficiency, counts = nu_net.utils.compute_metrics(grp_pred[0].reshape([-1, 1]), data_blob['grp'][0])
                # print('\tScores: ', scores[0].mean(), scores[0].std())
                # print('\tPurity:', purity)
                # print('\tEfficiency:', efficiency)
                # print('\tCounts:', counts, '\n')
            if label_key is not None:
                acc, softmax = nu_net.utils.store_segmentation(handlers.data_io,
                                                              [idx],
                                                              data_blob['label'],
                                                              segmentations)

        epoch = handlers.iteration * float(flags.BATCH_SIZE) / handlers.data_io.num_entries()
        log(handlers, tstart_iteration, tsum, data_blob['label'], res,
            alpha, flags, idx, epoch, tstamp_iteration)
        handlers.iteration += 1

    if handlers.csv_logger:
        handlers.csv_logger.close()
    handlers.data_io.finalize()
