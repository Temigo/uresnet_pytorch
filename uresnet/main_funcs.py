from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import time
import datetime
import glob
import sys
import numpy as np
from uresnet.iotools import io_factory
from uresnet.trainval import trainval
import uresnet.utils as utils
import torch


def iotest(flags):
    # IO configuration
    io = io_factory(flags)
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
    if flags.FULL:
        full_inference_loop(flags, handlers)
    else:
        inference_loop(flags, handlers)


def prepare(flags):
    torch.cuda.set_device(flags.GPUS[0])
    handlers = Handlers()

    # IO configuration
    handlers.data_io = io_factory(flags)
    handlers.data_io.initialize()
    if 'sparse' in flags.IO_TYPE:
        handlers.data_io.start_threads()
        handlers.data_io.next()
    if 'sparse' in flags.MODEL_NAME and 'sparse' not in flags.IO_TYPE:
        sys.stderr.write('Sparse UResNet needs sparse IO.')
        sys.exit(1)

    # Trainer configuration
    flags.NUM_CHANNEL = handlers.data_io.num_channels()
    handlers.trainer = trainval(flags)

    # Restore weights if necessary
    handlers.iteration = 0
    loaded_iteration = 0
    if not flags.FULL:
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
        handlers.csv_logger = utils.CSVData(logname)
        if not flags.TRAIN and flags.FULL:
            handlers.metrics_logger = utils.CSVData('%s/metrics_log-%07d.csv' % (flags.LOG_DIR, loaded_iteration))
    return handlers


def get_keys(flags):
    data_key = flags.DATA_KEYS[0]
    label_key, weight_key = None, None
    if len(flags.DATA_KEYS) > 1:
        label_key = flags.DATA_KEYS[1]
    if len(flags.DATA_KEYS) > 2:
        label_key = flags.DATA_KEYS[2]
    return data_key, label_key, weight_key

def log(handlers, tstamp_iteration, tspent_iteration, tsum, res, flags, epoch):

    report_step  = flags.REPORT_STEP and ((handlers.iteration+1) % flags.REPORT_STEP == 0)

    loss_seg = np.mean(res['loss_seg'])
    acc_seg  = np.mean(res['accuracy'])

    mem = utils.round_decimals(torch.cuda.max_memory_allocated()/1.e9, 3)

    # Report (logger)
    if handlers.csv_logger:
        handlers.csv_logger.record(('iter', 'epoch', 'titer', 'tsumiter'),
                                   (handlers.iteration,epoch,tspent_iteration,tsum))
        handlers.csv_logger.record(('tio', 'tsumio'),
                                   (handlers.data_io.tspent_io,handlers.data_io.tspent_sum_io))
        handlers.csv_logger.record(('mem', ), (mem, ))
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
        loss_seg = utils.round_decimals(loss_seg,   4)
        tmap  = handlers.trainer.tspent
        tfrac = utils.round_decimals(tmap['train']/tspent_iteration*100., 2)
        tabs  = utils.round_decimals(tmap['train'], 3)
        epoch = utils.round_decimals(epoch, 2)

        if flags.TRAIN:
            msg = 'Iter. %d (epoch %g) @ %s ... train time %g%% (%g [s]) mem. %g GB \n'
            msg = msg % (handlers.iteration, epoch, tstamp_iteration, tfrac, tabs, mem)
        else:
            msg = 'Iter. %d (epoch %g) @ %s ... forward time %g%% (%g [s]) mem. %g GB \n'
            msg = msg % (handlers.iteration, epoch, tstamp_iteration, tfrac, tabs, mem)
        msg += '   Segmentation: loss %g accuracy %g\n' % (loss_seg, acc_seg)
        print(msg)
        sys.stdout.flush()
        if handlers.csv_logger: handlers.csv_logger.flush()
        if handlers.train_logger: handlers.train_logger.flush()


def get_data_minibatched(handlers, flags, data_key, label_key, weight_key):
    """
    Handles minibatching the data
    """
    data_blob = {'data': [], 'idx_v': []}
    if label_key  is not None: data_blob['label' ] = []
    if weight_key is not None: data_blob['weight'] = []

    for _ in range(int(flags.BATCH_SIZE / (flags.MINIBATCH_SIZE * len(flags.GPUS)))):
        idx, blob = handlers.data_io.next()
        data_blob['data'].append(blob[data_key])
        data_blob['idx_v'].append(idx)
        if label_key  is not None: data_blob['label' ].append(blob[label_key ])
        if weight_key is not None: data_blob['weight'].append(blob[weight_key])

    return data_blob


def train_loop(flags, handlers):
    data_key, label_key, weight_key = get_keys(flags)
    tsum = 0.
    handlers.data_io.next()
    handlers.data_io.next()
    while handlers.iteration < flags.ITERATION:
        epoch = handlers.iteration * float(flags.BATCH_SIZE) / handlers.data_io.num_entries()
        tstamp_iteration = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
        tstart_iteration = time.time()

        checkpt_step = flags.CHECKPOINT_STEP and flags.WEIGHT_PREFIX and ((handlers.iteration+1) % flags.CHECKPOINT_STEP == 0)

        data_blob = get_data_minibatched(handlers, flags, data_key, label_key, weight_key)

        # Train step
        res = handlers.trainer.train_step(data_blob, epoch=float(epoch),
                                          batch_size=flags.BATCH_SIZE)
        # Save snapshot
        if checkpt_step:
            handlers.trainer.save_state(handlers.iteration)

        tspent_iteration = time.time() - tstart_iteration
        tsum += tspent_iteration
        log(handlers, tstamp_iteration, tspent_iteration, tsum, res, flags, epoch)

        # Increment iteration counter
        handlers.iteration += 1

    # Finalize
    if handlers.csv_logger:
        handlers.csv_logger.close()
    handlers.data_io.finalize()


def inference_loop(flags, handlers):
    data_key, label_key, weight_key = get_keys(flags)
    tsum = 0.
    handlers.data_io.next()
    handlers.data_io.next()
    while handlers.iteration < flags.ITERATION:
        tstamp_iteration = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
        tstart_iteration = time.time()

        idx, blob = handlers.data_io.next()

        data_blob = {}
        data_blob['data'] = [blob[data_key]]
        if label_key is not None:
            data_blob['label'] = [blob[label_key]]
        if weight_key is not None:
            data_blob['weight'] = [blob[weight_key]]

        # Run inference
        res = handlers.trainer.forward(data_blob,
                                       batch_size=flags.BATCH_SIZE)
        # Store output if requested
        if flags.OUTPUT_FILE:
            handlers.data_io.store_segment(idx, blob[data_key], res['softmax'])

        epoch = handlers.iteration * float(flags.BATCH_SIZE) / handlers.data_io.num_entries()
        tspent_iteration = time.time() - tstart_iteration
        tsum += tspent_iteration
        log(handlers, tstamp_iteration, tspent_iteration, tsum, res, flags, epoch)
        handlers.iteration += 1

    # Finalize
    if handlers.csv_logger:
        handlers.csv_logger.close()
    handlers.data_io.finalize()


def full_inference_loop(flags, handlers):
    data_key, label_key, weight_key = get_keys(flags)
    tsum = 0.
    # Metrics for each event
    global_metrics = {
        'iteration': [],
        'acc': [],
        'correct_softmax': [],
        'id': [],
        'nonzero_pixels': [],
        'class_acc': [],
        'class_pixel': [],
        'class_mean_softmax': [],
        'cluster_acc': [],
        'class_cluster_acc': []
    }
    weights = glob.glob(flags.MODEL_PATH)
    print(weights)
    idx_v, blob_v = [], []
    for i in range(flags.ITERATION):
        idx, blob = handlers.data_io.next()
        idx_v.append(idx)
        blob_v.append(blob)

    for weight in weights:
        handlers.trainer._flags.MODEL_PATH = weight
        loaded_iteration   = handlers.trainer.initialize()
        handlers.iteration = 0
        while handlers.iteration < flags.ITERATION:
            tstamp_iteration = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
            tstart_iteration = time.time()

            # idx, blob = handlers.data_io.next()
            idx, blob = idx_v[handlers.iteration], blob_v[handlers.iteration]

            data_blob = {}
            data_blob['data'] = blob[data_key]
            if label_key is not None:
                data_blob['label'] = blob[label_key]
            if weight_key is not None:
                data_blob['weight'] = blob[weight_key]

            # Run inference
            res = handlers.trainer.forward(data_blob)

            # Store output if requested
            # if flags.OUTPUT_FILE:
            #     handlers.data_io.store_segment(idx,blob[data_key],res['softmax'])

            epoch = handlers.iteration * float(flags.BATCH_SIZE) / handlers.data_io.num_entries()
            tspent_iteration = time.time() - tstart_iteration
            tsum += tspent_iteration
            log(handlers, tstamp_iteration, tspent_iteration, tsum, res,
                flags, epoch)
            # Log metrics
            if label_key is not None:
                if flags.MODEL_NAME == 'uresnet_sparse':
                    metrics, dbscans = utils.compute_metrics_sparse(blob[data_key], blob[label_key], res['softmax'])
                else:
                    metrics = utils.compute_metrics_dense(blob[data_key], blob[label_key], res['softmax'])
                metrics['id'] = idx
                metrics['iteration'] = [loaded_iteration] * len(idx) * len(idx[0])
                for key in global_metrics:
                    global_metrics[key].extend(metrics[key])
                #ninety_quantile = utils.quantiles(blob[label_key], res['softmax'])

            # Store output if requested
            # Study low acc Michels
            # class_acc = np.array(metrics['class_acc'])
            # low_michel = class_acc[:, -1].mean() <= 0.6
            # michel_pixel = np.array(metrics['class_pixel'])[:, -1]
            # if flags.OUTPUT_FILE and low_michel and michel_pixel[0] > 200:
            #     print(idx, class_acc,np.array( metrics['class_cluster_acc'])[:, -1], michel_pixel)
            #     if label_key is None:
            #         handlers.data_io.store_segment(idx, blob[data_key], res['softmax'])
            #     else:
            #         handlers.data_io.store_segment(idx, blob[data_key], res['softmax'], clusters=dbscans)
            handlers.iteration += 1

    # Metrics
    if len(global_metrics['id']):
        global_metrics['id'] = np.hstack(global_metrics['id'])
    global_metrics['iteration'] = np.hstack(global_metrics['iteration'])
    # global_metrics['iteration'] = np.repeat(global_metrics['iteration'][:, None], global_metrics['id'].shape[1])
    for key in global_metrics:
        global_metrics[key] = np.array(global_metrics[key])

    res = {}
    # 90% quantile
    res['90q'] = np.percentile(global_metrics['acc'], 90)
    # 80% quantile
    res['80q'] = np.percentile(global_metrics['acc'], 80)
    # Mean accuracy
    res['mean_acc'] = np.mean(global_metrics['acc'])
    # Median accuracy
    res['50q'] = np.median(global_metrics['acc'])
    # Mean accuracy of the worst 5%
    worst5_index = global_metrics['acc'] <= np.percentile(global_metrics['acc'], 5)
    res['worst5'] = np.mean(global_metrics['acc'][worst5_index])
    # Mean softmax score for the correct prediction
    for key in global_metrics:
        res[key] = global_metrics[key]
    print(res)
    for i, idx in enumerate(res['id']):
        handlers.metrics_logger.record(('iteration', 'id', 'correct_softmax', 'acc', 'nonzero_pixels', 'cluster_acc'),
                (res['iteration'][i], idx, res['correct_softmax'][i], res['acc'][i], res['nonzero_pixels'][i], res['cluster_acc'][i]))
        handlers.metrics_logger.record(['class_%d_acc' % c for c in range(flags.NUM_CLASS)], [res['class_acc'][i][c] for c in range(flags.NUM_CLASS)])
        handlers.metrics_logger.record(['class_%d_cluster_acc' % c for c in range(flags.NUM_CLASS)], [res['class_cluster_acc'][i][c] for c in range(flags.NUM_CLASS)])
        handlers.metrics_logger.record(['class_%d_pixel' % c for c in range(flags.NUM_CLASS)], [res['class_pixel'][i][c] for c in range(flags.NUM_CLASS)])
        handlers.metrics_logger.record(['class_%d_mean_softmax' % c for c in range(flags.NUM_CLASS)], [res['class_mean_softmax'][i][c] for c in range(flags.NUM_CLASS)])
        handlers.metrics_logger.write()
    # Finalize
    if handlers.csv_logger:
        handlers.csv_logger.close()
    if handlers.metrics_logger:
        handlers.metrics_logger.close()
    handlers.data_io.finalize()
