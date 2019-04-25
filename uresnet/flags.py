from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import numpy as np
import argparse
import os
from uresnet.main_funcs import train, iotest, inference
from distutils.util import strtobool


class URESNET_FLAGS:

    # flags for model
    NUM_CLASS  = 2
    MODEL_NAME = ""
    TRAIN      = True
    DEBUG      = False
    FULL = False

    # Flags for Sparse UResNet model
    URESNET_NUM_STRIDES = 3
    URESNET_FILTERS = 16
    SPATIAL_SIZE = 192
    BN_MOMENTUM = 0.9

    # flags for train/inference
    COMPUTE_WEIGHT = False
    SEED           = -1
    LEARNING_RATE  = 0.001
    GPUS           = []
    WEIGHT_PREFIX  = ''
    NUM_POINT      = 2048
    NUM_CHANNEL    = -1
    ITERATION      = 10000
    REPORT_STEP    = 100
    CHECKPOINT_STEP  = 500

    # flags for IO
    IO_TYPE    = ''
    INPUT_FILE = ''
    OUTPUT_FILE = ''
    MINIBATCH_SIZE = -1
    BATCH_SIZE = -1
    LOG_DIR    = ''
    MODEL_PATH = ''
    DATA_KEYS  = ''
    SHUFFLE    = 1
    LIMIT_NUM_SAMPLE = -1
    NUM_THREADS = 1
    DATA_DIM = 3
    PARTICLE = False

    def __init__(self):
        self._build_parsers()

    def _attach_common_args(self,parser):
        parser.add_argument('-db','--debug',type=strtobool,default=self.DEBUG,
                            help='Extra verbose mode for debugging [default: %s]' % self.DEBUG)
        parser.add_argument('-ld','--log_dir', default=self.LOG_DIR,
                            help='Log dir [default: %s]' % self.LOG_DIR)
        parser.add_argument('-sh','--shuffle',type=strtobool,default=self.SHUFFLE,
                            help='Shuffle the data entries [default: %s]' % self.SHUFFLE)
        parser.add_argument('--gpus', type=str, default='',
                            help='GPUs to utilize (comma-separated integers')
        parser.add_argument('-nc','--num_class', type=int, default=self.NUM_CLASS,
                            help='Number of classes [default: %s]' % self.NUM_CLASS)
        parser.add_argument('-it','--iteration', type=int, default=self.ITERATION,
                            help='Iteration to run [default: %s]' % self.ITERATION)
        parser.add_argument('-bs','--batch_size', type=int, default=self.BATCH_SIZE,
                            help='Batch size during training for updating weights [default: %s]' % self.BATCH_SIZE)
        parser.add_argument('-mbs','--minibatch_size', type=int, default=self.MINIBATCH_SIZE,
                            help='Mini-batch size (sample/gpu) during training for updating weights [default: %s]' % self.MINIBATCH_SIZE)
        parser.add_argument('-rs','--report_step', type=int, default=self.REPORT_STEP,
                            help='Period (in steps) to print out loss and accuracy [default: %s]' % self.REPORT_STEP)
        parser.add_argument('-mn','--model_name', type=str, default=self.MODEL_NAME,
                            help='model name identifier [default: %s]' % self.MODEL_NAME)
        parser.add_argument('-mp','--model_path', type=str, default=self.MODEL_PATH,
                            help='model checkpoint file path [default: %s]' % self.MODEL_PATH)
        parser.add_argument('-io','--io_type',type=str,default=self.IO_TYPE,
                            help='IO handler type [default: %s]' % self.IO_TYPE)
        parser.add_argument('-if','--input_file',type=str,default=self.INPUT_FILE,
                            help='comma-separated input file list [default: %s]' % self.INPUT_FILE)
        parser.add_argument('-of','--output_file',type=str,default=self.OUTPUT_FILE,
                            help='output file name [default: %s]' % self.OUTPUT_FILE)
        parser.add_argument('-dkeys','--data_keys',type=str,default=self.DATA_KEYS,
                            help='A keyword to fetch data from file [default: %s]' % self.DATA_KEYS)
        parser.add_argument('-lns','--limit_num_sample',type=int,default=self.LIMIT_NUM_SAMPLE,
                            help='Limit number of samples to read from input file [default: %s]' % self.LIMIT_NUM_SAMPLE)
        parser.add_argument('-nt','--num-threads',type=int,default=self.NUM_THREADS,
                            help='Number of threads to read input file [default: %s]' % self.NUM_THREADS)
        parser.add_argument('-dd','--data-dim',type=int,default=self.DATA_DIM,
                            help='Data dimension [default: %s]' % self.DATA_DIM)
        parser.add_argument('-ss','--spatial_size',type=int,default=self.SPATIAL_SIZE,
                            help='Length of one side of the cubical data (2d/3d) [default: %s]' % self.SPATIAL_SIZE)
        parser.add_argument('-uns','--uresnet-num-strides',type=int,default=self.URESNET_NUM_STRIDES,
                            help='Depth for UResNet [default: %s]' % self.URESNET_NUM_STRIDES)
        parser.add_argument('-uf','--uresnet-filters',type=int,default=self.URESNET_FILTERS,
                            help='Number of base filters for UResNet [default: %s]' % self.URESNET_FILTERS)
        parser.add_argument('-bnm','--bn-momentum',type=float,default=self.BN_MOMENTUM,
                            help='BatchNorm Momentum for UResNet [default: %s]' % self.BN_MOMENTUM)
        parser.add_argument('-cw','--compute_weight',default=self.COMPUTE_WEIGHT, action='store_true',
                            help='Compute pixel loss weighting factor on the fly [default: %s' % self.COMPUTE_WEIGHT)
        parser.add_argument('-sd','--seed', default=self.SEED,
                                  help='Seed for random number generators [default: %s]' % self.SEED)
        return parser

    def _build_parsers(self):

        self.parser = argparse.ArgumentParser(description="Edge-GCNN Configuration Flags")
        subparsers = self.parser.add_subparsers(title="Modules", description="Valid subcommands", dest='script', help="aho")

        # train parser
        train_parser = subparsers.add_parser("train", help="Train Edge-GCNN")
        train_parser.add_argument('-wp','--weight_prefix', default=self.WEIGHT_PREFIX,
                                  help='Prefix (directory + file prefix) for snapshots of weights [default: %s]' % self.WEIGHT_PREFIX)
        train_parser.add_argument('-lr','--learning_rate', type=float, default=self.LEARNING_RATE,
                                  help='Initial learning rate [default: %s]' % self.LEARNING_RATE)
        train_parser.add_argument('-chks','--checkpoint_step', type=int, default=self.CHECKPOINT_STEP,
                                  help='Period (in steps) to store snapshot of weights [default: %s]' % self.CHECKPOINT_STEP)

        # inference parser
        inference_parser = subparsers.add_parser("inference",help="Run inference of Edge-GCNN")
        inference_parser.add_argument('-full', '--full', default=self.FULL, action='store_true',
                                      help='Full inference mode [default: %s]' % self.FULL)
        inference_parser.add_argument('-p', '--particle', default=self.PARTICLE, action='store_true',
                                      help='Include particle branch [default: %s]' % self.PARTICLE)
        # IO test parser
        iotest_parser = subparsers.add_parser("iotest", help="Test iotools for Edge-GCNN")

        # attach common parsers
        self.train_parser     = self._attach_common_args(train_parser)
        self.inference_parser = self._attach_common_args(inference_parser)
        self.iotest_parser    = self._attach_common_args(iotest_parser)

        # attach executables
        self.train_parser.set_defaults(func=train)
        self.inference_parser.set_defaults(func=inference)
        self.iotest_parser.set_defaults(func=iotest)

    def parse_args(self):
        args = self.parser.parse_args()
        self.update(vars(args))
        print("\n\n-- CONFIG --")
        for name in vars(self):
            attribute = getattr(self,name)
            if type(attribute) == type(self.parser): continue
            print("%s = %r" % (name, getattr(self, name)))

        # Set random seed for reproducibility
        np.random.seed(self.SEED)
        torch.manual_seed(self.SEED)
        args.func(self)

    def update(self, args):
        for name,value in args.items():
            if name in ['func','script']: continue
            setattr(self, name.upper(), args[name])
        # os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = self.GPUS
        if len(self.GPUS) > 0:
            self.GPUS = list(range(len(self.GPUS.split(','))))
        # self.GPUS = [int(gpu) for gpu in self.GPUS.split(',')]
        self.INPUT_FILE=[str(f) for f in self.INPUT_FILE.split(',')]
        self.DATA_KEYS=self.DATA_KEYS.split(',')
        if self.SEED < 0:
            import time
            self.SEED = int(time.time())
        else:
            self.SEED = int(self.SEED)
        # Batch size checker
        if self.BATCH_SIZE < 0 and self.MINIBATCH_SIZE < 0:
            print('Cannot have both BATCH_SIZE (-bs) and MINIBATCH_SIZE (-mbs) negative values!')
            raise ValueError
        # Assign non-default values
        if self.BATCH_SIZE < 0:
            self.BATCH_SIZE = int(self.MINIBATCH_SIZE * max(1, len(self.GPUS)))
        if self.MINIBATCH_SIZE < 0:
            self.MINIBATCH_SIZE = int(self.BATCH_SIZE / max(1, len(self.GPUS)))
        # Check consistency
        if not (self.BATCH_SIZE % (self.MINIBATCH_SIZE * max(1, len(self.GPUS)))) == 0:
            print('BATCH_SIZE (-bs) must be multiples of MINIBATCH_SIZE (-mbs) and GPU count (--gpus)!')
            raise ValueError
        # Check compute_weight option
                # Compute weights if specified
        if self.COMPUTE_WEIGHT:
            if len(self.DATA_KEYS)>2:
                sys.stderr.write('ERROR: cannot compute weight if producer is specified ("%s")\n' % self.DATA_KEYS[2])
                raise KeyError
            if '_weights_' in self.DATA_KEYS:
                sys.stderr.write('ERROR: cannot compute weight if any data has label "_weights_"\n')
                raise KeyError
            if len(self.DATA_KEYS) < 2:
                sys.stderr.write('ERROR: you must provide data and label (2 data product keys) to compute weights\n')
                raise KeyError
            self.DATA_KEYS.append('_weights_')

if __name__ == '__main__':
    flags = URESNET_FLAGS()
    flags.parse_args()
