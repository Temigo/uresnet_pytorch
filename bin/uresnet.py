#!/usr/bin/python
import os
import sys
URESNET_DIR = os.path.dirname(os.path.abspath(__file__))
URESNET_DIR = os.path.dirname(URESNET_DIR)
sys.path.insert(0, URESNET_DIR)
from uresnet.flags import URESNET_FLAGS

def main():
    flags = URESNET_FLAGS()
    flags.parse_args()

if __name__ == '__main__':
    main()
