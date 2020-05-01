#!/usr/bin/env python
"""
Create a client which loads MNIST data and waits for jobs
to evaluate models. The rabbitmq service should be running
in 'localhost'.
"""

import os
import sys
import argparse

from gentun import GentunClient
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

parser = argparse.ArgumentParser("Neural Architecture Search Client")
parser.add_argument('-g', '--gpu', type=str, default="1", help="Select GPU Device")
parser.add_argument('-s', '--host', type=str, default="223.195.37.85", help="Select GPU Device")
parser.add_argument('-p', '--port', type=str, default="5672", help="Select GPU Device")
parser.add_argument('-u', '--user', type=str, default="test", help="Select GPU Device")
parser.add_argument('-k', '--password', type=str, default="test", help="Select GPU Device")
args = parser.parse_args()

if __name__ == '__main__':
    gc = GentunClient(args.gpu, host=args.host, port=int(args.port), user=args.user, password=args.password)
    gc.work()
