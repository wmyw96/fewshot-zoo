import numpy as np
import datetime
import tensorflow as tf
import json, sys, os
from os import path
import time
import shutil
import matplotlib
import importlib
import argparse
from tqdm import tqdm
from utils import *
from agents.dae_agent import DAE
from agents.supportquery_agent import SupportQueryAgent
from agents.dve_agent import DVE
from data.dataset import load_dataset
from agents.utils import *

# os settings
sys.path.append(os.getcwd() + '/..')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Parse cmdline args
parser = argparse.ArgumentParser(description='Few-Shot-Learning')
parser.add_argument('--logdir', default='../../data/fewshot-zoo-logs/', type=str)

parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--exp_id', default='dae.exp_syn1', type=str)
parser.add_argument('--gpu', default=-1, type=int)
parser.add_argument('--model', default='dae', type=str)
parser.add_argument('--pretrain_dir', default='', type=str)
parser.add_argument('--type', default='train', type=str)
parser.add_argument('--stat', default=0, type=int)

args = parser.parse_args()

if len(args.pretrain_dir) < 2:
    args.pretrain_dir = None

# GPU settings
if args.gpu > -1:
    print("GPU COMPATIBLE RUN...")
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

# Print experiment details
print('Booting with exp params: {}...'.format(args.exp_id))
time.sleep(2)

# Experiment parameters
mod = importlib.import_module('saved_params.' + args.exp_id)
params = mod.generate_params()

# set seed
params['train']['seed'] = args.seed
np.random.seed(args.seed)
tf.set_random_seed(args.seed)

log_dir = os.path.join(args.logdir, args.exp_id + datetime.datetime.now().strftime('%m_%d_%H_%M'))
print('Experiment Logs will be written at {}'.format(log_dir))
logger = LogWriter(log_dir, 'main.log')

# fetch model
if args.model == 'dae':
    agent = DAE(params, logger, args.gpu)
elif args.model == 'protonet':
    agent = SupportQueryAgent(params, logger, args.gpu)
elif args.model == 'dve':
    agent = DVE(params, logger, args.gpu)
else:
    raise NotImplementedError

# get dataset

dataset = load_dataset(params)

valid, test = None, None

if len(params['data']['split']) == 1:
    train = dataset['train']
elif len(params['data']['split']) == 2:
    train, test = dataset['train'], dataset['test']
elif len(params['data']['split']) == 3:
    train, valid, test = dataset['train'], dataset['valid'], dataset['test']

# start training
if args.type == 'train':
    if args.pretrain_dir is not None:
        agent.start(args.pretrain_dir, train)
    else:
        agent.start()
    done = False
    for epoch in range(params['train']['num_epoches']):
        if params['train']['valid_interval'] is not None:
            if epoch % params['train']['valid_interval'] == 0:
                #agent.evallll(valid)
                agent.eval(epoch, valid, test)
        #if args.stat:
        #    agent.get_statistics(epoch, 'train', train, color_set)
        #    agent.get_statistics(epoch, 'val', valid, color_set)
        #    agent.get_statistics(epoch, 'test', test, color_set)

        for iters in tqdm(range(params['train']['iter_per_epoch'])):
            done = agent.train_iter(train)
            if done:
                break
        agent.print_log(epoch)
        #agent.visualize2d('logs/syn2d', train, epoch, color_set)
        agent.take_step()
        if args.stat:
            agent.get_statistics(epoch, 'train', train, color_set)
            agent.get_statistics(epoch, 'val', valid, color_set)
            agent.get_statistics(epoch, 'test', test, color_set)

        if done:
            break
elif args.type == 'pretrain':
    agent.pretrain(train, args.pretrain_dir)
