import numpy as np
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
from data.dataset import load_dataset

# os settings
sys.path.append(os.getcwd() + '/..')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Parse cmdline args
parser = argparse.ArgumentParser(description='Few-Shot-Learning')
parser.add_argument('--logdir', default='./logs/temp', type=str)

parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--exp_id', default='dae.exp_syn1', type=str)
parser.add_argument('--gpu', default=-1, type=int)
parser.add_argument('--model', default='dae', type=str)

args = parser.parse_args()

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
tf.random.set_random_seed(args.seed)

# fetch model
if args.model == 'dae':
    agent = DAE(params)
elif args.model == 'protonet':
    agent = SupportQueryAgent(params, )

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
agent.start()
done = False
for epoch in range(params['train']['num_epoches']):
    if params['train']['valid_interval'] is not None:
        if epoch % params['train']['valid_interval'] == 0:
            agent.eval(epoch, valid)
    for iters in tqdm(range(params['train']['iter_per_epoch'])):
        done = agent.train_iter(train)
        if done:
            break
    agent.print_log(epoch)
    #agent.visualize2d('logs/syn2d', train, epoch, color_set)
    agent.take_step()
    if done:
        break

