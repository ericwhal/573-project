#!/usr/bin/env python3
import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

filepath = os.path.abspath(sys.argv[1])

# Read in the data, dropping rows with empty data
df = pd.read_csv(filepath).dropna()
failcodes = df.exit_code.str.extract(r'0x(?P<accel-mismatch>[0-9a-fA-F]{4})(?P<naive-mismatch>[0-9a-fA-F]{4})').map(lambda x:int(x,16))
df = pd.concat([df,failcodes])

# calculate interesting statistics per sample
# % speedup
df['normalized cycles'] = df['accel-system.cpu.numCycles'] / df['naive-system.cpu.numCycles']
df['normalized insts'] = df['accel-simInsts'] / df['naive-simInsts']

# stats per record
df['accel cycles per record'] = df['accel-system.cpu.numCycles'] / df['record_len']
df['naive cycles per record'] = df['naive-system.cpu.numCycles'] / df['record_len']
df['accel insts per record'] = df['accel-simInsts'] / df['record_len']
df['naive insts per record'] = df['naive-simInsts'] / df['record_len']

# Mismatch data
df['failure delta per record'] = (df['accel-mismatch'] - df['naive-mismatch']) / df['record_len']
df['accel failures per record'] = df['accel-mismatch'] / df['record_len']
df['naive failures per record'] = df['naive-mismatch'] / df['record_len']

# plot the distributions of record_len and mismatches
os.makedirs(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../figures'), exist_ok=True)
sns.displot(data=df, x='record_len', hue='name', multiple='dodge').figure.savefig('figures/record_distribution.png')

sns.displot(data=df, x='normalized cycles', hue='name').figure.savefig('figures/normalized_cycles.png')
sns.displot(data=df, x='normalized insts', hue='name').figure.savefig('figures/normalized_insts.png')

sns.displot(data=df, x='accel cycles per record', hue='name').figure.savefig('figures/accel_cycles_per_record.png')
sns.displot(data=df, x='naive cycles per record', hue='name').figure.savefig('figures/naive_cycles_per_record.png')
sns.displot(data=df, x='accel insts per record', hue='name').figure.savefig('figures/accel_insts_per_record.png')
sns.displot(data=df, x='naive insts per record', hue='name').figure.savefig('figures/naive_insts_per_record.png')

sns.displot(data=df, x='failure delta per record', hue='name', multiple='dodge').figure.savefig('figures/accel-mismatches.png')
sns.displot(data=df, x='accel failures per record', hue='name', multiple='dodge').figure.savefig('figures/accel-mismatches.png')
sns.displot(data=df, x='naive failures per record', hue='name', multiple='dodge').figure.savefig('figures/naive-mismatches.png')
