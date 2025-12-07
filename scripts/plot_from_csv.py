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
failcodes = df.exit_code.str.extract(r'0x(?P<accel_mismatch>[0-9a-fA-F]{4})(?P<naive_mismatch>[0-9a-fA-F]{4})').map(lambda x:int(x,16))
df = pd.concat([df,failcodes],axis=1)
print(df.head())
print(df.describe())

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
df['failure delta per record'] = (df['accel_mismatch'] - df['naive_mismatch']) / df['record_len']
df['accel failures per record'] = df['accel_mismatch'] / df['record_len']
df['naive failures per record'] = df['naive_mismatch'] / df['record_len']

# plot the distributions of record_len and mismatches
os.makedirs(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../figures'), exist_ok=True)
print('generating record distribution')
sns.displot(data=df, x='record_len', hue='name', multiple='dodge', common_bins=True).figure.savefig('figures/record_distribution.png')

print('generating normalized plots')
sns.displot(data=df, x='normalized cycles', hue='name', common_bins=True).figure.savefig('figures/normalized_cycles.png')
sns.displot(data=df, x='normalized insts', hue='name', common_bins=True).figure.savefig('figures/normalized_insts.png')
sns.relplot(data=df, x='record_len', y='normalized cycles', hue='name').figure.savefig('figures/normalized_cycles_rel.png')

print('generating performance per record plots')
sns.displot(data=df, x='accel cycles per record', hue='name', common_bins=True).figure.savefig('figures/accel_cycles_per_record.png')
sns.displot(data=df, x='naive cycles per record', hue='name', common_bins=True).figure.savefig('figures/naive_cycles_per_record.png')
sns.displot(data=df, x='accel insts per record', hue='name', common_bins=True).figure.savefig('figures/accel_insts_per_record.png')
sns.displot(data=df, x='naive insts per record', hue='name', common_bins=True).figure.savefig('figures/naive_insts_per_record.png')

print('generating failure per record plots')
sns.displot(data=df, x='failure delta per record', hue='name', multiple='dodge', common_bins=True).figure.savefig('figures/accel_mismatches.png')
sns.displot(data=df, x='accel failures per record', hue='name', multiple='dodge', common_bins=True).figure.savefig('figures/accel_mismatches.png')
sns.displot(data=df, x='naive failures per record', hue='name', multiple='dodge', common_bins=True).figure.savefig('figures/naive_mismatches.png')

print('generating pairplot')
sns.pairplot(data=df, hue='name', vars=['accel-system.cpu.numCycles','naive-system.cpu.numCycles','record_len','accel_mismatch','naive_mismatch']).figure.savefig('figures/pairplot.png')
