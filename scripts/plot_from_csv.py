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
failcodes = df.exit_code.str.extract(r'0x(?P<naive_mismatch>[0-9a-fA-F]{4})(?P<accel_mismatch>[0-9a-fA-F]{4})').map(lambda x:int(x,16))
df = pd.concat([df,failcodes],axis=1)
# df = df.groupby('name').sample(n=10000, replace=False)

print(df.head())
print(df.describe())

# calculate interesting statistics per sample
# % speedup
df['normalized cycles'] = df['accel-system.cpu.numCycles'] / df['naive-system.cpu.numCycles']
df['normalized insts'] = df['accel-simInsts'] / df['naive-simInsts']
df['percent change from naive'] = (df['accel-system.cpu.numCycles'] - df['naive-system.cpu.numCycles']) / df['naive-system.cpu.numCycles'] * 100

# stats per record
df['accel cycles per record'] = df['accel-system.cpu.numCycles'] / df['record_len']
df['naive cycles per record'] = df['naive-system.cpu.numCycles'] / df['record_len']
df['accel insts per record'] = df['accel-simInsts'] / df['record_len']
df['naive insts per record'] = df['naive-simInsts'] / df['record_len']

# Mismatch data
df['failure delta per record'] = (df['accel_mismatch'] - df['naive_mismatch']) / df['record_len']
df['accel failures per record'] = df['accel_mismatch'] / df['record_len']
df['naive failures per record'] = df['naive_mismatch'] / df['record_len']

# Power data
# for unopt, we need switching adder, switching divider
# for opt, we need total comparator(*ncycles), switching adder, switching divider
# we can see the expected power ratios of both tho
addSwitchPwr = 2.628e-04
divSwitchPwr = 8.760e-05
comparatorTotalPwr = 78.5908 * 1e-6
df['accel adds'] = df['accel-sameAdd'] + df['accel-switchingAdd']
df['accel divs'] = df['accel-sameDiv'] + df['accel-switchingDiv']
df['accel add switching power estimate'] = df['accel-switchingAdd'] * addSwitchPwr
df['accel div switching power estimate'] = df['accel-switchingDiv'] * divSwitchPwr
df['accel TOTAL comparator power estimate'] = df['accel-system.cpu.numCycles'] * comparatorTotalPwr 

# plot the distributions of record_len and mismatches
os.makedirs(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../figures'), exist_ok=True)
print('generating record distribution')
sns.displot(data=df, x='record_len', hue='name', common_bins=True, bins=25, multiple='stack').figure.savefig('figures/record_distribution.png')
plt.clf()

print('generating power plots')
sums = df.groupby('name').sum()
sums['total switching power estimate'] = sums['accel add switching power estimate'] + sums['accel div switching power estimate']
sums['accel add switching pwr %'] = sums['accel add switching power estimate'] / sums['total switching power estimate'] * 100
sums['accel div switching pwr %'] = sums['accel div switching power estimate'] / sums['total switching power estimate'] * 100
sums[['accel add switching pwr %', 'accel div switching pwr %']].plot(kind='bar', stacked='True')
print(sums)
plt.gcf().tight_layout()
plt.gcf().savefig('figures/power_rto.png')

print('generating normalized plots')
sns.displot(data=df, x='normalized cycles', hue='name', common_bins=True, bins=25, multiple='stack').figure.savefig('figures/normalized_cycles.png')
plt.clf()
sns.displot(data=df, x='normalized insts', hue='name', common_bins=True, bins=25, multiple='stack').figure.savefig('figures/normalized_insts.png')
plt.clf()
sns.relplot(data=df, x='record_len', y='normalized cycles', hue='name').figure.savefig('figures/normalized_cycles_rel.png')
plt.clf()

print('generating performance per record plots')
sns.displot(data=df, x='accel cycles per record', hue='name', common_bins=True, bins=25, multiple='stack').figure.savefig('figures/accel_cycles_per_record.png')
plt.clf()
sns.displot(data=df, x='naive cycles per record', hue='name', common_bins=True, bins=25, multiple='stack').figure.savefig('figures/naive_cycles_per_record.png')
plt.clf()
sns.displot(data=df, x='accel insts per record', hue='name', common_bins=True, bins=25, multiple='stack').figure.savefig('figures/accel_insts_per_record.png')
plt.clf()
sns.displot(data=df, x='naive insts per record', hue='name', common_bins=True, bins=25, multiple='stack').figure.savefig('figures/naive_insts_per_record.png')
plt.clf()

print('generating failure per record plots')
sns.displot(data=df, x='failure delta per record', hue='name', common_bins=True, bins=25, multiple='stack').figure.savefig('figures/accel_minus_naive_mismatches.png')
plt.clf()
sns.displot(data=df, x='accel failures per record', hue='name', common_bins=True, bins=25, multiple='stack').figure.savefig('figures/accel_mismatches.png')
plt.clf()
sns.displot(data=df, x='naive failures per record', hue='name', common_bins=True, bins=25, multiple='stack').figure.savefig('figures/naive_mismatches.png')
plt.clf()

print('generating speedup plots')
sns.barplot(data=df, y='name', x='percent change from naive', orient='h').figure.savefig('figures/pct_cycles.png')
plt.clf()
sns.barplot(data=df, y='name', x='normalized cycles', orient='h').figure.savefig('figures/speedup.png')
plt.clf()

print('melting df')
df = df.drop(['record', 'exit_code'], axis=1)
melt = pd.melt(df, id_vars=['name', 'record_len'], value_vars=['accel cycles per record','naive cycles per record'], value_name='Cycles per Record', ignore_index=True)
sns.barplot(data=melt, x="name", y="Cycles per Record", hue='variable').figure.savefig('figures/melted_catplot.png')
plt.clf()

# Plots that vary the figure size
print('generating raw mismatches, not sharing axis')
f = sns.relplot(data=df, kind='scatter', x='naive_mismatch', y='accel_mismatch', col='name', hue='name', facet_kws=dict(sharex=False, sharey=False))
f.figure.tight_layout()
f.figure.savefig('figures/raw_mismatches.png')
plt.clf()

print('generating pairplot')
sns.pairplot(data=df, hue='name', vars=['accel-system.cpu.numCycles','naive-system.cpu.numCycles','record_len','accel_mismatch','naive_mismatch']).figure.savefig('figures/pairplot.png')
plt.clf()

