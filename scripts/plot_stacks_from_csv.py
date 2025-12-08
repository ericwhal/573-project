#!/usr/bin/env python3
import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

opti_filepath = os.path.abspath(sys.argv[1])
nopt_filepath = os.path.abspath(sys.argv[2])
naiv_filepath = os.path.abspath(sys.argv[3])

optimized = pd.read_csv(opti_filepath).sort_values(['name','record'],inplace=False).reset_index()
unoptimized = pd.read_csv(nopt_filepath).sort_values(['name','record'],inplace=False).reset_index()
naive = pd.read_csv(naiv_filepath).sort_values(['name','record'],inplace=False).reset_index()

# plot things that need naive
aggregate = pd.concat([optimized, unoptimized, naive], axis=0)
sums = aggregate.groupby(['name','accel']).sum()
sums['cycles per element'] = sums['system.cpu.numCycles'] / sums.groupby('name')['system.cpu.numCycles'].max()
sums = sums.reindex(index=['cifar10_vitb', 'cifar100_vitb', 'imagenet1k_vitb', 'wt103_gpt2'], level=0)
sums.unstack(-1).plot(kind='bar',y='cycles per element', rot=0)
plt.gca().set_ylabel('Speedup')
plt.gcf().tight_layout()
plt.gcf().savefig('figures/cyclecnt.png')
plt.clf()

sums['cycles'] = sums['system.cpu.numCycles']
sums.unstack(-1).plot(kind='bar',y='cycles', rot=0)
plt.semilogy()
plt.gca().set_ylabel('Cycles')
plt.gcf().tight_layout()
plt.gcf().savefig('figures/raw_cyclecnt.png')
plt.clf()


# plot things that don't need naive
addSwitchPwr = 2.628e-04
divSwitchPwr = 8.760e-05
comparatorTotalPwr = 78.5908 * 1e-6
optimized['power'] = optimized['record_len']*comparatorTotalPwr + optimized['switchingAdd']*addSwitchPwr + optimized['switchingDiv']*divSwitchPwr
optimized['comparator power']        = optimized['record_len']*comparatorTotalPwr
optimized['adder switching power']   = optimized['switchingAdd']*addSwitchPwr
optimized['divider switching power'] = optimized['switchingDiv']*divSwitchPwr
unoptimized['power'] = unoptimized['switchingAdd']*addSwitchPwr + unoptimized['switchingDiv']*divSwitchPwr
unoptimized['comparator power']        = 0
unoptimized['adder switching power']   = unoptimized['switchingAdd']*addSwitchPwr
unoptimized['divider switching power'] = unoptimized['switchingDiv']*divSwitchPwr

aggregate = pd.concat([optimized, unoptimized], axis=0)
aggregate['mismatch-unnorm'] = aggregate['exit_code'].map(lambda x: int(x, 16))
aggregate['mismatch'] = aggregate['mismatch-unnorm'] / aggregate['record_len']
sns.boxplot(data=aggregate, x='name', hue='accel', y='mismatch').figure.savefig('figures/boxplot.png')

sums = aggregate.drop('exit_code',axis=1).groupby(['name','accel']).sum()
sums['adder switching power']   /= sums['record_len'] * (aggregate['power'] / aggregate['record_len']).max()
sums['divider switching power'] /= sums['record_len'] * (aggregate['power'] / aggregate['record_len']).max()
sums['comparator power']        /= sums['record_len'] * (aggregate['power'] / aggregate['record_len']).max()

sums['adder switching']   = sums['switchingAdd'] / (sums['record_len'] * ((aggregate['switchingAdd'] + aggregate['switchingDiv'])/aggregate['record_len']).max())
sums['divider switching'] = sums['switchingDiv'] / (sums['record_len'] * ((aggregate['switchingAdd'] + aggregate['switchingDiv'])/aggregate['record_len']).max())

# sums.set_index(['name', 'accel'], inplace=True)
df0 = sums.reorder_levels(['name', 'accel']).reindex(index=['cifar10_vitb', 'cifar100_vitb', 'imagenet1k_vitb', 'wt103_gpt2'], level=0)
df0 = df0.unstack(level=-1)

colors = plt.cm.Paired.colors
fig, ax = plt.subplots()
(df0['adder switching power'] + df0['divider switching power'] + df0['comparator power']).plot(kind='bar', color=[colors[3], colors[2]], ax=ax, rot=0)
(df0['adder switching power'] + df0['divider switching power']).plot(kind='bar', color=[colors[1], colors[0]], ax=ax, rot=0)
(df0['adder switching power']).plot(kind='bar', color=[colors[5], colors[4]], ax=ax, rot=0)
handles, _ = ax.get_legend_handles_labels()
labels =  ['optimized comparator', None] + ['optimized divider', 'unoptimized divider'] + ['optimized adder', 'unoptimized adder']
plt.legend([h for h, l in zip(handles, labels) if l is not None], [l for l in labels if l is not None])
ax.set_xlabel('Dataset')
ax.set_ylabel('Power per Element')
plt.tight_layout()
fig.savefig('figures/paired_pwrplot.png')
plt.clf()

fig, ax = plt.subplots()
(df0['adder switching'] + df0['divider switching']).plot(kind='bar', color=[colors[3], colors[2]], ax=ax, rot=0)
(df0['adder switching']).plot(kind='bar', color=[colors[5], colors[4]], ax=ax, rot=0)
handles, _ = ax.get_legend_handles_labels()
labels =  ['optimized divider', 'unoptimized divider'] + ['optimized adder', 'unoptimized adder']
plt.legend(labels)
ax.set_xlabel('Dataset')
ax.set_ylabel('Switching per Element')
plt.tight_layout()
fig.savefig('figures/paired_switching_plot.png')


plt.clf()
g = sns.catplot(optimized, x='name', y='zeroCount', hue='accel')
sns.move_legend(g, 'center')
plt.tight_layout()
g.figure.savefig('figures/zeros_vs_record')
