import os
import re
import sys

import m5
from m5.objects import *
from m5.objects import (
    BasicPioDevice,
    MemCpyAccel,
)

# Return a dict of ordered lists of stats, in the encountered order
number_regex = re.compile(r'\d*\.?\d+')
key_order = ['system.cpu.numCycles','simInsts','simOps'
            ,'system.memcpy_accel.zeroCount'
            ,'system.memcpy_accel.sameAdd'
            ,'system.memcpy_accel.switchingAdd'
            ,'system.memcpy_accel.sameDiv'
            ,'system.memcpy_accel.switchingDiv'
            ]
def parse_stats(file):
    stats = {key:[] for key in key_order}
    with open(file, 'r') as f:
        for line in f.readlines():
            for key in stats.keys():
                if key in line:
                    stats[key].append(number_regex.search(line).group())
    return stats

# File stuff
thispath = os.getcwd()
m5outdir = os.path.join(thispath, "m5out/")
m5statsfile = os.path.join(m5outdir, "stats.txt")
reportfile = os.path.join(thispath, "stats.csv")
binary = os.path.join(thispath, "eval")

# --- Workload ---
system = System()

system.clk_domain = SrcClockDomain()
system.clk_domain.clock = "1GHz"
system.clk_domain.voltage_domain = VoltageDomain()

system.mem_mode = "timing"
# system.mem_ranges = [
#     AddrRange((0x40000000,0x60000000)),   # normal DRAM
#     AddrRange((0x60000000,0x60000020))      # MemCpyAccel PIO registers
# ]
system.mem_ranges = [AddrRange(0x80000000, size=0xC0000000-0x80000000)]

system.cpu = RiscvTimingSimpleCPU()

system.membus = SystemXBar()

system.memcpy_accel = MemCpyAccel(pioAddr=0xC0000000, pio_size=0x20)
system.memcpy_accel.dma = system.membus.cpu_side_ports
system.memcpy_accel.pio = system.membus.mem_side_ports

system.cpu.icache_port = system.membus.cpu_side_ports
system.cpu.dcache_port = system.membus.cpu_side_ports

system.cpu.createInterruptController()

system.mem_ctrl = MemCtrl()
system.mem_ctrl.dram = DDR3_1600_8x8()
system.mem_ctrl.dram.range = system.mem_ranges[0]

system.mem_ctrl.port = system.membus.mem_side_ports

system.system_port = system.membus.cpu_side_ports

# Set up SE workload
system.workload = SEWorkload.init_compatible(binary)

process = Process()
process.cmd = [binary]
system.cpu.workload = process
system.cpu.createThreads()

# --- Instantiate and simulate ---
root = Root(full_system=False, system=system)
m5.instantiate()

# map program memory
system.cpu.workload[0].map(        0x10000000,         0x10000000, 0x20000000, cacheable=True)
system.cpu.workload[0].map(0xFFFFFFFFF0000000,         0x80000000, 0x10000000, cacheable=True)
system.cpu.workload[0].map(        0x00000000,         0x90000000, 0x00001000, cacheable=True)
system.cpu.workload[0].map(        0xA0000000,         0xA0000000, 0x20000000, cacheable=True)
system.cpu.workload[0].map(        0xC0000000,         0xC0000000, 0x00000020, cacheable=False)

print("Beginning simulation!")
exit_event = m5.simulate()
print(f"Exiting @ tick {m5.curTick()} because {exit_event.getCause()}")

# Parse m5out, append to csv
stats_dict = parse_stats(m5statsfile)
strs = [stats_dict[key][0] for key in key_order] +  \
       [stats_dict[key][1] for key in key_order]
with open(reportfile, 'a+') as f:
    f.write(',')
    f.write(','.join(strs))

print(flush=True)
