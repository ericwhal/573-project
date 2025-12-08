import os
import re
import sys
import json
import tqdm
import shutil
import struct
import subprocess
import multiprocessing

####################################################################################################
# In order to collect evaluations without the ability to read in data, we need to compile
# lots of binaries. We will inject the logits & probs into each binary as a vector in a header file.
# According to the data README:
# Each data directory includes three files: 
# - logits.bin
# - probs.bin
# - manifest.json
# We want to get the number of records from manifest.json (we won't be checking the checksums here)
# and using that number of records, read the .bin files.
# The .bin files have record format: 64 bits of header as length (4 bytes), id (2 bytes), flags (2 bytes)
# followed by float32[length].
# Flags are 0 (none), 1 (PADDED with -inf), 2 (truncated)
####################################################################################################

# Set up the build paths:
# 573-project
#   /scripts/THIS_FILE
#   /src/
#   /build
#      /DATA_DIRS/
thispath = os.path.dirname(os.path.realpath(__file__))

databasedir = os.path.join(thispath, "../data/data/")
datadirs = [datadir for datadir in os.listdir(databasedir) if os.path.isdir(os.path.join(databasedir, datadir))]
source_datapaths = [os.path.join(databasedir, datadir) for datadir in datadirs]

def get_records(logits, probs, num_records, max_records=100):
    if max_records is None:
        max_records = num_records
    for record in range(min(max_records,num_records)):
        # 32 bits of length
        # 32 bits of meta
        logits_length = int.from_bytes(logits.read(4), byteorder='little')
        probs_length = int.from_bytes(probs.read(4), byteorder='little')

        if(logits_length != probs_length):
            raise ValueError(f'Record mismatch! Logits expect {logits_length} floats, probabilities expect {probs_length} floats. Record {record}')

        _ = logits.read(4), probs.read(4)

        logits_floats = struct.unpack(f'{logits_length}f', logits.read(logits_length*4))
        probs_floats = struct.unpack(f'{probs_length}f', probs.read(probs_length*4))
        yield record, logits_length, logits_floats, probs_floats


#==================================================
# Main
#==================================================
# iterate over workload types
# One configurable setting, maximum number of records to read per file
max_records = None
if len(sys.argv) > 1:
    max_records = int(sys.argv[1])

for datapath in source_datapaths:
    if 'cifar10_' not in datapath:
        continue
    logit_filepath = os.path.join(datapath, "logits.bin")
    probs_filepath = os.path.join(datapath, "probs.bin")
    manifest_filepath = os.path.join(datapath, "manifest.json")

    with open(logit_filepath, 'rb') as logit_file,  \
         open(probs_filepath, 'rb') as probs_file,  \
         open(manifest_filepath, 'r') as manifest_file:
        num_records = json.load(manifest_file)['count']

        print('Processing', datapath, f'with {num_records} records')

        # parallelize over records. Unpack x (we know it's a tuple returned from get_records)
        def parallel_worker_wrap(x):
            return parallel_worker(*x, datapath=datapath)

        # Using the unordered imap lazily consumes from get_records, which in turn enables tqdm to track time accurately. Starmap isn't lazy.
        for args in get_records(logit_file, probs_file, num_records, max_records=max_records):
            record, record_len, logit_record, prob_record = args
            print('l:',logit_record)
            print('p:',prob_record)
