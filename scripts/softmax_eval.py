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
srcpath = os.path.join(thispath, "../src/")
gem5_include_path = os.path.join(thispath, "../573-gem5/include/")
gem5_path = os.path.join(thispath, "../573-gem5/build/RISCV/gem5.opt")
run_one_path = os.path.join(thispath, "run_one_se_mode.py")
# gem5_command = f"LD_LIBRARY_PATH=/lib/ {gem5_path} --debug-flags=MemCpyAccelDebug,PortTrace {run_one_path}"
gem5_command = f"LD_LIBRARY_PATH=/lib/ {gem5_path} {run_one_path}"

m5_ld_path = os.path.join(thispath, "../573-gem5/util/m5/build/riscv/out")
buildpath = os.path.join(thispath, "../build/")
databasedir = os.path.join(thispath, "../data/data/")
datadirs = [datadir for datadir in os.listdir(databasedir) if os.path.isdir(os.path.join(databasedir, datadir))]
source_datapaths = [os.path.join(databasedir, datadir) for datadir in datadirs]

reportfile = os.path.join(buildpath, "stats.csv")

# Write the csv header to the report file
# Ordered as accel, naive
with open(reportfile, 'w') as f:
    f.write("name,accel,record,record_len,system.cpu.numCycles,simInsts,simOps,zeroCount,sameAdd,switchingAdd,sameDiv,switchingDiv,exit_code")

eval_main_file = os.path.join(srcpath, "eval_main.c")
naive_main_file = os.path.join(srcpath, "eval_naive_main.c")
main_files = [eval_main_file]*2 + [naive_main_file]
run_kinds = ['accel-opt','accel-unopt','naive']

run_idx = 1
main_file = main_files[run_idx]
run_kind = run_kinds[run_idx]

softmaxlib_file = os.path.join(srcpath, "softmaxlib.c")

compiler_flags = '-static -nostdlib -O2'
ld_flags = ' '.join(['-L' + path for path in [m5_ld_path]])
libs = ' '.join(['-lm', '-lc', '-lm5'])

def get_records(logits, probs, num_records, max_records=100):
    if max_records is None:
        max_records = num_records
    for record in tqdm.trange(min(max_records,num_records)):
        # 32 bits of length
        # 32 bits of meta
        logits_length = int.from_bytes(logits.read(4), byteorder='little')
        probs_length = int.from_bytes(probs.read(4), byteorder='little')

        if(logits_length != probs_length):
            raise ValueError(f'Record mismatch! Logits expect {logits_length} floats, probabilities expect {probs_length} floats. Record {record}')

        _ = logits.read(4), probs.read(4)

        logits_floats = struct.unpack(f'{logits_length}f', logits.read(logits_length*4))
        # probs_floats = struct.unpack(f'{probs_length}f', probs.read(probs_length*4))
        _ = probs.read(probs_length*4)
        yield record, logits_length, logits_floats # , probs_floats


# These get *copied* to the parallel_worker context!
exit_code_re = re.compile(r'exit: (0x[0-9A-Fa-f]{8})')
crashing_files = set()
def parallel_worker(*args, datapath=None):
    # early exits
    if datapath is None:
        return None
    if os.path.basename(datapath) in crashing_files:
        return None

    # unpack args
    # record, record_len, logit_record, prob_record = args
    record, record_len, logit_record = args

    # make sure threadlocaldir exists
    thread_id = str(multiprocessing.current_process().pid)
    threadlocaldir = os.path.normpath(os.path.join(buildpath, thread_id))
    os.makedirs(threadlocaldir, exist_ok=True)

    with open(os.path.join(threadlocaldir, 'data.h'), 'w') as header:
        header.write(f'#define data_length {record_len}\n')
        header.write(f'float logits[{len(logit_record)}] = ' + '{' + ', '.join([str(f) for f in logit_record]) + '};\n')
        # header.write(f'float probs[{len(prob_record)}] = ' + '{' + ', '.join([str(f) for f in prob_record]) + '};\n')

    build_file = os.path.join(threadlocaldir, f'eval')
    include_flags = ' '.join(['-I' + path for path in [threadlocaldir, srcpath, gem5_include_path]])

    # With the header generated, compile the eval program
    compile_command = f"riscv64-unknown-linux-gnu-gcc {include_flags} {ld_flags} {compiler_flags} {softmaxlib_file} {main_file} {libs} -o {build_file}";
    subprocess.run(compile_command, shell=True)

    # prep reportfile with record & datapath after compile
    strs = [os.path.basename(datapath), run_kind, str(record), str(record_len)]
    localreportfile = os.path.join(threadlocaldir, "stats.csv")
    with open(localreportfile, 'a+') as f:
        f.write('\n')
        f.write(','.join(strs))

    # run gem5
    result = subprocess.run([gem5_command], shell=True, cwd=threadlocaldir, capture_output=True, text=True, encoding='latin-1')
    exit_code = exit_code_re.search(result.stdout)
    with open(localreportfile, 'a+') as f:
        if exit_code is None:
          f.write(','*11+'CRASHED')
          if os.path.basename(datapath) in crashing_files:
              return threadlocaldir
          crashing_files.add(os.path.basename(datapath))
          print('========== BAD EXIT ==========')
          print(f'[!] Record {record} with len {record_len} failed!')
          print('[!] compile command, stdout & stderr below')
          print('------------------------------')
          print(compile_command)
          print('------------------------------')
          print(result.stdout)
          print('------------------------------')
          print(result.stderr)
          print('==============================', flush=True)
        else:
          f.write(',' + exit_code.group(1))
    return threadlocaldir


#==================================================
# Main
#==================================================
# iterate over workload types
# One configurable setting, maximum number of records to read per file
max_records = None
if len(sys.argv) > 1:
    max_records = int(sys.argv[1])

for datapath in source_datapaths:
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
        with multiprocessing.Pool(None) as p:
            workdirs = set(p.imap_unordered(parallel_worker_wrap, get_records(logit_file, probs_file, num_records, max_records=max_records), chunksize=50 if max_records is None else min(50, max(1, max_records//16))))

        print(f'Collecting data from {len(workdirs) - (1 if None in workdirs else 0)} workers')
        with open(reportfile, 'a+') as report:
            for d in tqdm.tqdm(workdirs):
                if d is None:
                    continue
                csv = os.path.join(d, 'stats.csv')
                with open(csv, 'r') as csv:
                    data = csv.read()
                report.write(data)

                # remove the workdir
                shutil.rmtree(d)
