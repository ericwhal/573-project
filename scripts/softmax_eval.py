import os
import json
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
gem5_command = f"LD_LIBRARY_PATH=/lib/ {gem5_path} {run_one_path}"

m5_ld_path = os.path.join(thispath, "../573-gem5/util/m5/build/riscv/out")
buildpath = os.path.join(thispath, "../build/")
databasedir = os.path.join(thispath, "../data/data/")
datadirs = [datadir for datadir in os.listdir(databasedir) if os.path.isdir(os.path.join(databasedir, datadir))]

reportfile = os.path.join(buildpath, "stats.csv")

# Write the csv header to the report file
# Ordered as accel, naive
with open(reportfile, 'w') as f:
    f.write("name,record,system.cpu.numCycles,simInsts,simOps,system.cpu.numCycles,simInsts,simOps")

source_datapaths = [os.path.join(databasedir, datadir) for datadir in datadirs]
main_file = os.path.join(srcpath, "eval_main.c")
softmaxlib_file = os.path.join(srcpath, "softmaxlib.c")

compiler_flags = '-static -nostdlib -O2'
ld_flags = ' '.join(['-L' + path for path in [m5_ld_path]])
libs = ' '.join(['-lm', '-lc','-lm5'])

def get_records(logits, probs, num_records):
    for record in range(num_records):
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


workdirs = set()
def parallel_worker(*args):
        # unpack args
        record, record_len, logit_record, prob_record = args        

        # make sure threadlocaldir exists
        thread_id = str(multiprocessing.current_process().pid)
        threadlocaldir = os.path.normpath(os.path.join(buildpath, thread_id))
        os.makedirs(threadlocaldir, exist_ok=True)
        workdirs.add(threadlocaldir)

        with open(os.path.join(threadlocaldir, 'data.h'), 'w') as header:
            header.write(f'int data_length = {record_len};\n')
            header.write(f'float logits[{len(logit_record)}] =' + '{' + ', '.join([str(f) for f in logit_record]) + '};\n')
            header.write(f'float probs[{len(prob_record)}] =' + '{' + ', '.join([str(f) for f in prob_record]) + '};\n')

        build_file = os.path.join(threadlocaldir, f'eval')
        include_flags = ' '.join(['-I' + path for path in [threadlocaldir, srcpath, gem5_include_path]])

        # With the header generated, compile the eval program
        compile_command = f"riscv64-unknown-linux-gnu-gcc {include_flags} {ld_flags} {compiler_flags} {softmaxlib_file} {main_file} {libs} -o {build_file}";
        subprocess.run(compile_command, shell=True)

        # prep reportfile with record & datapath after compile
        strs = [os.path.basename(datapath), str(record)]
        localreportfile = os.path.join(threadlocaldir, "stats.csv")
        with open(localreportfile, 'a+') as f:
            f.write('\n')
            f.write(','.join(strs))

        # run gem5
        subprocess.run([gem5_command], shell=True, cwd=threadlocaldir)




# iterate over workload types
for datapath in source_datapaths:
    logit_filepath = os.path.join(datapath, "logits.bin")
    probs_filepath = os.path.join(datapath, "probs.bin")
    manifest_filepath = os.path.join(datapath, "manifest.json")

    with open(logit_filepath, 'rb') as logit_file,  \
         open(probs_filepath, 'rb') as probs_file,  \
         open(manifest_filepath, 'r') as manifest_file:
        num_records = json.load(manifest_file)['count']

        print('\nProcessing', datapath, f'with {num_records} records', end=' ')

        # parallelize over records
        with multiprocessing.Pool(5) as p:
            p.starmap(parallel_worker, get_records(logit_file, probs_file, num_records), chunksize=50)

        with open(reportfile, 'a') as report:
            for d in workdirs:
                csv = os.path.join(d, 'stats.csv')
                with open(csv, 'r') as csv:
                    lines = [line for line in csv.readlines() if len(line) > 1]
                report.writelines(lines)

                # remove the workdir
                shutil.rmtree(d)
