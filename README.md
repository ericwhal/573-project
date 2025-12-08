# 573-project
Parent repo for our ECE573 project.

## Repo Structure
```
$GIT_ROOT/
  > 573_gem5                      # Our modified gem5 (collected as submodule)
  > 573-softmax-verilog           # Our RTL (collected as submodule)
  > riscv-gnu-toolchain           # The SiFive riscv toolchain (collected as submodule)
  > data                          # Empty directory for data
    [>] data                      # Directory you will unzip the datasets into
      [>] cifar100_vitb           # CIFAR dataset with 100 labels
      [>] cifar10_vitb            # CIFAR dataset with 10 labels
      [>] imagenet1k_vitb         # Imagenet dataset with 1000 labels
      [>] wt103_gpt2              # NLP dataset with 50257 labels
  > scripts                       # Holds scripts (all to be run from GIT_ROOT)
  > src                           # C source and header files for benchmarking and testing
 [>]build                         # Directory created by script(s) for build and run.
 [>]figures                       # Directory created by script(s) for figures.
 [>]riscv64-unknown-gnu-linux     # Recommended (but not required) directory for installing the riscv64 toolchain. Remember to add to $PATH!
```

## Build prerequisites
1. Make sure to run in a linux environment compatible with gem5.
2. If you are using Nix, `nix-shell` will create a new shell with all necessary dependencies. Otherwise, follow the instructions for each submodule. 
3. Create a python virtual environment, activate it, and install `requirements.txt`.
4. Run `$export GIT_ROOT=$(git rev-parse --show-toplevel)` (or remember to this directory). No scripts depend on this environment variable, it is purely a convenience for the instructions below.
5. Prepare the data directory.

### DATA
Download data from [here](https://drive.google.com/drive/folders/1iJfnUBAFhoRsth77pEQ2_yc-zqt8Hbdk?usp=drive_link) into a directory named 'data', then unzip. The final directory structure after decompressing should be `$GIT_ROOT/data/data/<dataset>/`.
The evaluation script dynamically collects available datasets, so if you want to skip one simply don't unzip it.

### NIX [nixos.org](https://nixos.org/)
Used to create a repeatable environment. Nix is not mandatory, but will make dependency collection easier.

## Build and run instructions
1. Initialize submodules
```bash
$ git submodule update --init --depth 1
```
2. Install the riscv64-unknown-linux-gnu toolchain by following the instructions in `$GIT_ROOT/riscv-gnu-toolchain/README.md` and running `make linux`.
3. Compile gem5 following the instructions in `573-gem5/README.md`. Use the gem5 virutal environment for the rest of the project.
4. Compile m5ops using the (instructions.)[https://www.gem5.org/documentation/general_docs/m5ops/]
5. Test the toolchain by navigating to `$GIT_ROOT` and running `./scripts/build_and_run.sh`. This should compile and run a short executable on gem5 in debug mode. Truncated sample output:
```
[...]
3406311000: system.memcpy_accel: PIO write: addr=0xc0000000 pioAddr=0xc0000000
3406311000: system.memcpy_accel: PIO write: offset=0 value=0xb0000000
3406311000: system.memcpy_accel:   Set src=0xb0000000
3406763000: system.memcpy_accel: PIO write: addr=0xc0000004 pioAddr=0xc0000000
3406763000: system.memcpy_accel: PIO write: offset=0x4 value=0xb0000190
3406763000: system.memcpy_accel:   Set dst=0xb0000190
Starting large memcpy accel test:
Length (bytes): 0x00000190
3482455000: system.memcpy_accel: PIO write: addr=0xc0000008 pioAddr=0xc0000000
3482455000: system.memcpy_accel: PIO write: offset=0x8 value=0x80000190
3482455000: system.memcpy_accel:   ctrl_and_len=0x80000190 len=400
3482455000: system.memcpy_accel:   Start bit set, calling startMemcpy()
3482455000: system.memcpy_accel: Starting memcpy: src=0xb0000000 dst=0xb0000190 len=400 (bytes)
3482455000: system.memcpy_accel: About to call dmaRead: addr=0xb0000000 size=400 pendingReadBuf=`K��V
3482562000: system.memcpy_accel: ENTER dmaReadComplete (bytes=<bad format>u)
3482574000: system.memcpy_accel: Cutoff -inf
3482574000: system.memcpy_accel: Cutoff -inf
3482574000: system.memcpy_accel: Cutoff -inf
3482574000: system.memcpy_accel: Cutoff -inf
3482574000: system.memcpy_accel: Cutoff -inf
3482574000: system.memcpy_accel: ZeroCount (cumulative) = 0
3482574000: system.memcpy_accel: AdderTree: switches=0→119 same=0→133
3482574000: system.memcpy_accel: Division: switching 0→32  same 0→68
3482665000: system.memcpy_accel: ENTER dmaWriteComplete
3482665000: system.memcpy_accel: DMA write complete: dst=0xb0000190 produced <bad format>u bytes; done bit set
Computation complete.
Sample outputs:
dst[0]   = 0x3c23d70b
dst[1]   = 0x3c23d70b
dst[10]  = 0x3c23d70b
dst[N-1] = 0x3c23d70b
Done.
Exiting @ tick 3697207000 because exiting with last active thread context
```
6. This lets us know that gem5 is compiled for the _unoptimized_ accelerator. If the Cutoff grows beyond -inf, we have compiled for the _optimized_ workload.
7. Navigate to `$GIT_ROOT`. Run `python3 scripts/softmax_eval.py 1` for a full unoptimized evaluation. Copy the path of the final reported file.
   - Modify gem5 by navigating to `573-gem5/src/dev/acc/`, opening `memcpyaccel.cc` and replacing `constexpr float cutoff` with either negative infinity (unoptimized) or -149.0 (optimized).
8. Navigate to `$GIT_ROOT`. Run `python3 scripts/softmax_eval.py 0` for a full optimized evaluation. Copy the path of the final reported file.
9. Navigate to `$GIT_ROOT`. Run `python3 scripts/softmax_eval.py 2` for a full naive evaluation. This will take a long time. Copy the path of the final reported file.
10. Generate charts by running `./scripts/plot_stacks_from_csv.py OPTIMIZED_CSV UNOPTIMIZED_CSV NAIVE_CSV`. The three `.csv`s should be in `$GIT_ROOT/data/`. The images will be saved to `$GIT_ROOT/figures/`.
11. Navigate to `$GIT_ROOT/573-softmax-verilog` and follow the README.md to get power numbers (our results hardcoded into `plot_stacks_from_csv.py`)
