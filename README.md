# 573-project
Parent repo for our ECE573 project.

## NIX
Used to create a repeatable sandbox-y environment.

## Building a minimal Linux image & disk for gem5:
Following instructions (here)[https://gem5.googlesource.com/public/gem5-resources/+/HEAD/src/riscv-fs/README.md], with the following modifications:
- use `--enable-multilib` when configuring the riscv-gnu-toolchain, the proxy kernel requires it.
- use branch `linux-rolling-stable` (conflict between new linker default behavior and old linux makefile expectation)
- Some tweaks for NixOS/FHSEnv: you might need to manually set `LD_LIBRARY_PATH=/lib/` before each make invocation when compiling linux
- You may need to modify `busybox/scripts/kconfig/lxdialog/check-lxdialog.sh` to add `int` before `main() {}` for `make menuconfig` to work.
- In order to hold all the data, the disk needs to be quite large. 20GB is overkill, but is nice and round.

## DATA
Download data from (here)[https://drive.google.com/drive/folders/1iJfnUBAFhoRsth77pEQ2_yc-zqt8Hbdk?usp=drive_link] and unzip into a directory 'data'.

