#!/usr/bin/env bash

mkdir -p ./build/

riscv64-unknown-linux-gnu-gcc ./573-gem5/src/dev/acc/big_test.c -o ./build/big_test -static -nostdlib -ffreestanding -nostartfiles -nodefaultlibs -Ttext=0x00000000 \
	-I ./573-gem5/include -L./573-gem5/util/m5/build/riscv/out -lm5

./573-gem5/build/RISCV/gem5.opt --debug-flags=MemCpyAccelDebug ./573-gem5/src/dev/acc/memcpy_test_v2.py
