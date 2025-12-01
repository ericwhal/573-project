#!/usr/bin/env bash

# riscv32-unknown-linux-gnu-gcc ./src/softmaxlib.c ./src/simple_test.c -o ./build/simple_test -static -nostdlib -ffreestanding -nodefaultlibs -nostartfiles -Ttext=0xD0000000 -mabi=ilp32
riscv64-unknown-elf-gcc ./src/softmaxlib.c ./src/simple_test.c -o ./build/simple_test -static -nostdlib -ffreestanding -nostartfiles -nodefaultlibs -Ttext=0x00000000 \
	-I ./573-gem5/include -L./573-gem5/util/m5/build/riscv/out -lm5

riscv64-unknown-elf-gcc ./573-gem5/src/dev/acc/big_test.c -o ./build/big_test -static -nostdlib -ffreestanding -nostartfiles -nodefaultlibs -Ttext=0x00000000 \
	-I ./573-gem5/include -L./573-gem5/util/m5/build/riscv/out -lm5 

./573-gem5/build/RISCV/gem5.opt ./573-gem5/src/dev/acc/memcpy_test_v2.py
