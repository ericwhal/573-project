#!/usr/bin/env bash

# riscv32-unknown-linux-gnu-gcc ./src/softmaxlib.c ./src/simple_test.c -o ./build/simple_test -static -nostdlib -ffreestanding -nodefaultlibs -nostartfiles -Ttext=0xD0000000 -mabi=ilp32
riscv32-unknown-elf-gcc ./src/softmaxlib.c ./src/simple_test.c -o ./build/simple_test -static -nostdlib -ffreestanding -nodefaultlibs -nostartfiles -Ttext=0x00000000

./573-gem5/build/RISCV/gem5.opt ./573-gem5/src/dev/acc/memcpy_test_v2.py
