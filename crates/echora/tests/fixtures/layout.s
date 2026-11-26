# Two function symbols for the ELF layout tests. From this directory:
# as --64 -o /tmp/echora-layout.o layout.s
# ld -m elf_x86_64 -z noseparate-code --build-id=none -e hot -o layout.elf /tmp/echora-layout.o
.text
.global hot, warm
.type hot, @function
hot:
    nop
    ret
.size hot, .-hot
.type warm, @function
warm:
    ret
.size warm, .-warm
