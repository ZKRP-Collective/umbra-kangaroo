.text
.globl rck_add256_raw_asm
.type rck_add256_raw_asm,@function
rck_add256_raw_asm:
    movq 0(%rsi), %rax
    addq 0(%rdx), %rax
    movq %rax, 0(%rdi)

    movq 8(%rsi), %rax
    adcq 8(%rdx), %rax
    movq %rax, 8(%rdi)

    movq 16(%rsi), %rax
    adcq 16(%rdx), %rax
    movq %rax, 16(%rdi)

    movq 24(%rsi), %rax
    adcq 24(%rdx), %rax
    movq %rax, 24(%rdi)

    setc %al
    movzbl %al, %eax
    ret
.size rck_add256_raw_asm, .-rck_add256_raw_asm

.macro RCK_COMBA_TERM aoff, boff
    movq \aoff(%rsi), %rdx
    mulxq \boff(%rcx), %rax, %r11
    addq %rax, %r8
    adcq %r11, %r9
    adcq $0, %r10
.endm

.macro RCK_COMBA_STORE ooff
    movq %r8, \ooff(%rdi)
    movq %r9, %r8
    movq %r10, %r9
    xorl %r10d, %r10d
.endm

.p2align 4
.globl rck_mul256_raw_bmi2
.type rck_mul256_raw_bmi2,@function
rck_mul256_raw_bmi2:
    movq %rdx, %rcx
    xorl %r8d, %r8d
    xorl %r9d, %r9d
    xorl %r10d, %r10d

    RCK_COMBA_TERM 0, 0
    RCK_COMBA_STORE 0

    RCK_COMBA_TERM 0, 8
    RCK_COMBA_TERM 8, 0
    RCK_COMBA_STORE 8

    RCK_COMBA_TERM 0, 16
    RCK_COMBA_TERM 8, 8
    RCK_COMBA_TERM 16, 0
    RCK_COMBA_STORE 16

    RCK_COMBA_TERM 0, 24
    RCK_COMBA_TERM 8, 16
    RCK_COMBA_TERM 16, 8
    RCK_COMBA_TERM 24, 0
    RCK_COMBA_STORE 24

    RCK_COMBA_TERM 8, 24
    RCK_COMBA_TERM 16, 16
    RCK_COMBA_TERM 24, 8
    RCK_COMBA_STORE 32

    RCK_COMBA_TERM 16, 24
    RCK_COMBA_TERM 24, 16
    RCK_COMBA_STORE 40

    RCK_COMBA_TERM 24, 24
    RCK_COMBA_STORE 48
    movq %r8, 56(%rdi)
    ret
.size rck_mul256_raw_bmi2, .-rck_mul256_raw_bmi2

.section .note.GNU-stack,"",@progbits
