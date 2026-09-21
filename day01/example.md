# Day 1 example: from C++ to PTX to SASS

One kernel, followed all the way down. Everything below is real compiler output from CUDA 12.9 targeting `sm_80`, not a reconstruction. Reproduce it with:

```
nvcc -arch=compute_80 -ptx add.cu -o add.ptx      # C++ -> PTX
ptxas -arch=sm_80 add.ptx -o add.cubin            # PTX  -> SASS (cubin)
nvdisasm -c add.cubin                             # show the SASS
```

On a machine with a GPU, `nvcc -arch=native -cubin add.cu -o add.cubin` followed by `cuobjdump --dump-sass add.cubin` does the same for the card you are actually running on. The SASS will differ from what is printed here if the architecture differs.

## The source

```cuda
__global__ void add(const float *A, const float *B, float *C, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}
```

## PTX

```ptx
.version 8.8
.target sm_80
.address_size 64

.visible .entry _Z3addPKfS0_Pfi(
	.param .u64 _Z3addPKfS0_Pfi_param_0,
	.param .u64 _Z3addPKfS0_Pfi_param_1,
	.param .u64 _Z3addPKfS0_Pfi_param_2,
	.param .u32 _Z3addPKfS0_Pfi_param_3
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<4>;
	.reg .b32 	%r<6>;
	.reg .b64 	%rd<11>;

	ld.param.u64 	%rd1, [_Z3addPKfS0_Pfi_param_0];
	ld.param.u64 	%rd2, [_Z3addPKfS0_Pfi_param_1];
	ld.param.u64 	%rd3, [_Z3addPKfS0_Pfi_param_2];
	ld.param.u32 	%r2, [_Z3addPKfS0_Pfi_param_3];
	mov.u32 	%r3, %ntid.x;
	mov.u32 	%r4, %ctaid.x;
	mov.u32 	%r5, %tid.x;
	mad.lo.s32 	%r1, %r3, %r4, %r5;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd4, %rd1;
	mul.wide.s32 	%rd5, %r1, 4;
	add.s64 	%rd6, %rd4, %rd5;
	cvta.to.global.u64 	%rd7, %rd2;
	add.s64 	%rd8, %rd7, %rd5;
	ld.global.f32 	%f1, [%rd8];
	ld.global.f32 	%f2, [%rd6];
	add.f32 	%f3, %f2, %f1;
	cvta.to.global.u64 	%rd9, %rd3;
	add.s64 	%rd10, %rd9, %rd5;
	st.global.f32 	[%rd10], %f3;

$L__BB0_2:
	ret;
}
```

Line by line:

| Line | What it does |
|---|---|
| `.version 8.8` | The PTX ISA version this file is written in. It is a property of the compiler, not of the GPU. |
| `.target sm_80` | The virtual architecture the code was generated for, from `-arch=compute_80`. |
| `.address_size 64` | Pointers are 64-bit. |
| `.visible .entry _Z3addPKfS0_Pfi` | A kernel entry point visible outside this module. The name is the C++ mangled form of `add(float const*, float const*, float*, int)`; `extern "C"` would keep it as `add`. |
| `.param .u64 ..._param_0` | Kernel parameters live in a separate parameter space, not on a stack. Three 64-bit pointers and one 32-bit `int`. |
| `.reg .pred %p<2>;` | Declares a pool of up to 2 predicate registers. PTX has an unlimited virtual register file; `ptxas` maps it onto the real one later. |
| `.reg .f32 %f<4>;` etc. | The same for float, 32-bit and 64-bit untyped registers. `%r` holds `int` and the built-in indices, `%rd` holds addresses. |
| `ld.param.u64 %rd1, [..._param_0]` | Copies parameter A out of parameter space into a register. Parameters are not usable in place. |
| `mov.u32 %r3, %ntid.x` | `blockDim.x`. `%ntid` is a built-in special register: number of threads in the block. |
| `mov.u32 %r4, %ctaid.x` | `blockIdx.x`. CTA — cooperative thread array — is the PTX name for a block. |
| `mov.u32 %r5, %tid.x` | `threadIdx.x`. |
| `mad.lo.s32 %r1, %r3, %r4, %r5` | `i = blockDim.x * blockIdx.x + threadIdx.x` as one multiply-add. `.lo` keeps the low 32 bits of the product. |
| `setp.ge.s32 %p1, %r1, %r2` | Sets predicate `%p1` to `i >= n`. Note the inversion: the source asks `i < n`, and the compiler tests the negation so that it can branch over the body. |
| `@%p1 bra $L__BB0_2` | Predicated branch: if `%p1` is true, jump past the body to the return. This is the whole `if`. |
| `cvta.to.global.u64 %rd4, %rd1` | Converts a generic address to a global-space address. A CUDA pointer is generic and could point at global, shared or local memory; proving it is global lets the following load use `ld.global` instead of the slower generic `ld`. |
| `mul.wide.s32 %rd5, %r1, 4` | `i * sizeof(float)`, with a 32-bit input and a 64-bit result, so the offset cannot overflow. |
| `add.s64 %rd6, %rd4, %rd5` | `&A[i]`. |
| `ld.global.f32 %f1, [%rd8]` | Loads `B[i]`. The two loads are issued back to back, before the add, so the second does not wait on the first. |
| `add.f32 %f3, %f2, %f1` | The addition. |
| `st.global.f32 [%rd10], %f3` | Stores to `C[i]`. |
| `$L__BB0_2: ret;` | The branch target, and the return. |

Three things worth pointing out to the group: the `if` became a predicate plus a branch, not a masked instruction; the compiler inverted the condition; and address arithmetic is explicit, with a separate address-space conversion for each pointer.

## SASS

`ptxas` is a real optimising compiler, not an assembler that maps PTX one to one. The same kernel, assembled for `sm_80`:

```sass
_Z3addPKfS0_Pfi:
        /*0000*/  MOV R1, c[0x0][0x28] ;
        /*0010*/  S2R R6, SR_CTAID.X ;
        /*0020*/  S2R R3, SR_TID.X ;
        /*0030*/  IMAD R6, R6, c[0x0][0x0], R3 ;
        /*0040*/  ISETP.GE.AND P0, PT, R6, c[0x0][0x178], PT ;
        /*0050*/  @P0 EXIT ;
        /*0060*/  HFMA2.MMA R7, -RZ, RZ, 0, 2.384185791015625e-07 ;
        /*0070*/  ULDC.64 UR4, c[0x0][0x118] ;
        /*0080*/  IMAD.WIDE R4, R6, R7, c[0x0][0x168] ;
        /*0090*/  IMAD.WIDE R2, R6.reuse, R7.reuse, c[0x0][0x160] ;
        /*00a0*/  LDG.E R4, [R4.64] ;
        /*00b0*/  LDG.E R3, [R2.64] ;
        /*00c0*/  IMAD.WIDE R6, R6, R7, c[0x0][0x170] ;
        /*00d0*/  FADD R9, R4, R3 ;
        /*00e0*/  STG.E [R6.64], R9 ;
        /*00f0*/  EXIT ;
```

Line by line:

| Line | What it does |
|---|---|
| `MOV R1, c[0x0][0x28]` | Sets up the stack pointer from the constant bank. Present in almost every kernel; nothing to do with this code. |
| `S2R R6, SR_CTAID.X` | Special-register to register: reads `blockIdx.x`. `S2R` has long latency, which is why both reads are issued first. |
| `S2R R3, SR_TID.X` | Reads `threadIdx.x`. |
| `IMAD R6, R6, c[0x0][0x0], R3` | `i = blockIdx.x * blockDim.x + threadIdx.x`. `blockDim.x` was never loaded into a register: it is read directly out of the constant bank at offset `0x0`. Constant-bank operands are free in an instruction. |
| `ISETP.GE.AND P0, PT, R6, c[0x0][0x178], PT` | `P0 = (i >= n)`. `n` is also read straight from the constant bank — `0x178` is the fourth kernel parameter on `sm_80`, where the parameter block starts at `0x160`. `PT` is the always-true predicate, used as the identity for the `.AND`. |
| `@P0 EXIT` | Threads with `i >= n` leave here. The PTX branch over an empty tail became an early exit. |
| `HFMA2.MMA R7, -RZ, RZ, 0, 2.384185791015625e-07` | This puts the integer constant `4` into `R7`. `2.384185791015625e-07` is 2^-22, whose fp16 bit pattern is `0x0004`, and the other half is `0`, so the 32-bit result is `0x00000004`. The compiler builds the constant on the half-precision pipe because the integer pipe is busy with the address arithmetic. It is a scheduling decision, and a good illustration that SASS mnemonics cannot be read as intent. |
| `ULDC.64 UR4, c[0x0][0x118]` | Loads the global memory descriptor into a uniform register, used by the `LDG`/`STG` below. Uniform registers hold values identical across the warp and are read once for all 32 lanes. |
| `IMAD.WIDE R4, R6, R7, c[0x0][0x168]` | `&B[i]`, as one instruction: `i * 4 + base_B`, 32-bit inputs, 64-bit result into the register pair `R4:R5`. The three PTX instructions `cvta` + `mul.wide` + `add` collapsed into this. |
| `IMAD.WIDE R2, R6.reuse, R7.reuse, c[0x0][0x160]` | `&A[i]`. The `.reuse` flags say the operand collector can hold `R6` and `R7` for the next instruction instead of reading the register file again — an energy optimisation, not a correctness one. |
| `LDG.E R4, [R4.64]` | Loads `B[i]` from global memory. `.E` means a 64-bit ("extended") address. |
| `LDG.E R3, [R2.64]` | Loads `A[i]`. Both loads are in flight before anything consumes them. |
| `IMAD.WIDE R6, R6, R7, c[0x0][0x170]` | `&C[i]`, computed while the two loads are outstanding. This is latency hiding inside a single thread's instruction stream, on top of the warp-level kind. |
| `FADD R9, R4, R3` | The addition. The hardware stalls here until both loads return. |
| `STG.E [R6.64], R9` | Stores `C[i]`. |
| `EXIT` | Ends the thread. |

What the comparison shows:

- The kernel arguments never became registers. On `sm_80` they sit in the constant bank from `0x160` and are addressed there directly.
- Twelve PTX instructions of address arithmetic became three `IMAD.WIDE`.
- The order changed. The compiler hoisted both address computations and both loads ahead of the add, and computed the store address while the loads were in flight.
- `HFMA2.MMA` is doing integer constant materialisation. There is no half-precision arithmetic anywhere in the source.

PTX is what nvcc produces and what ships inside the binary for forward compatibility. SASS is what the GPU runs. Optimisation decisions that matter for performance are made in the step between them, which is why a profiler reports SASS and not PTX.

## Embedding assembly

Device code can contain PTX directly, with the same extended-asm syntax as GCC.

```cuda
__device__ __forceinline__ unsigned lane_id() {
    unsigned r;
    asm("mov.u32 %0, %%laneid;" : "=r"(r));
    return r;
}

__device__ __forceinline__ float fma_rn(float a, float b, float c) {
    float d;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(d) : "f"(a), "f"(b), "f"(c));
    return d;
}

__device__ __forceinline__ unsigned mad_via_temp(unsigned a, unsigned b, unsigned c) {
    unsigned d;
    asm("{\n\t"
        ".reg .u32 t;\n\t"
        "mul.lo.u32 t, %1, %2;\n\t"
        "add.u32    %0, t, %3;\n\t"
        "}"
        : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}

__device__ __forceinline__ void publish(int *flag, int v) {
    asm volatile("st.global.cg.s32 [%0], %1;" :: "l"(flag), "r"(v) : "memory");
}
```

The form is `asm(template : outputs : inputs : clobbers);`.

- `%0`, `%1`, ... are the operands in order, outputs first. A literal `%` in PTX — the special registers `%laneid`, `%tid`, `%smid` — must be written `%%`.
- Constraint letters name the register class, and therefore the width: `"h"` for 16-bit, `"r"` for 32-bit, `"l"` for 64-bit, `"f"` for `float`, `"d"` for `double`. There is no memory operand constraint; an address is passed as an `"l"` value.
- `"="` marks a write-only output, `"+"` a read-write one.
- Multiple instructions go in one `asm` block wrapped in braces, with `.reg` declaring any temporaries. Declaring them inside the braces keeps them out of the surrounding register pool.
- `volatile` stops the compiler from moving the block, merging it with an identical one, or deleting it when its result is unused. Needed for anything with a side effect.
- **The clobber list accepts one entry: `"memory"`.** Unlike host inline assembly there is no way to name clobbered registers, because the operands are virtual registers the compiler assigns. `"memory"` tells the compiler the block reads or writes memory it cannot see, so it must not keep values in registers across it or reorder memory operations around it. Any `asm` that loads or stores needs it.

Used in a kernel, the PTX comes out as written, framed by markers:

```ptx
	mov.f32 	%f3, 0f40000000;
	mov.f32 	%f4, 0f3F800000;
	// begin inline asm
	fma.rn.f32 %f1, %f2, %f3, %f4;
	// end inline asm
	// begin inline asm
	mov.u32 %r6, %laneid;
	// end inline asm
	...
	// begin inline asm
	{
	.reg .u32 t;
	mul.lo.u32 t, %r1, %r9;
	add.u32    %r7, t, %r10;
	}
	// end inline asm
```

The compiler substituted its own register names into the template and otherwise left the text alone. It does not verify the PTX: an error inside the block is reported by `ptxas`, at the next stage, against generated code rather than against the source line.

`ptxas` still schedules and optimises around the block, and still allocates the registers. So inline PTX gives control over instruction selection, not over register allocation or scheduling. It is worth using where an instruction has no intrinsic — some cache operators, `%smid`, `%laneid`, a specific rounding mode — and not otherwise, because anything expressible through an intrinsic is expressible through an intrinsic that also survives the next architecture.
