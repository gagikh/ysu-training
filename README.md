# CUDA դասընթաց համալսարանի դասախոսների համար

- **Անվանում**՝ GPU-ի ճարտարապետությունը և CUDA C/C++-ը դասախոսների համար
- **Համալսարան**՝ Երևանի պետական համալսարան, ինֆորմատիկայի և կիրառական մաթեմատիկայի ֆակուլտետ
- **Ձևաչափ**՝ 20 ակադեմիական ժամ, 10 պարապմունք՝ 2-ական ժամ, շաբաթը մեկ անգամ
- **Մասնակիցներ**՝ 6 դասախոս, որոնք ունեն OpenMP-ի, MPI-ի և thread-երի հետ աշխատելու փորձ, բայց GPU-ի հետ չեն աշխատել
- **Դասախոս**՝ Գագիկ Հակոբյան

Այս դասընթացը [ուսանողների 15-օրյա դասընթացի](https://github.com/gagikh/cuda) տարբերակն է, որում շեշտը դրված է ճարտարապետության վրա։ Մասնակիցներն արդեն տիրապետում են զուգահեռ ծրագրավորմանը, ուստի ծրագրավորման մոդելն անցնում ենք արագ, իսկ հիմնական ուշադրությունը դարձնում ենք hardware-ի աշխատանքին. streaming multiprocessor-ներ, warp scheduling, հիշողության հիերարխիա, cache-եր, և թե ինչպես է դրանցից յուրաքանչյուրը երևում չափումներում։

---

## Նպատակներ

1. Գրել GPU-ի կոդ, profile անել և օպտիմալացնել այն։
2. Ղեկավարել կուրսային աշխատանքներ և մագիստրոսական թեզեր AI-ի և computer vision-ի ոլորտում. գնահատել, թե արդյոք թեման իրագործելի է առկա hardware-ի վրա, ստուգել արագացման մասին պնդումները և որոշել, թե երբ է ձեռքով գրված kernel-ն արդարացված։

## Թեմաներ

| Օր | Թեմա |
|---|---|
| [1](day01/README.md) | CUDA-ի ծրագրավորման մոդելը և GPU-ի ճարտարապետությունը |
| [2](day02/README.md) | Thread-երի հիերարխիան, ինդեքսավորումը և launch configuration-ը |
| [3](day03/README.md) | SIMT կատարումը, warp-երը, divergence-ը և latency hiding-ը |
| [4](day04/README.md) | Հիշողության տեսակները և host-device փոխանցումները |
| [5](day05/README.md) | Shared memory-ն և bank conflict-ները |
| [6](day06/README.md) | Coalescing-ը, cache-երը և bandwidth-ը |
| [7](day07/README.md) | Warp intrinsic-ները, reduction-ը և atomic-ները |
| [8](day08/README.md) | Stream-երը, event-ները, ասինխրոնությունը և CUDA graph-երը |
| [9](day09/README.md) | Գրադարանները, tensor core-երը և ճշգրտությունը |
| [10](day10/README.md) | Առաջարկների քննարկում |

## Դասեր և լաբորատոր աշխատանքներ

| Օր | Դաս | Լաբորատոր աշխատանք |
|---|---|---|
| 1 | Ծրագրավորման մոդել, SM, հիշողության հիերարխիա | `report_device_capabilities()`, առաջին kernel-ը, կլաստերի GPU-ի պարամետրերի գրանցում |
| 2 | Thread, block, grid, ինդեքսավորում, occupancy | Վեկտորների գումարում, ապա 2D ինդեքսավորում երկու պատկերի վրա |
| 3 | SIMT pipeline, warp, divergence, latency hiding | Վեկտորների գումարում և ժամանակի համեմատություն CPU-ի ցիկլի հետ, ապա BGR-ից grayscale |
| 4 | Pageable, pinned, mapped և unified հիշողություն, փոխանցումներ | Pinned և pageable փոխանցումների համեմատություն Nsight Systems-ում |
| 5 | Shared memory, bank-եր, conflict-ներ, tiling | Tiled 2D ֆիլտր halo-ով, ապա Sobel |
| 6 | Coalescing, sector-ներ, L1/L2, coarsening | Օր 5-ի ֆիլտրի օպտիմալացում, արդյունքը՝ peak bandwidth-ի տոկոսով |
| 7 | Warp shuffle, reduction, scan, atomic-ներ | Պատկերի միջին արժեքը warp reduction-ով, histogram՝ global atomic-ներով և privatisation-ով |
| 8 | Stream-եր, event-ներ, ասինխրոն պատճենում, CUDA graph-եր | Մասերով pipeline մի քանի stream-ում, ապա նույնը՝ graph-ի տեսքով |
| 9 | cuBLAS, cuDNN, tensor core-եր, ճշգրտություն | Ձեռքով գրված kernel-ների փոխարինում գրադարանային կանչերով, tensor core-երի միացում |
| 10 | Քննարկում | Մասնակիցները ներկայացնում են իրենց առաջարկները |

Ամեն օրվա թղթապանակում կա այդ օրվա `README.md`-ն և `template.cu`-ն, որից սկսվում է աշխատանքը։ Ընդհանուր օժանդակ ֆայլերը [`common/`](common) թղթապանակում են՝ `cuda_check.h`, `device_info.h`, `timer.h`, `image_io.h`, `bmp_utilities.h`։

Դասընթացում ընդգրկված չեն textures և surfaces, stream-ordered memory allocation, մի քանի GPU-ով աշխատանքը և մի քանի node-ով MPI-ը։ Առաջին երկուսն ընդգրկված են [ուսանողների դասընթացում](https://github.com/gagikh/cuda)։

## Դասընթացի հիմնական գաղափարը

Օր 1-ում ստանում ենք կլաստերի GPU-ի իրական պարամետրերը՝ SM-ների քանակը, մեկ SM-ի ռեգիստրներն ու shared memory-ն, warp size-ը, bus width-ը, FP32-ի peak throughput-ը և peak bandwidth-ը։ Հաջորդ բոլոր օրերին չափումները համեմատվում են այդ թվերի հետ, ոչ թե գնահատվում «արագ» կամ «դանդաղ» բառերով։ Օր 6-ում kernel-ի արագությունն արտահայտվում է peak bandwidth-ի տոկոսով, իսկ Օր 10-ում նույն մեթոդով որոշվում է, թե արդյոք ուսանողի առաջարկած նախագիծն ընդհանրապես իրագործելի է։

## Կոմպիլացիա և գործարկում

```
./compile.sh day01/template.cu
sbatch submit.sh template
sbatch submit.sh template data/camera.bmp    # Օր 5-9
```

Արդյունքը գրվում է `logs/` թղթապանակում։ Թիրախը՝ H100 (`sm_90`), CUDA 12.8։ Օր 5–9-ում BMP պատկերները կարդացվում և գրվում են `common/bmp_utilities.h`-ով, իսկ փորձնական պատկերները `data/` թղթապանակում են։

## Առաջին պարապմունքից առաջ

- [CLUSTER.md](CLUSTER.md)՝ միացում, միջավայրի կարգավորում, առաջին գործարկում, ֆայլերի պատճենում կլաստերից։
- Բոլոր մասնակիցներն ունեն մուտք կլաստեր, CUDA-ի նույն տարբերակը և նույն միջավայրը։
- `./compile.sh day01/template.cu && sbatch submit.sh template` հրամանը ստեղծում է log, որում տպված են GPU-ի իրատեսական պարամետրերը։

## Օժանդակ նյութեր

- [Տերմինների ցանկ](GLOSSARY.md), [Intrinsic-ների ամփոփ աղյուսակ](INTRINSICS.md) (անգլերեն)
- [Ճարտարապետության մանրամասն նկարագրություն](ARCHITECTURE.md) (անգլերեն), [SM-ի անիմացիաներ](sm_animations.html)
- [Օպտիմալացման ստուգաթերթ](PERFORMANCE.md) (անգլերեն)
- [100 գործնական առաջադրանք](TASKS.md) (անգլերեն)

## Գրականություն

**Հիմնական**

- NVIDIA. *CUDA C Programming Guide* (PDF) — https://docs.nvidia.com/cuda/pdf/CUDA_C_Programming_Guide.pdf
  Դասընթացի հիմնական գրականությունը։ Ամեն օրվա «Գրականություն» բաժնում նշված են այդ պարապմունքի համար անհրաժեշտ գլուխները։
- NVIDIA. *CUDA C++ Best Practices Guide* — https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/ (v13.4)
- NVIDIA. *CUDA Programming Guide* (HTML, v13.4.2) — https://docs.nvidia.com/cuda/cuda-programming-guide/
- Hwu W., Kirk D., El Hajj I. *Programming Massively Parallel Processors*, 5th ed., Elsevier, 2026

**Profiling**

- NVIDIA. *Nsight Compute* — https://docs.nvidia.com/nsight-compute/ (v2026.3.1)
- NVIDIA. *Nsight Systems* — https://docs.nvidia.com/nsight-systems/ (v2026.4)
- Williams S., Waterman A., Patterson D. Roofline: An Insightful Visual Performance Model. *CACM* 52(4), 2009

**Գրադարաններ և ճշգրտություն**

- cuBLAS, cuSOLVER, cuSPARSE, cuFFT, cuRAND — https://docs.nvidia.com/cuda/
- cuDNN — https://docs.nvidia.com/deeplearning/cudnn/
- CUB — https://nvidia.github.io/cccl/cub/ · Thrust — https://nvidia.github.io/cccl/thrust/
- NVIDIA. *Train With Mixed Precision* — https://docs.nvidia.com/deeplearning/performance/
- Micikevicius P. et al. Mixed Precision Training. *ICLR*, 2018. arXiv:1710.03740

**Computer vision**

- Szeliski R. *Computer Vision: Algorithms and Applications*, 2nd ed. Անվճար PDF՝ https://szeliski.org/Book/

**Դասավանդման նյութեր**

- NVIDIA DLI Teaching Kit — Accelerated Computing. https://developer.nvidia.com/teaching-kits
- Oxford-ի CUDA դասընթացը, Mike Giles — https://people.maths.ox.ac.uk/~gilesm/cuda/
