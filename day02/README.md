# Օր 2. Thread-երի հիերարխիան, ինդեքսավորումը և launch configuration-ը

## Նպատակներ
- Բաժանել աշխատանքը thread-երի, block-երի և grid-ի միջև՝ 1D և 2D տվյալների համար
- Գրել global index-ի բանաձևը և բացատրել, թե ինչու այն ունի հենց այդ տեսքը
- Տարբերել coalesced և uncoalesced ինդեքսավորումը, և ինդեքսավորել այնպես, որ warp-ը դիմի հաջորդական հասցեների
- Ընտրել block size և ստուգել ընտրությունը `cudaOccupancyMaxActiveBlocksPerMultiprocessor`-ով
- Գրել grid-stride loop, որը ճիշտ է աշխատում մուտքի ցանկացած չափի դեպքում՝ նույն launch configuration-ով

## Հիմնական հասկացություններ
- Thread, block, grid՝ կառուցվածքը և համարակալումը
- Launch configuration և kernel-ի կանչ
- Thread-երի 1D և 2D ինդեքսավորում
- Memory coalescing-ը որպես ինդեքսի բանաձևի հետևանք
- Occupancy և block size-ի ընտրություն (առաջին ծանոթություն. hardware-ի պատճառները՝ Օր 3-ում և Օր 6-ում)
- Grid-stride loop

## Սահմանումներ
**Thread** — Կատարման ամենափոքր միավորը։ Block-ի ներսում thread-ի դիրքը տալիս է `threadIdx`-ը, grid-ի ներսում՝ `threadIdx`-ի, `blockIdx`-ի և `blockDim`-ի համակցությունը։

**Block** — Thread-երի խումբ (առավելագույնը `maxThreadsPerBlock` thread), որոնք կատարվում են նույն SM-ի վրա և կարող են համագործակցել shared memory-ի ու `__syncthreads()`-ի միջոցով։ Kernel-ի launch-ը ստեղծում է block-երի grid։

**Grid** — Մեկ kernel-ի կանչով գործարկված block-երի ամբողջությունը՝ `<<<grid, block>>>`։

**Launch configuration** — Kernel-ի կանչի արգումենտները՝ grid-ի չափերը block-երով և block-ի չափերը thread-երով, երկուսն էլ՝ մինչև երեք չափողականությամբ, ինչպես նաև ոչ պարտադիր dynamic shared memory-ի չափը և stream-ը։

**`blockIdx`, `threadIdx`, `blockDim`, `gridDim`** — Ներկառուցված (built-in) փոփոխականներ, որոնք հասանելի են device-ի կոդում առանց հայտարարելու կամ փոխանցելու՝ block-ի index-ը grid-ում, thread-ի index-ը block-ում, block-ի չափերը, grid-ի չափերը։ Դրանք միայն կարդալու համար են, `uint3` տիպի են (չափերը՝ `dim3`), և ամեն thread տեսնում է իր սեփական արժեքները։

Դրանց արժեքները ձեր կոդը չի տալիս, և դրանք հիշողության մեջ պահվող սովորական փոփոխականներ չեն.

| Փոփոխական | PTX | Ով է տալիս արժեքը |
|---|---|---|
| `threadIdx` | `%tid` | hardware-ը, երբ SM-ը ստեղծում է block-ի thread-երը |
| `blockIdx` | `%ctaid` | GPU-ի work distributor-ը, երբ block-ը վերագրում է SM-ին |
| `blockDim` | `%ntid` | host-ը՝ launch configuration-ով (`<<<grid, block>>>`) |
| `gridDim` | `%nctaid` | host-ը՝ launch configuration-ով |

Առաջին երկուսը տալիս է hardware-ը, և դրանցով են thread-երը տարբերվում իրարից։ Մյուս երկուսը launch-ի պարամետրեր են և նույնն են grid-ի բոլոր thread-երի համար։ Դրանցից որևէ մեկը կարդալը հրաման է, ոչ թե հիշողության դիմում։ Օր 1-ի օրինակում այս PTX-ը

```ptx
mov.u32  %r3, %ntid.x;     // blockDim.x
mov.u32  %r4, %ctaid.x;    // blockIdx.x
mov.u32  %r5, %tid.x;      // threadIdx.x
```

դառնում է այսպիսի SASS, որտեղ `threadIdx`-ը և `blockIdx`-ը կարդացվում են special register-ից, իսկ `blockDim`-ը՝ ուղիղ constant bank-ից.

```sass
S2R  R6, SR_CTAID.X ;
S2R  R3, SR_TID.X ;
IMAD R6, R6, c[0x0][0x0], R3 ;   // c[0x0][0x0]-ն blockDim.x-ն է
```

**Resident block-եր** — Մեկ SM-ին միաժամանակ վերագրված block-երը։ Դրանց քանակը երեք մեծություններից ամենափոքրն է՝ մեկ SM-ում block-երի hardware-ային սահմանը, մեկ SM-ի ռեգիստրների քանակը՝ բաժանած մեկ block-ի պահանջած ռեգիստրների վրա, և մեկ SM-ի shared memory-ն՝ բաժանած մեկ block-ի պահանջած shared memory-ի վրա։

**Occupancy** — Մեկ SM-ում միաժամանակ resident warp-երի քանակի հարաբերությունն այն առավելագույնին, որ SM-ը կարող է պահել։ Այն սահմանափակվում է այն ռեսուրսով, որն առաջինն է սպառվում՝ մեկ thread-ի ռեգիստրներ, մեկ block-ի shared memory կամ thread-երի քանակի սահման։ Occupancy-ն **latency hiding**-ի միջոց է, ոչ թե նպատակ. մոտավորապես 50 տոկոսից հետո դրա աճը գրեթե չի արագացնում կատարումը, իսկ thread coarsening-ը դիտմամբ իջեցնում է occupancy-ն՝ ամեն thread-ին ավելի շատ աշխատանք տալով (Օր 6)։

**`cudaOccupancyMaxActiveBlocksPerMultiprocessor`** — Runtime-ի կանչ, որը հաշվում է, թե տվյալ kernel-ի և block size-ի դեպքում քանի block կլինի resident մեկ SM-ում՝ առանց kernel-ը գործարկելու։

```c
cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    int        *numBlocks,        // ելք՝ resident block-երի քանակը մեկ SM-ում
    const void *func,             // kernel-ի սիմվոլը
    int         blockSize,        // block-ի thread-երի քանակը, որով նախատեսում եք գործարկել
    size_t      dynamicSMemSize); // մեկ block-ի dynamic shared memory-ն բայթերով, կամ 0
```

`func`-ը հենց kernel-ի անունն է. C++-ում այն վերածվում է ֆունկցիայի հասցեի, ուստի կանչը գրվում է այսպես՝ `cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, my_kernel, 256, 0)`։ `dynamicSMemSize`-ը launch-ի երրորդ արգումենտն է՝ `my_kernel<<<grid, block, smem>>>`, այլ ոչ թե ստատիկ հայտարարված `__shared__` զանգվածների չափը. դրանք կոմպիլյատորն արդեն հաշվի է առել։

Արդյունքը block-երի քանակ է, ոչ թե տոկոս։ Occupancy-ն հաշվվում է դրանից.

```c
int n;
CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, my_kernel, blockSize, 0));

cudaDeviceProp p;
CUDA_CHECK(cudaGetDeviceProperties(&p, 0));

double occupancy = double(n * blockSize) / p.maxThreadsPerMultiProcessor;
```

Պատասխանը հաշվվում է կոմպիլացված kernel-ի ռեգիստրների քանակից և shared memory-ի պահանջից՝ համեմատելով դրանք այս device-ի սահմանների հետ։ Ոչինչ չի գործարկվում, ուստի սա residency-ի վերին սահմանն է, ոչ թե չափում. այն ասում է, թե քանի block **կարող է** resident լինել, և ոչ թե որքան էր SM-ն իրականում զբաղված։ Իրական չափումը Nsight Compute-ի `sm__warps_active.avg.pct_of_peak_sustained_active` մետրիկն է։ Երկու թվերը տարբերվում են, երբ block-երն ավարտվում են տարբեր ժամանակ, կամ երբ grid-ը չափազանց փոքր է ամբողջ device-ը զբաղեցնելու համար։

Երկու հարակից ֆունկցիա.

- `cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags` — նույն ֆունկցիան՝ flags պարամետրով. `cudaOccupancyDefault`-ը պահում է սովորական վարքը, իսկ `cudaOccupancyDisableCachingOverride`-ն անջատում է platform-ից կախված caching-ի ճշգրտումը։
- `cudaOccupancyMaxPotentialBlockSize` — պատասխանում է հակառակ հարցին։ Ձեր ընտրած block size-ը գնահատելու փոխարեն այն առաջարկում է ամենաբարձր occupancy տվող block size-ը և այն ամենափոքր grid-ը, որով այդ occupancy-ն ձեռք է բերվում։ Օգտակար է որպես ելակետ, ոչ թե որպես վերջնական պատասխան. ամենաբարձր occupancy-ն դեռ չի նշանակում ամենաարագ կատարում։

**Coalescing (memory coalescing)** — Երբ warp-ի 32 lane-երը դիմում են հաջորդական հասցեների, hardware-ը դրանք սպասարկում է մեկ 128-բայթանոց transaction-ով՝ մինչև 32 առանձին transaction-ի փոխարեն։ Coalescing-ը warp-ի հատկություն է, ոչ թե thread-ի. կարևորն այն է, թե մեկ հրամանի ընթացքում 32 lane-երը միասին ինչ հասցեների են դիմում, և ոչ թե այն, թե մեկ thread-ը ժամանակի ընթացքում որ հասցեներով է անցնում։

**Grid-stride loop** — Kernel գրելու ձև, երբ grid-ը տվյալների չափին չի հարմարեցվում. ֆիքսված քանակով thread-երից ամեն մեկը ցիկլով մշակում է մի քանի տարր՝ ամեն քայլին առաջ գնալով thread-երի ընդհանուր քանակով։ Ճիշտ է աշխատում մուտքի ցանկացած չափի դեպքում՝ առանց launch-ի չափերը վերահաշվելու։

## Ֆունկցիաներ

```c
// Block-ի barrier. ոչ մի thread այն չի անցնում, մինչև block-ի բոլոր thread-երը չհասնեն
void __syncthreads(void);

// *numBlocks-ում գրում է, թե տվյալ block size-ի և dynamic shared memory-ի դեպքում
// func kernel-ից քանի block կլինի resident մեկ SM-ում
cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int *numBlocks, const void *func,
                                                          int blockSize, size_t dynamicSMemSize);

// Նույնը՝ flags-ով. cudaOccupancyDefault կամ cudaOccupancyDisableCachingOverride
cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int *numBlocks, const void *func,
                                                                   int blockSize, size_t dynamicSMemSize,
                                                                   unsigned int flags);

// Առաջարկում է ամենաբարձր occupancy տվող block size-ը և դրա համար անհրաժեշտ ամենափոքր grid-ը
template <class T>
cudaError_t cudaOccupancyMaxPotentialBlockSize(int *minGridSize, int *blockSize, T func,
                                               size_t dynamicSMemSize = 0, int blockSizeLimit = 0);
```

## Ինչպես է block-ը տեղաբաշխվում SM-ի վրա
Block-ը վերագրվում է մեկ SM-ի և մնում այնտեղ մինչև ավարտը։ Թե քանի block կարող է միաժամանակ տեղավորվել SM-ում, որոշում են երեք սահմանափակում՝ մեկ thread-ի ռեգիստրները, մեկ block-ի shared memory-ն և resident block-երի hardware-ային սահմանը։ Որոշիչը ամենախիստն է։ Հենց դրա համար block size-ը hardware-ի հարց է, ոչ թե ճաշակի։

## Պատկեր
![Block-երի grid, ամեն block՝ thread-երի երկչափ զանգված, կողքին՝ global index-ի բանաձևը](thread_hierarchy.svg)

Launch-ը ստեղծում է block-երի grid, և ամեն block ինքը thread-երի միաչափ, երկչափ կամ եռաչափ զանգված է։ `blockIdx`-ը ցույց է տալիս, թե thread-ը որ block-ում է, իսկ `threadIdx`-ը՝ որ դիրքում է այդ block-ի ներսում։ Գծապատկերում բերված բանաձևը օգտագործվում է գրեթե բոլոր հաջորդ kernel-ներում։

## Անիմացիա
![Global index-ի բանաձևը հաշվվում է երեք block-ի չորս thread-ի համար. blockIdx.x-ից, blockDim.x-ից և threadIdx.x-ից ստացվում է կոնկրետ թիվ](thread_indexing.svg)

Բանաձևը միշտ `blockIdx.x * blockDim.x + threadIdx.x` է. փոխվում են միայն block-ի և thread-ի արժեքները։

![Չորս thread 4 քայլով անցնում է 16 տարրանոց զանգվածով. ամեն կրկնությանը ամեն thread մշակում է նոր տարր](grid_stride_loop.svg)

Չորս thread-ը չորս կրկնությամբ մշակում է տասնվեց տարր։ Եթե զանգվածը մեծացնենք մինչև երեսուներկու տարր, կլինի ութ կրկնություն՝ նույն launch configuration-ով։ Սա լաբորատոր առաջադրանքի grid-stride loop-ն է։

## Գրականություն
- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/pdf/CUDA_C_Programming_Guide.pdf) — 2. Programming Model · 5. Performance Guidelines
- CUDA C++ Best Practices Guide — Occupancy, Coalesced Access to Global Memory

## Լաբորատոր առաջադրանք
Վեկտորների գումարում, ապա նույն kernel-ը՝ ընդլայնված 2D ինդեքսավորմամբ, երկու grayscale պատկերի գումարման համար։

## Ինքնուրույն աշխատանք
1. Իրականացնել վեկտորների 1D գումարում տարբեր չափի զանգվածների համար։ Սահմանները ստուգել այն չափերի դեպքում, որոնք block size-ի բազմապատիկ չեն։
2. Ընդլայնել 2D ինդեքսավորմամբ և գումարել երկու grayscale պատկեր՝ պիքսել առ պիքսել։
3. Չափել 32, 64, 128 և 256 block size-երի ժամանակը։ Ամեն մեկի համար կանչել `cudaOccupancyMaxActiveBlocksPerMultiprocessor` և նշել, թե որ կետից սկսած չափված ժամանակն այլևս չի համապատասխանում occupancy-ի թվերին։
4. Իրականացնել նույն kernel-ը grid-stride loop-ով։ Ստուգել, որ այն ճիշտ է աշխատում նաև այն դեպքում, երբ `n`-ը շատ ավելի մեծ է `blocks * threads`-ից, առանց launch configuration-ը փոխելու։
5. Գրել երկու copy kernel՝ մեկը `blockIdx.x * blockDim.x + threadIdx.x` ինդեքսավորմամբ, մյուսը՝ `threadIdx.x * gridDim.x + blockIdx.x`։ Երկուսն էլ ճիշտ են։ Չափել երկուսի ժամանակը։
6. Ընտրած block size-ի համար ձեռքով հաշվել, թե քանի block է resident մեկ SM-ում, ապա ստուգել occupancy-ի API-ով։

## Ինքնաստուգում
Պատասխանները տրված չեն։

1. Ի՞նչ սխալ կառաջանա, եթե global index-ի բանաձևից բաց թողնեք `blockDim.x`-ը։
2. Ինչու՞ է grid-stride loop-ը ճիշտ մնում, երբ `n`-ը կրկնապատկվում է, իսկ այն kernel-ը, որտեղ ամեն thread մշակում է մեկ տարր՝ ոչ։
3. 5-րդ առաջադրանքի երկու բանաձևն էլ ճիշտ արդյունք են տալիս։ Ո՞ր չափումով կորոշեք, թե որն ընտրել։
4. Kernel-ը գործարկում եք `<<<100, 256>>>`-ով 20 000 տարրի համար՝ սահմանների ստուգմամբ։ Քանի՞ thread ոչ մի աշխատանք չի կատարում։

## Կոդի template
Տես [`template.cu`](template.cu)։
