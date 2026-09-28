# Օր 2. Thread-երի հիերարխիան, ինդեքսավորումը և launch configuration-ը

## Նպատակներ
- Բաժանել աշխատանքը thread-երի, block-երի և grid-ի միջև՝ 1D և 2D տվյալների համար
- Գրել global index-ի բանաձևը և բացատրել, թե ինչու է այն հենց այդ տեսքն ունի
- Տարբերել coalesced և uncoalesced ինդեքսավորումը, և ինդեքսավորել այնպես, որ warp-ի դիմումները հարակից լինեն
- Ընտրել block size և ստուգել ընտրությունը `cudaOccupancyMaxActiveBlocksPerMultiprocessor`-ով
- Գրել grid-stride loop, որը ճիշտ է մուտքի ցանկացած չափի համար՝ նույն launch configuration-ով

## Հիմնական հասկացություններ
- Thread, block, grid՝ կառուցվածքը և համարակալումը
- Launch configuration և kernel-ի կանչ
- 1D և 2D thread ինդեքսավորում
- Memory coalescing-ը որպես ինդեքսի բանաձևի հետևանք
- Occupancy և block size-ի ընտրություն (առաջին անգամ. hardware-ի պատճառը՝ Օր 3 և Օր 6)
- Grid-stride loop

## Սահմանումներ
**Thread** — Կատարման ամենափոքր միավորը։ Իր block-ի ներսում որոշվում է `threadIdx`-ով, grid-ի ներսում՝ `threadIdx`-ի, `blockIdx`-ի և `blockDim`-ի համակցությամբ։

**Block** — Thread-երի խումբ, մինչև `maxThreadsPerBlock`, որոնք կատարվում են նույն SM-ի վրա և կարող են համագործակցել shared memory-ով ու `__syncthreads()`-ով։ Kernel-ի launch-ը ստեղծում է block-երի grid։

**Grid** — Մեկ kernel-ի կանչով գործարկված block-երի ամբողջությունը՝ `<<<grid, block>>>`։

**Launch configuration** — Kernel-ի կանչի արգումենտները՝ grid-ի չափերը block-երով և block-ի չափերը thread-երով, ամեն մեկը մինչև երեքչափանի, գումարած ոչ պարտադիր dynamic shared memory-ի չափը և stream-ը։

**`blockIdx`, `threadIdx`, `blockDim`, `gridDim`** — Ներկառուցված (built-in) փոփոխականներ, հասանելի device-ի կոդում առանց հայտարարելու կամ փոխանցելու՝ block-ի index-ը grid-ում, thread-ի index-ը block-ում, block-ի չափերը, grid-ի չափերը։ Միայն-կարդալու են, `uint3` տիպի (չափերը՝ `dim3`), և ամեն thread տեսնում է իր սեփական արժեքները։

Ձեր կոդում ոչինչ դրանք չի սկզբնարժեքավորում, և դրանք հիշողության մեջ դրված սովորական փոփոխականներ չեն․

| Փոփոխական | PTX | Ում կողմից է դրվում |
|---|---|---|
| `threadIdx` | `%tid` | hardware-ի, երբ SM-ը ստեղծում է block-ի thread-երը |
| `blockIdx` | `%ctaid` | GPU-ի work distributor-ի, երբ block-ը վերագրում է SM-ին |
| `blockDim` | `%ntid` | launch configuration-ի. host-ը այն գրել է `<<<grid, block>>>`-ում |
| `gridDim` | `%nctaid` | նույն կերպ՝ launch configuration-ի |

Առաջին երկուսը hardware-ի ինքնությունն են, մյուս երկուսը՝ launch-ի պարամետրերը, և հենց դրա համար են վերջին երկուսը նույնը grid-ի բոլոր thread-երի համար։ Դրանցից որևէ մեկը կարդալը հրաման է, ոչ թե հիշողության դիմում։ Օր 1-ի օրինակում PTX-ը

```ptx
mov.u32  %r3, %ntid.x;     // blockDim.x
mov.u32  %r4, %ctaid.x;    // blockIdx.x
mov.u32  %r5, %tid.x;      // threadIdx.x
```

դառնում է SASS, որտեղ `threadIdx`-ը և `blockIdx`-ը կարդացվում են special register-ի հրամանով, իսկ `blockDim`-ը՝ ուղիղ constant bank-ից․

```sass
S2R  R6, SR_CTAID.X ;
S2R  R3, SR_TID.X ;
IMAD R6, R6, c[0x0][0x0], R3 ;   // c[0x0][0x0]-ն blockDim.x-ն է
```

**Resident block-եր** — Մեկ SM-ին միաժամանակ վերագրված block-երը։ Քանակը երեք սահմանափակումից ամենափոքրն է՝ block-երի hardware-ային սահմանը մեկ SM-ում, մեկ SM-ի ռեգիստրները բաժանած block-ի ռեգիստրների պահանջի վրա, և մեկ SM-ի shared memory-ն բաժանած block-ի shared memory-ի պահանջի վրա։

**Occupancy** — Քանի warp է միաժամանակ resident մեկ SM-ի վրա՝ հարաբերած այն առավելագույնին, որ SM-ը կարող է պահել։ Սահմանափակում է այն ռեսուրսը, որն առաջինն է սպառվում՝ ռեգիստրներ մեկ thread-ում, shared memory մեկ block-ում, կամ thread-երի քանակի սահմանը։ Սա միջոց է **latency hiding**-ի համար, ոչ թե նպատակ. մոտավորապես 50 տոկոսից հետո շահույթը հարթվում է, իսկ thread coarsening-ը դիտմամբ իջեցնում է occupancy-ն՝ մեկ thread-ին ավելի շատ աշխատանք տալու դիմաց (Օր 6)։

**`cudaOccupancyMaxActiveBlocksPerMultiprocessor`** — Runtime-ի կանչ, որը վերադարձնում է, թե տվյալ kernel-ի և block size-ի դեպքում քանի block կլինի resident մեկ SM-ում՝ առանց kernel-ը գործարկելու։

```c
cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    int        *numBlocks,        // ելք՝ resident block-երը մեկ SM-ում
    const void *func,             // kernel-ի սիմվոլը
    int         blockSize,        // thread-երը մեկ block-ում, որով մտադիր եք գործարկել
    size_t      dynamicSMemSize); // dynamic shared memory-ն մեկ block-ում, բայթերով, 0՝ եթե չկա
```

`func`-ը հենց kernel-ի անունն է. C++-ում այն վերածվում է ֆունկցիայի հասցեի, ուստի կանչը գրվում է այսպես՝ `cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, my_kernel, 256, 0)`։ `dynamicSMemSize`-ը launch-ի երրորդ արգումենտն է՝ `my_kernel<<<grid, block, smem>>>`, ոչ թե ստատիկ հայտարարված `__shared__` զանգվածները. դրանք կոմպիլյատորն արդեն հաշվի է առել։

Վերադառնում է block-երի քանակ, ոչ թե տոկոս։ Occupancy-ն ստացվում է դրանից․

```c
int n;
CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, my_kernel, blockSize, 0));

cudaDeviceProp p;
CUDA_CHECK(cudaGetDeviceProperties(&p, 0));

double occupancy = double(n * blockSize) / p.maxThreadsPerMultiProcessor;
```

Պատասխանը հաշվվում է կոմպիլացված kernel-ի ռեգիստրների քանակից և shared memory-ի պահանջից՝ այս device-ի սահմանների դեմ։ Ոչինչ չի գործարկվում, ուստի սա residency-ի վերին սահմանն է, ոչ թե չափում. ասում է, թե քանի block **կարող էր** resident լինել, ոչ թե թե SM-ը իրականում որքան զբաղված էր։ Չափումը Nsight Compute-ի `sm__warps_active.avg.pct_of_peak_sustained_active`-ն է, և երկուսը տարբերվում են ամեն անգամ, երբ block-երն ավարտվում են տարբեր ժամանակ, կամ grid-ը շատ փոքր է device-ը լցնելու համար։

Երկու հարակից կանչ․

- `cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(..., unsigned int flags)` — նույն կանչը, `cudaOccupancyDefault`՝ սովորական վարքի համար, կամ `cudaOccupancyDisableCachingOverride`՝ platform-ից կախված caching-ի ճշգրտումը անջատելու համար։
- `cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, func, dynamicSMemSize, blockSizeLimit)` — հակառակ հարցը։ Ձեր ընտրած block size-ը գնահատելու փոխարեն վերադարձնում է այն block size-ը, որը տալիս է լավագույն հնարավոր occupancy, և ամենափոքր grid-ը, որով դրան հասնում է։ Հարմար է որպես մեկնարկային կետ, ոչ թե որպես պատասխան. լավագույն occupancy-ն և ամենաարագը նույն բանը չեն։

**Coalescing (memory coalescing)** — Երբ warp-ի 32 lane-երը դիմում են հաջորդական հասցեների, hardware-ը դրանք սպասարկում է մեկ 128-բայթանոց transaction-ով՝ մինչև 32 առանձինի փոխարեն։ Հատկությունը warp-ինն է, ոչ թե thread-ինը. կարևորը մեկ հրամանի ընդհանուր հետքն է 32 lane-երի վրայով, ոչ թե այն, թե մեկ thread-ը ժամանակի ընթացքում ինչ ճանապարհ է անցնում։

**Grid-stride loop** — Գործարկման ձև, որտեղ ֆիքսված քանակով thread-երից ամեն մեկը ցիկլով մշակում է մի քանի տարր՝ քայլելով thread-երի ընդհանուր քանակով, grid-ը տվյալների չափին հարմարեցնելու փոխարեն։ Ճիշտ է մուտքի ցանկացած չափի համար՝ առանց launch-ի չափերը վերահաշվելու։

## Ինչպես է block-ը տեղադրվում
Block-ը վերագրվում է մեկ SM-ի և մնում այնտեղ մինչև ավարտը։ Թե քանի block է միաժամանակ տեղավորվում SM-ում, որոշվում է միաժամանակ երեք սահմանափակմամբ՝ ռեգիստրներ մեկ thread-ում, shared memory մեկ block-ում, և resident block-երի hardware-ային սահման։ Հաղթում է ամենացածրը։ Հենց դրա համար block size-ը hardware-ի հարց է, ոչ թե ճաշակի։

## Պատկեր
![Block-երի grid, ամեն block՝ thread-երի երկչափ զանգված, կողքին՝ global index-ի բանաձևը](thread_hierarchy.svg)

Launch-ը ստեղծում է block-երի grid, և ամեն block ինքը thread-երի մեկ-, երկ- կամ եռաչափ զանգված է։ `blockIdx`-ը ասում է, թե thread-ը որ block-ում է, `threadIdx`-ը՝ որ տեղում այդ block-ի ներսում։ Գծապատկերի բանաձևը կրկնվում է գրեթե ամեն հաջորդ kernel-ում։

## Անիմացիա
![Global index-ի բանաձևը հաշվվում է չորս thread-ի համար երեք block-ի վրայով, ամեն մեկը blockIdx.x, blockDim.x և threadIdx.x-ից հասնում է կոնկրետ թվի](thread_indexing.svg)

Մեկ բանաձև, չորս կոնկրետ thread։ Միշտ `blockIdx.x * blockDim.x + threadIdx.x` է. փոխվում են միայն block-ի և thread-ի արժեքները։

![Չորս ֆիքսված thread անցնում են 16 տարրանոց զանգվածով՝ 4 քայլով, ամեն կրկնությանը ամեն thread վերցնում է այլ տարր](grid_stride_loop.svg)

Չորս thread չորս կրկնությամբ ծածկում է տասնվեց տարր։ Զանգվածը երեսուներկուսի հասցնելը տալիս է ութ կրկնություն՝ նույն launch configuration-ով։ Սա լաբորատոր առաջադրանքի grid-stride loop-ն է։

## Գրականություն
- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/pdf/CUDA_C_Programming_Guide.pdf) — 2. Programming Model · 5. Performance Guidelines
- CUDA C++ Best Practices Guide — Occupancy, Coalesced Access to Global Memory

## Լաբորատոր առաջադրանք
Վեկտորների գումարում, ապա նույն kernel-ը ընդլայնված 2D ինդեքսավորմամբ՝ երկու grayscale պատկերի վրա։

## Ինքնուրույն աշխատանք
1. Իրականացնել 1D վեկտորների գումարում մի քանի չափի զանգվածների համար, սահմանների ստուգմամբ այն չափերի համար, որոնք block size-ի բազմապատիկ չեն։
2. Ընդլայնել 2D ինդեքսավորմամբ և գումարել երկու grayscale պատկեր՝ պիքսել առ պիքսել։
3. Չափել 32, 64, 128 և 256 block size-երը։ Ամեն մեկի համար կանչել `cudaOccupancyMaxActiveBlocksPerMultiprocessor` և նշել, թե որտեղից է չափված ժամանակը դադարում հետևել occupancy-ի թվերին։
4. Իրականացնել նույն kernel-ը grid-stride loop-ով։ Ստուգել, որ այն ճիշտ է մնում `n`-ը `blocks * threads`-ից շատ ավելի մեծացնելուց հետո՝ առանց launch configuration-ը փոխելու։
5. Գրել երկու copy kernel՝ մեկը `blockIdx.x * blockDim.x + threadIdx.x` ինդեքսավորմամբ, մյուսը՝ `threadIdx.x * gridDim.x + blockIdx.x`։ Երկուսն էլ ճիշտ են։ Չափել երկուսը։
6. Ընտրած block size-ի համար ձեռքով հաշվել, թե քանի block է resident մեկ SM-ում, ապա ստուգել occupancy-ի API-ով։

## Ինքնաստուգում
Պատասխանները տրված չեն։

1. Ի՞նչ է սխալ գնում global index-ի բանաձևում, եթե բաց թողնեք `blockDim.x`-ը։
2. Ինչու՞ է grid-stride loop-ը ճիշտ մնում, երբ `n`-ը կրկնապատկվում է, իսկ մեկ thread մեկ տարրի kernel-ը՝ ոչ։
3. 5-րդ առաջադրանքի երկու բանաձևն էլ ճիշտ արդյունք են տալիս։ Ո՞ր չափումն է ասում, թե որը պահել։
4. Գործարկում եք `<<<100, 256>>>` 20 000 տարրի վրա՝ սահմանների ստուգմամբ։ Քանի՞ thread աշխատանք չի կատարում։

## Կոդի template
Տես [`template.cu`](template.cu)։
