# Օր 1. CUDA-ի ծրագրավորման մոդելը և GPU-ի ճարտարապետությունը

## Նպատակներ
- Հասկանալ host-ի և device-ի բաժանումը՝ հենվելով MPI-ի փորձի վրա։ Հիշողությունը բաժանված է երկու առանձին address space-ի, փոխանցումները գրվում են բացահայտ, coherence չի ապահովվում
- Նկարագրել CUDA-ի ծրագրավորման մոդելը՝ kernel, thread, block, grid
- Կոմպիլացնել և գործարկել `.cu` ֆայլ `nvcc`-ով։ Բացատրել, թե ինչ է ստեղծում nvcc-ն՝ host-ի object code, PTX և SASS
- Պարզել սեփական GPU-ի իրական պարամետրերը՝ SM-ների քանակը, մեկ SM-ի ռեգիստրներն ու shared memory-ն, warp size-ը, հիշողության bus width-ը։ Այս արժեքներն օգտագործվելու են ամբողջ դասընթացում
- Ստուգել CUDA-ի ամեն կանչը։ Բացատրել, թե ինչու է kernel-ի launch-ը պահանջում այլ ստուգում, քան մյուս կանչերը

## Հիմնական հասկացություններ
- Host և device. ինչու է GPU-ն throughput մեքենա, իսկ CPU-ն՝ latency մեքենա
- Kernel, thread, block, grid (միայն ակնարկ, մանրամասները՝ Օր 2-ում)
- Streaming multiprocessor՝ ռեգիստրներ, ALU, SFU, tensor core, warp scheduler, load/store unit
- `nvcc`՝ host-ի և device-ի կոդի բաժանում, PTX և SASS, virtual և real architecture-ի flag-եր
- Սխալների ստուգում՝ `CUDA_CHECK`, `CUDA_CHECK_LAST_ERROR`
- `cudaGetDeviceProperties` և `report_device_capabilities()`

## Սահմանումներ
**Host** — CPU-ն, ի տարբերություն **device**-ի (GPU)։

**Device** — GPU-ն, ի տարբերություն **host**-ի (CPU)։ Ունի սեփական հիշողություն՝ VRAM, որին host-ը դիմում է PCIe-ով կամ NVLink-ով։

**Throughput մեքենա և latency մեքենա** — CPU-ն չիպի մակերեսը ծախսում է cache-երի, branch prediction-ի և out-of-order կատարման վրա, որպեսզի հրամանների մեկ հոսքն արագ կատարվի, այսինքն՝ latency-ն փոքր լինի։ GPU-ն նույն մակերեսը ծախսում է execution unit-ների և register file-ի վրա, որպեսզի միաժամանակ շատ warp պահի աշխատանքի մեջ։ Latency-ն վերացնելու փոխարեն GPU-ն այն հանդուրժում է։

**Kernel** — `__global__`-ով նշված ֆունկցիա, որը գործարկվում է host-ի կոդից `<<<grid, block>>>` գրելաձևով և device-ի վրա կատարվում է զուգահեռ՝ շատ thread-երով։

**SM (Streaming Multiprocessor)** — GPU-ի հիմնական հաշվողական միավորը։ Ժամանակակից GPU-ում դրանք տասնյակներով են, երբեմն՝ հարյուրից ավելի։ Ամեն block ամբողջությամբ կատարվում է մեկ SM-ի վրա։ Ձեր GPU-ի SM-ների իրական քանակը և սահմանները տպում է `report_device_capabilities()`-ը։

**Warp scheduler** — SM-ի միավոր, որն ամեն ցիկլ ընտրում է մեկ պատրաստ (eligible) warp և նրա հաջորդ հրամանն ուղարկում execution unit-ներին։ Մեկ SM-ում դրանք մի քանիսն են։

**SFU (Special Function Unit)** — SM-ի միավորներ, որոնք հաշվում են տրանսցենդենտ ֆունկցիաներ՝ սինուս, կոսինուս, էքսպոնենտ, հակադարձ արժեք, քառակուսի արմատի հակադարձ։ Դրանց throughput-ը ցածր է FP32 միավորների throughput-ից։

**Load/store unit** — SM-ի միավորներ, որոնք ուղարկում են հիշողության հրամանները և հաշվում հասցեները global, local և shared memory-ի համար։

**nvcc** — CUDA-ի կոմպիլյատորը (compiler driver)։ `.cu` ֆայլից առանձնացնում է host-ի և device-ի կոդը, device-ի մասն ինքն է կոմպիլացնում, host-ի մասը փոխանցում է համակարգի կոմպիլյատորին, ապա երկուսը միավորում է մեկ binary-ում։

**PTX** — NVIDIA-ի virtual GPU assembly-ն, որը համատեղելի է հետագա architecture-ների հետ։ nvcc-ն device-ի կոդը նախ դարձնում է PTX, ապա `ptxas`-ը PTX-ից հավաքում է կոնկրետ architecture-ի իրական մեքենայական կոդը՝ **SASS**։

**SASS** — Մեկ կոնկրետ GPU architecture-ի իրական մեքենայական կոդը (cubin), որը `ptxas`-ը հավաքում է PTX-ից։

**Virtual և real architecture** — `-arch=compute_XX`-ը նշում է virtual architecture-ը, որի համար ստեղծվում է PTX-ը, իսկ `-code=sm_XX`-ը՝ real architecture-ը, որի համար ստեղծվում է SASS-ը։ `-arch=sm_XX`-ը սահմանում է երկուսն էլ։

**Fat binary** — Այն մեկ գործարկվող ֆայլը, որ ստեղծում է nvcc-ն։ Host-ի մեքենայական կոդի կողքին այն պարունակում է device-ի կոդի մեկ կամ մի քանի տարբերակ՝ PTX, SASS կամ երկուսը։ Գործարկման պահին driver-ը վերցնում է համապատասխան SASS-ը, իսկ եթե այդպիսին չկա, JIT-ով կոմպիլացնում է ներդրված PTX-ը։

**Compute capability** — Տարբերակի համար, օրինակ՝ `8.6`, որը նշում է GPU-ի architecture-ի սերունդը և հնարավորությունների հավաքածուն։ nvcc-ի flag-երում գրվում է `sm_XX` և `compute_XX` ձևով։

**`cudaGetDeviceProperties`** — API-ի կանչ, որը լրացնում է `cudaDeviceProp` կառուցվածքը device-ի պարամետրերով՝ SM-ների քանակ, warp size, մեկ SM-ի ռեգիստրների և shared memory-ի սահմաններ, clock-եր, հիշողության bus width, compute capability։ `common/device_info.h`-ի `report_device_capabilities()`-ը տպում է այս կառուցվածքը։

## Ֆունկցիաներ

```c
// Հատկացնում է size բայթ device-ի հիշողության մեջ և հասցեն գրում *devPtr-ում
cudaError_t cudaMalloc(void **devPtr, size_t size);

// Ազատում է cudaMalloc-ով հատկացված հիշողությունը
cudaError_t cudaFree(void *devPtr);

// Պատճենում է count բայթ src-ից dst-ի մեջ։ Ուղղությունը տալիս է kind-ը, օրինակ՝
// cudaMemcpyHostToDevice։ Վերադառնում է պատճենումն ավարտվելուց հետո
cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind);

// Սպասում է, մինչև device-ի ամբողջ աշխատանքն ավարտվի
cudaError_t cudaDeviceSynchronize(void);

// Վերադարձնում է վերջին սխալը և զրոյացնում այն։ Կանչվում է kernel-ի launch-ից հետո
cudaError_t cudaGetLastError(void);

// Լրացնում է *prop-ը device համարով GPU-ի պարամետրերով
cudaError_t cudaGetDeviceProperties(struct cudaDeviceProp *prop, int device);

// Device-ի կոդից տպում է ծրագրի stdout-ում։ Տեքստը երևում է սինխրոնացումից հետո,
// օրինակ՝ cudaDeviceSynchronize()-ից
int printf(const char *format, ...);

// common/device_info.h։ Տպում է device 0-ի պարամետրերը stderr-ում և վերադարձնում նրա անունը
std::string report_device_capabilities();
```

## Ծանոթագրություն այս լսարանի համար
Մասնակիցներն արդեն ծրագրավորում են OpenMP-ով և MPI-ով։ Արժե բացահայտ կատարել երկու համեմատություն. դրանք մոդելը հասկանալու ամենակարճ ճանապարհն են։

- **OpenMP-ի համեմատ։** OpenMP-ի thread-ը պլանավորում է օպերացիոն համակարգը, և յուրաքանչյուրն ունի սեփական ամբողջական context։ CUDA-ի thread-ը warp-ի մեկ lane է, և 32 այդպիսի thread կիսում են հրամանների մեկ հոսքը։ Ուստի divergence-ը hardware-ի ծախս է, ոչ թե պլանավորման։
- **MPI-ի համեմատ։** Host-ի և device-ի հիշողությունները առանձին address space-եր են՝ բացահայտ փոխանցումներով, ինչը ծանոթ է։ Տարբերությունն այն է, որ փոխանցման ծախսը ցանցինը չէ, այլ PCIe-ինը կամ NVLink-ինը։ Սովորաբար հենց դա է առաջին սահմանափակումը, երբ կոդը նոր է տեղափոխվել GPU-ի վրա։

## Պատկեր
![Host (CPU, քիչ բայց արագ core-եր, համակարգի RAM), միացած PCIe-ով կամ NVLink-ով device-ին (GPU, շատ core, VRAM)](host_device.svg)

Host-ը և device-ը երկու address space են՝ միացած մեկ կապով։ Հիշողության հատկացումը, փոխանցումը և launch-ը անցնում են այդ կապով։ Driver-ը նկարում ցույց տրված չէ. այն աշխատում է host-ի կողմում, և հենց նա է `cudaMemcpy`-ն ու launch-ը վերածում կապով անցնող տվյալների և GPU-ի հերթում դրվող հրամանների։

## Անիմացիա
![Տվյալները host-ի RAM-ից անցնում են device-ի VRAM, ապա կատարվում է kernel-ի launch-ը, ապա արդյունքը վերադառնում է host](host_device_transfer.svg)

Սովորական CUDA ծրագրի ցիկլը՝ copy in, launch, copy out։ Երկու պատճենումը launch-ից շատ ավելի երկար են տևում։ Օր 4-ը, Օր 6-ը և Օր 8-ը նվիրված են հենց այդ անհամաչափությանը։

![`.cu` ֆայլը բաժանվում է երկու ճանապարհի՝ device-ի (nvcc-ի frontend, PTX, ptxas, SASS/cubin) և host-ի (host-ի կոմպիլյատոր), որոնք linker-ում միավորվում են մեկ binary-ում](nvcc_toolchain.svg)

Device-ի ճանապարհին մինչև մեքենայական կոդը չորս քայլ է, host-ի ճանապարհին՝ երկու։ Հետո host-ի մասը սպասում է linker-ում, որովհետև binary-ն չի կարող հավաքվել, քանի դեռ երկու մասն էլ պատրաստ չեն։ PTX-ում է ամրագրվում virtual architecture-ը, SASS-ում՝ real-ը։

## Գրականություն
- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/pdf/CUDA_C_Programming_Guide.pdf) — 2. Programming Model · 4. Hardware Implementation · 16. Compute Capabilities
- CUDA C++ Best Practices Guide — Assess, Parallelize, Optimize, Deploy
- SM-ի կառուցվածքի գծապատկերը և անիմացիաները՝ [`sm_anatomy.svg`](../sm_anatomy.svg), [`sm_animations.html`](../sm_animations.html)
- [`ARCHITECTURE.md`](../ARCHITECTURE.md) — ինչ կա SM-ի ներսում, մանրամասն

## Լաբորատոր առաջադրանք
Գործարկել `report_device_capabilities()`-ը կլաստերի GPU-ի վրա և գրանցել ստացված արժեքները։ Ապա գրել և գործարկել նվազագույն kernel, որը տպում է իր block-ի և thread-ի index-ը։ Նպատակն է առանց սխալների կոմպիլացնել և գործարկել ծրագիրը, և գրանցել hardware-ի այն արժեքները, որոնց հետ համեմատվելու են դասընթացի բոլոր չափումները։

## Ինքնուրույն աշխատանք
1. Device-ի `printf`-ով տպել `Hello from block X, thread Y`՝ kernel-ը գործարկելով `<<<1,1>>>`, `<<<2,4>>>` և `<<<4,32>>>` կոնֆիգուրացիաներով։
2. Թող ամեն thread իր `blockIdx.x`-ը և `threadIdx.x`-ը գրի երկու զանգվածում։ Զանգվածները պատճենել host և ստուգել, որ համապատասխանում են launch configuration-ին։
3. `report_device_capabilities()`-ի ելքից հաշվել կլաստերի GPU-ի տեսական peak FP32 throughput-ը և peak memory bandwidth-ը։ Համեմատել NVIDIA-ի հրապարակած թվերի հետ և բացատրել տարբերությունը։
4. Կոմպիլացնել նույն kernel-ը `-arch=sm_75`-ով և `-arch=sm_90`-ով, երկուսի SASS-ը ստանալ `cuobjdump --dump-sass`-ով և համեմատել `diff`-ով։
5. Kernel-ը գործարկել 5000 thread-անոց block-ով՝ մեկ անգամ առանց `CUDA_CHECK_LAST_ERROR()`-ի, մեկ անգամ դրանով։ Նշել, թե ինչ է ցույց տալիս ամեն գործարկումը։

## Ինքնաստուգում
Պատասխանները տրված չեն։

1. Ինչու՞ kernel-ի launch-ը չի կարող `cudaError_t` վերադարձնել այնպես, ինչպես `cudaMalloc`-ը։
2. Ինչու՞ է nvcc-ն մեքենայական կոդից առաջ ստեղծում PTX, և ի՞նչ է դա տալիս, երբ կլաստերը թարմացվում է։
3. Ձեր GPU-ն ունի N SM և B GB/s peak bandwidth։ Այս երկուսից ո՞րն է սահմանափակելու վեկտորների գումարումը, և ինչպե՞ս կարելի է դա իմանալ մինչև գործարկելը։
4. OpenMP-ի thread-ը և CUDA-ի thread-ը երկուսն էլ կոչվում են «thread»։ Նշել երկու հատկություն, որոնցով դրանք տարբերվում են։

## Կոդի template
Տես [`template.cu`](template.cu)։

Մանրամասն օրինակ՝ [`example.md`](example.md). մեկ kernel՝ C++-ից մինչև PTX և SASS, իրական կոմպիլյատորի ելքով, և ինչպես ներդնել PTX device-ի կոդում։
