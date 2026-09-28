# Օր 1. CUDA-ի ծրագրավորման մոդելը և GPU-ի ճարտարապետությունը

## Նպատակներ
- Հասկանալ host-ի և device-ի բաժանումը՝ հենվելով MPI-ի փորձի վրա։ Հիշողությունը երկու առանձին address space է, փոխանցումները գրվում են բացահայտ, coherence չի ապահովվում
- Նկարագրել CUDA-ի ծրագրավորման մոդելը՝ kernel, thread, block, grid
- Կոմպիլացնել և գործարկել `.cu` ֆայլ `nvcc`-ով։ Բացատրել, թե nvcc-ն ինչ է արտադրում՝ host-ի object code, PTX և SASS
- Կարդալ սեփական GPU-ի իրական թվերը՝ SM-ների քանակը, մեկ SM-ի ռեգիստրներն ու shared memory-ն, warp size-ը, հիշողության bus width-ը։ Այս թվերը պետք են ամբողջ դասընթացի ընթացքում
- Ստուգել CUDA-ի ամեն կանչը։ Բացատրել, թե ինչու է kernel-ի launch-ը պահանջում այլ ստուգում, քան մյուս կանչերը

## Հիմնական հասկացություններ
- Host և device. ինչու է GPU-ն throughput մեքենա, իսկ CPU-ն՝ latency մեքենա
- Kernel, thread, block, grid (միայն ակնարկ, մանրամասնը՝ Օր 2)
- Streaming multiprocessor՝ ռեգիստրներ, ALU, SFU, tensor core, warp scheduler, load/store unit
- `nvcc`՝ host-ի և device-ի բաժանում, PTX և SASS, virtual և real architecture-ի flag-եր
- Սխալների ստուգում՝ `CUDA_CHECK`, `CUDA_CHECK_LAST_ERROR`
- `cudaGetDeviceProperties` և `report_device_capabilities()`

## Սահմանումներ
**Host** — CPU-ն, ի տարբերություն **device**-ի (GPU)։

**Device** — GPU-ն, ի տարբերություն **host**-ի (CPU)։ Ունի իր հիշողությունը՝ VRAM, որին հասնում են PCIe-ով կամ NVLink-ով։

**Throughput մեքենա և latency մեքենա** — CPU-ն տարածք է ծախսում cache-երի, branch prediction-ի և out-of-order կատարման վրա, որպեսզի հրամանների մեկ հոսքն արագ լինի, այսինքն՝ latency-ն փոքրացնի։ GPU-ն նույն տարածքը ծախսում է execution unit-ների և register file-ի վրա, որպեսզի շատ warp պահի աշխատանքի մեջ։ Latency-ն հանելու փոխարեն այն հանդուրժում է։

**Kernel** — `__global__` նշված ֆունկցիա, որը host-ի կոդից գործարկվում է `<<<grid, block>>>` գրելաձևով և device-ի վրա կատարվում է շատ thread-երով զուգահեռ։

**SM (Streaming Multiprocessor)** — GPU-ի հաշվողական հիմնական միավորը։ Ժամանակակից GPU-ն ունի տասնյակից մինչև հարյուրից ավելի։ Ամեն block ամբողջությամբ մեկ SM-ի վրա է կատարվում։ Ձեր GPU-ի իրական քանակն ու սահմանները `report_device_capabilities()`-ում են։

**Warp scheduler** — SM-ի միավոր, որն ամեն ցիկլ ընտրում է մեկ պատրաստ (eligible) warp և նրա հաջորդ հրամանն ուղարկում execution unit-ներին։ SM-ում մի քանիսն է լինում։

**SFU (Special Function Unit)** — SM-ի միավորներ, որոնք հաշվում են transcendental ֆունկցիաներ՝ սինուս, կոսինուս, էքսպոնենտ, հակադարձ, հակադարձ քառակուսի արմատ։ Throughput-ը FP32 միավորներից ցածր է։

**Load/store unit** — SM-ի միավորներ, որոնք թողարկում են հիշողության հրամանները և հաշվում հասցեները global, local և shared memory-ի համար։

**nvcc** — CUDA-ի կոմպիլյատորի driver-ը։ `.cu` ֆայլը բաժանում է host-ի և device-ի կոդի, device-ի մասն ինքն է կոմպիլացնում, host-ի մասը տալիս է համակարգի կոմպիլյատորին, և երկուսը միացնում մեկ binary-ի մեջ։

**PTX** — NVIDIA-ի virtual GPU assembly-ն, որը համատեղելի է հետագա architecture-ների հետ։ nvcc-ն device-ի կոդը նախ PTX է դարձնում, ապա `ptxas`-ը PTX-ից հավաքում է կոնկրետ architecture-ի իրական մեքենայական կոդը՝ **SASS**։

**SASS** — Մեկ կոնկրետ GPU architecture-ի իրական մեքենայական կոդը (cubin), որը `ptxas`-ը հավաքում է PTX-ից։

**Virtual և real architecture** — `-arch=compute_XX`-ը նշում է virtual architecture-ը, որի համար PTX է գեներացվում, իսկ `-code=sm_XX`-ը՝ real architecture-ը, որի համար SASS է գեներացվում։ `-arch=sm_XX`-ը երկուսն էլ սահմանում է։

**Fat binary** — nvcc-ի արտադրած միակ գործարկվող ֆայլը, որում host-ի մեքենայական կոդի կողքին կա device-ի մեկ կամ մի քանի պատկեր՝ PTX, SASS կամ երկուսը։ Գործարկման պահին driver-ը վերցնում է համապատասխան SASS-ը, իսկ եթե այդպիսին չկա, JIT-ով կոմպիլացնում է ներդրված PTX-ը։

**Compute capability** — Տարբերակի համար, օրինակ `8.6`, որը նշում է GPU-ի architecture-ի սերունդը և հնարավորությունների հավաքածուն։ nvcc-ի flag-երում գրվում է `sm_XX` և `compute_XX` ձևով։

**`cudaGetDeviceProperties`** — API-ի կանչ, որը վերադարձնում է `cudaDeviceProp` կառուցվածք՝ device-ի SM-ների քանակով, warp size-ով, մեկ SM-ի ռեգիստրների և shared memory-ի սահմաններով, clock-երով, հիշողության bus width-ով և compute capability-ով։ `common/device_info.h`-ի `report_device_capabilities()`-ը տպում է այն։

## Ծանոթագրություն այս լսարանի համար
Մասնակիցներն արդեն գրում են OpenMP և MPI։ Երկու համեմատություն արժե բացահայտ անել. դրանք մոդելի մեջ մտնելու ամենակարճ ճանապարհն են։

- **OpenMP-ի համեմատ։** OpenMP-ի thread-ը պլանավորում է օպերացիոն համակարգը, և այն ունի լրիվ սեփական context։ CUDA-ի thread-ը warp-ի մեկ lane-ն է, և 32-ը կիսում են հրամանների մեկ հոսք։ Ուստի divergence-ը hardware-ի գին է, ոչ թե պլանավորման։
- **MPI-ի համեմատ։** Host-ի և device-ի հիշողությունները առանձին address space-եր են՝ բացահայտ փոխանցումներով, ինչը ծանոթ է։ Տարբերությունն այն է, որ փոխանցման գինը ցանցի գին չէ, այլ PCIe-ի կամ NVLink-ի։ Սովորաբար հենց դա է առաջին սահմանափակումը, երբ կոդը նոր է GPU տեղափոխվել։

## Պատկեր
![Host (CPU, քիչ բայց արագ core-եր, համակարգի RAM), միացած PCIe-ով կամ NVLink-ով device-ին (GPU, շատ core, VRAM)](host_device.svg)

Host-ը և device-ը երկու address space են՝ միացած մեկ կապով։ Հիշողության հատկացումը, փոխանցումը և launch-ը բոլորն անցնում են դրա վրայով։ Driver-ը նկարված չէ․ այն host-ի կողմում է և հենց այն է, որ `cudaMemcpy`-ը կամ launch-ը վերածում է այդ կապով անցնող տվյալների և GPU-ի հերթում դրվող հրամանների։

## Անիմացիա
![Տվյալները host-ի RAM-ից անցնում են device-ի VRAM, ապա kernel-ի launch, ապա արդյունքը վերադառնում է](host_device_transfer.svg)

Սովորական CUDA ծրագրի ցիկլը՝ copy in, launch, copy out։ Երկու copy-ն երկար են launch-ի համեմատ։ Օրեր 4, 6 և 8-ը հենց այդ անհամաչափության մասին են։

![`.cu` ֆայլը բաժանվում է device-ի ճանապարհի (nvcc-ի frontend, PTX, ptxas, SASS/cubin) և host-ի ճանապարհի (host-ի կոմպիլյատոր), որոնք linker-ում միանում են մեկ binary-ի](nvcc_toolchain.svg)

Device-ի ճանապարհը չորս քայլ է մինչև մեքենայական կոդի հայտնվելը, host-ինը՝ երկու, ապա սպասում է linker-ին, որովհետև binary-ն չի հավաքվի, քանի դեռ երկու կեսն էլ պատրաստ չեն։ PTX-ում է ամրագրվում virtual architecture-ը, SASS-ում՝ real-ը։

## Գրականություն
- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/pdf/CUDA_C_Programming_Guide.pdf) — 2. Programming Model · 4. Hardware Implementation · 16. Compute Capabilities
- CUDA C++ Best Practices Guide — Assess, Parallelize, Optimize, Deploy
- SM-ի կառուցվածքի գծապատկերը և անիմացիաները՝ [`sm_anatomy.svg`](../sm_anatomy.svg), [`sm_animations.html`](../sm_animations.html)
- [`ARCHITECTURE.md`](../ARCHITECTURE.md) — ինչ կա SM-ի ներսում, մանրամասն

## Լաբորատոր առաջադրանք
Գործարկել `report_device_capabilities()`-ը կլաստերի GPU-ի վրա և գրել թվերը։ Ապա գրել և գործարկել նվազագույն kernel, որը տպում է իր block-ի և thread-ի index-ը։ Նպատակը մաքուր կոմպիլացիա-գործարկում ցիկլն է, և այն hardware-ի թվերի գրանցումը, որի դեմ չափվելու է ամբողջ դասընթացը։

## Ինքնուրույն աշխատանք
1. Տպել `Hello from block X, thread Y` device-ի `printf`-ով, գործարկելով `<<<1,1>>>`, `<<<2,4>>>` և `<<<4,32>>>` կոնֆիգուրացիաներով։
2. Ամեն thread-ով գրել իր `blockIdx.x`-ը և `threadIdx.x`-ը երկու զանգվածի մեջ, պատճենել host և ստուգել launch configuration-ի դեմ։
3. `report_device_capabilities()`-ի ելքից հաշվել կլաստերի GPU-ի տեսական peak FP32 throughput-ը և peak memory bandwidth-ը։ Համեմատել NVIDIA-ի հրապարակած թվերի հետ և բացատրել տարբերությունը։
4. Կոմպիլացնել նույն kernel-ը `-arch=sm_75`-ով և `-arch=sm_90`-ով, երկուսի SASS-ը հանել `cuobjdump --dump-sass`-ով և diff անել։
5. Գործարկել 5000 thread մեկ block-ում՝ մեկ անգամ առանց `CUDA_CHECK_LAST_ERROR()`-ի, մեկ անգամ դրանով։ Նշել, թե ամեն գործարկումն ինչ է ցույց տալիս։

## Ինքնաստուգում
Պատասխանները տրված չեն։

1. Ինչու՞ kernel-ի launch-ը չի կարող `cudaError_t` վերադարձնել այնպես, ինչպես `cudaMalloc`-ը։
2. Ինչու՞ է nvcc-ն մեքենայական կոդից առաջ PTX արտադրում, և ի՞նչ է դա տալիս, երբ կլաստերը թարմացվում է։
3. Ձեր GPU-ն հայտնում է N SM և peak bandwidth B GB/s։ Այս երկուսից ո՞րն է սահմանափակելու վեկտորների գումարումը, և ինչպե՞ս եք դա իմանում մինչև գործարկելը։
4. OpenMP-ի thread-ը և CUDA-ի thread-ը երկուսն էլ «thread» են կոչվում։ Նշել երկու հատկություն, որոնք չեն փոխանցվում։

## Կոդի template
Տես [`template.cu`](template.cu)։

Մանրամասն օրինակ՝ [`example.md`](example.md) — նույն kernel-ը հետևված C++-ից PTX, ապա SASS, իրական կոմպիլյատորի ելքով, և ինչպես PTX ներդնել device-ի կոդի մեջ։
