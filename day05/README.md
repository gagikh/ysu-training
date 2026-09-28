# Օր 5. Shared memory-ն և bank conflict-ները

## Նպատակներ
- Բացատրել shared memory-ի bank-երի բաժանումը և ինչպես է առաջանում conflict-ը
- Ճիշտ սինխրոնացնել shared memory-ի հասանելիությունը, և ասել, թե `__syncthreads()`-ը ինչ է երաշխավորում և ինչ՝ ոչ
- Տարբերել global, shared, constant և pitched հիշողությունը և ընտրել դրանց միջև
- Իրականացնել tiled 2D ֆիլտր և հեռացնել նրա bank conflict-ները

## Հիմնական հասկացություններ
- 32 bank, առանց conflict-ի հասանելիություն, broadcast և N-way conflict-ներ
- `__syncthreads()`՝ ինչ է երաշխավորում, և ինչ արժե occupancy-ի առումով
- Shared memory-ն և L1-ը կիսում են նույն ֆիզիկական SRAM-ը, և բաժանումը կարգավորելի է
- Constant memory և broadcast-ի ուղին
- Pitched հիշողություն և ինչու է pitch-ը տարբերվում `width * sizeof(T)`-ից
- Tiling-ը որպես global memory-ի տրաֆիկի կրճատում

## Սահմանումներ
**Shared memory** — Բյուրեղի վրա գտնվող հիշողություն, հատկացվում է ամեն block-ի և կիսվում նրա thread-երի միջև։ Latency-ն ռեգիստրին մոտ է, կյանքի տևողությունը՝ block-ինը։ Հայտարարվում է `__shared__`-ով՝ ստատիկ կամ launch-ի երրորդ արգումենտով՝ դինամիկ։

**Bank** — Shared memory-ի 32 հավասար բաժանումներից մեկը։ Հաջորդական 32-բիթանոց բառերն ընկնում են հաջորդական bank-երում, և ամեն bank մեկ ցիկլում սպասարկում է մեկ բառ։

**Bank conflict** — Երբ warp-ի մի քանի thread մեկ transaction-ում դիմում են shared memory-ի հասցեների, որոնք նույն bank-ին են համապատասխանում, այդ դիմումները սերիականացվում են՝ զուգահեռ սպասարկվելու փոխարեն։ Լուծվում է padding-ով (այս օրը) կամ ինդեքսի swizzling-ով (Օր 6)։

**Broadcast** — Երբ warp-ի բոլոր lane-երը կարդում են shared memory-ի նույն բառը, hardware-ը դրանք սպասարկում է մեկ transaction-ով։ Սա conflict չէ։

**N-way conflict** — Երբ warp-ի N lane դիմում են նույն bank-ի տարբեր բառերի, դիմումը բաժանվում է N transaction-ի։

**`__syncthreads()`** — Block-ի barrier։ Ոչ մի thread այն չի անցնում, մինչև block-ի բոլոր thread-երը հասնեն, և մինչ այն կատարված shared ու global գրառումները դրանից հետո տեսանելի են block-ին։ Block-ի ամեն thread պետք է հասնի դրան, ուստի divergent control flow-ի ներսում դնելը undefined behaviour է։

**Barrier** — Կետ, որին բոլոր մասնակից thread-երը պետք է հասնեն, մինչև որևէ մեկը շարունակի։ `__syncthreads()`-ը block-ի barrier-ն է, `__syncwarp`-ը՝ warp-ինը։

**Constant memory** — 64 ԿԲ միայն-կարդալու տիրույթ, հայտարարվում է `__constant__`-ով, cache-վում է ամեն SM-ում։ Երբ warp-ի բոլոր lane-երը կարդում են նույն հասցեն, սպասարկվում է որպես մեկ broadcast, իսկ տարբեր հասցեները սերիականացվում են։

**Tiling** — Տվյալների մի կտոր մեկ անգամ shared memory տեղափոխել և չիպի վրա շատ անգամ կարդալ, ինչը կրճատում է global տրաֆիկը մոտավորապես կրկնակի օգտագործման գործակցի չափով։ Այս տեխնիկան է tiled matrix multiply-ի և այս դասընթացի ամեն stencil ու ֆիլտր kernel-ի հիմքում։

**Halo** — Եզրային տարրերը, որոնք tile-ին պետք են, բայց իրենը չեն. R շառավղով ֆիլտրի դեպքում՝ tile-ի շուրջ R տող և սյուն։ Դրանք պետք է tile-ի հետ միասին բեռնվեն shared memory։

**Pitch** — 2D հատկացման տողերի միջև իրական բայթային քայլը (`cudaMallocPitch`), սովորաբար `width * elementSize`-ից մեծ՝ հավասարեցման լրացման պատճառով։ Pitched հիշողությանը դիմող kernel-ները տողերը պետք է ինդեքսավորեն pitch-ով, ոչ թե width-ով։

## Պատկեր
![Առանց conflict-ի shared memory դիմում, որտեղ ամեն thread այլ bank է ընկնում, կողքին՝ bank conflict, որտեղ stride-32-ի դեպքում մի քանի thread ընկնում է bank 0](bank_conflicts.svg)

Shared memory-ն բաժանված է 32 bank-ի, որպեսզի warp-ը սպասարկվի մեկ transaction-ով, պայմանով որ ամեն lane այլ bank է ընկնում։ 32-ի բազմապատիկ քայլը — հենց այն, ինչ տալիս է 32 լայնությամբ tile-ով ինդեքսավորումը — ամեն ինչ կենտրոնացնում է մեկ bank-ում և սերիականացնում։ Տողի քայլը մեկ տարրով լրացնելը ստանդարտ լուծումն է, և հենց դրա համար է պատրաստված [`template.cu`](template.cu)-ի `tiled_filter`-ը։

![Tiled matrix multiplication՝ A-ի տողային tile-ը և B-ի սյունային tile-ը մեկ անգամ բեռնվում են shared memory և ամբողջ block-ը կրկին օգտագործում է դրանք C-ի մեկ ելքային tile-ը հաշվելու համար](tiled_matmul.svg)

Ընդլայնված առաջադրանքի համար։ Naive kernel-ը A-ի և B-ի նույն տողերն ու սյուները global memory-ից նորից է կարդում՝ ամեն ելքային տարրի համար մեկ անգամ։ Tiled տարբերակը ամեն block-ում մեկական tile բեռնում է shared memory, և block-ի ամեն thread կրկին օգտագործում է այն։ Նույն tiling-ը, ինչ ֆիլտրում, կիրառված matrix multiply-ի վրա։

## Անիմացիա
![32 lane միացած 32 bank-ի, լարերը վերադասավորվում են հինգ դիմումի ձևով՝ զուգահեռ stride 1-ի դեպքում, զույգերով միանալով stride 2-ին, չորսական՝ stride 4-ին, ամբողջովին bank 0 ընկնելով stride 32-ին, և երեքական մեկ bank-ում gather-ի դեպքում](bank_conflict_nway.svg)

Մեկ լար ամեն lane-ի համար՝ ցույց տալով, թե իրականում որ bank է ընկնում։ Նույն bank հասնող ամեն հավելյալ լար ևս մեկ սերիականացված transaction է. bank-ը մեկ ցիկլում սպասարկում է մեկ բառ, ուստի ամբողջ warp-ը սպասում է ամենազբաղվածին։

Կանոնը՝ lane `t`-ն կարդում է `t x stride` բառը, որը գտնվում է `(t x stride) mod 32` bank-ում, ուստի conflict-ի աստիճանը `gcd(stride, 32)` է։ Երկու հետևանք․

- Ամեն կենտ stride առանց conflict-ի է։ Stride 3, 7, 17 և 31-ը արժենում են մեկ transaction՝ ինչպես stride 1-ը։ Ավելի թանկ են միայն այն քայլերը, որոնք 32-ի հետ երկուսի ընդհանուր բազմապատիկ ունեն, և ամեն հավելյալ գործակից գինը կրկնապատկում է։ Tile-ը `[TILE][TILE+1]` դարձնելը զույգ տողային քայլը դարձնում է կենտ. ուրիշ ոչինչ չի անում։
- Հաստատուն քայլից աստիճանը միշտ երկուսի աստիճան է, քանի որ `gcd(s, 32)`-ը 32-ի բաժանարար է։ Հաստատուն քայլից եռակի conflict չի ստացվում։ Կենտ աստիճանների համար պետք է անկանոն ձև՝ անուղղակի gather `s[idx[t]]`, lookup table, compaction — գծապատկերի հինգերորդ փուլն է։ Գործնական տարբերությունը ախտորոշման մեջ է. կանոնավոր tiled kernel-ի համար `gcd(row_stride, 32)`-ը հարցը լուծում է թղթի վրա, gather-ի համար՝ միայն profiler-ը (`ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared`)։

Կարգավորելի տարբերակը՝ ցանկացած stride 1-ից 33, ցանկացած gather-ի աստիճան, ամեն lane-ի լարի հետագծում, և transpose-ի կարդալու ու գրելու փուլերը plain, padded ու swizzled դասավորությունների տակ — [`bank_conflict_animations.html`](bank_conflict_animations.html)։ Բացել տեղում, browser-ով։

## Գրականություն
- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/pdf/CUDA_C_Programming_Guide.pdf) — 3.2.4. Shared Memory
- CUDA C++ Best Practices Guide — Shared Memory
- Bank conflict-ի գծապատկերները՝ վերևի `Պատկեր` և `Անիմացիա` բաժիններում. stride-ի հետազոտիչը՝ [`bank_conflict_animations.html`](bank_conflict_animations.html)

## Լաբորատոր առաջադրանք
Shared memory-ով tiled 2D ֆիլտր իրական պատկերի վրա, ապա Sobel ֆիլտր։

## Ինքնուրույն աշխատանք
1. Իրականացնել tile-ի վրա հիմնված 2D convolution (սկսել box blur-ից) halo տիրույթով։
2. Դիտմամբ ստեղծել bank conflict-ներով shared memory դիմումի ձև, չափել գինը, ապա հեռացնել padding-ով։
3. Իրականացնել 2D Sobel ֆիլտր shared memory-ով։
4. Ձեր tile-ի համար ձեռքով հաշվել, թե warp-ը քանի bank conflict է կրում, ապա ստուգել թիվը Nsight Compute-ում։
5. Փոխել tile-ի չափը և գրանցել այն կետը, որտեղ occupancy-ն սահմանափակում է ոչ թե թվաբանությունը, այլ մեկ block-ի shared memory-ն։

## Ինքնաստուգում
Պատասխանները տրված չեն։

1. Ինչու՞ են ֆիքսված `k`-ի դեպքում `tile[threadIdx.x][k]` կարդացող 32 thread-երը բախվում մեկ bank-ում։
2. Warp-ը դիմում է shared memory-ին 3 քայլով։ Քանի՞ transaction է դա արժենում, և ինչու պատասխանը 3 չէ։
3. Ի՞նչ է `__syncthreads()`-ը բացահայտորեն չերաշխավորում։
4. Tile-ը մեկ սյունով լրացնելը shared memory է վատնում։ Ինչու՞ է դա սովորաբար ճիշտ փոխզիջում։

## Կոդի template
Տես [`template.cu`](template.cu)։
