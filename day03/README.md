# Օր 3. SIMT կատարումը, warp-երը, divergence-ը և latency hiding-ը

## Նպատակներ
- Բացատրել SIMT-ը՝ մեկ հրաման, որ մեկ անգամ է կարդացվում ու վերծանվում և ուղարկվում 32 lane-երի
- Նկարագրել հրամանների pipeline-ը և բացատրել, թե ինչու են register read-ը և memory-ն առանձին փուլեր
- Բացատրել, թե ինչպես է warp scheduler-ը թաքցնում latency-ն՝ անցնելով resident warp-երի միջև, և ինչու է դա փոխարինում այն մեծ cache-երին, որ CPU-ն ունի
- Ճանաչել և խուսափել branch divergence-ից. բացատրել, թե ինչու է այն սերիականացնում, ոչ թե զուգահեռացնում
- Կիրառել loop unrolling այնտեղ, որտեղ օգուտ է տալիս

## Հիմնական հասկացություններ
- SIMT և հրամանների pipeline՝ fetch, decode, warp scheduler, մեկ հրաման՝ ուղարկված բոլոր 32 lane-երին
- Warp-ի ձևավորումը `threadIdx`-ից. ինչու են 32-ի բազմապատիկ չհանդիսացող block size-երը lane-եր վատնում
- Eligible, active և stalled warp-եր. ինչ է նշանակում stall reason-ը profiler-ում
- Branch divergence և reconvergence
- Loop unrolling

## Սահմանումներ
**Warp** — Block-ի ներսում 32 thread-ից բաղկացած խումբ, որը hardware-ը պլանավորում և կատարում է միասին. մեկ հրաման միաժամանակ ուղարկվում է բոլոր 32-ին, ուստի ցանկացած պահի բոլորը նույն հրամանի վրա են։ Warp մակարդակի intrinsic-ները հենց այս միավորի վրա են աշխատում։

**SIMT (Single Instruction, Multiple Threads)** — NVIDIA-ի կատարման մոդելը՝ մեկ հրաման մեկ անգամ է կարդացվում ու վերծանվում և միաժամանակ ուղարկվում warp-ի բոլոր 32 thread-երին։

**Հրամանների pipeline** — Փուլերը, որոնցով անցնում է հրամանը՝ fetch, decode, register read, execute, memory, writeback։ Միաժամանակ մի քանի հրաման է շարժման մեջ՝ մեկը ամեն փուլում։

**Register file** — Մեկ SM-ի պահոց, որից հատկացվում են բոլոր thread-երի ռեգիստրները։ Չափը ֆիքսված է, ուստի մեկ thread-ի ռեգիստրները և resident warp-երի քանակը մրցում են միմյանց հետ։

**Eligible, active և stalled warp** — Active, այսինքն՝ resident warp-ը զբաղեցնում է SM-ի warp slot։ Այն **eligible** է՝ պատրաստ, երբ կարող է ուղարկել հաջորդ հրամանը։ Դրա համար երեք պայման է պետք․ հրամանն արդեն վերծանված է, նրա կարդացած բոլոր արժեքները հասանելի են, այսինքն՝ նախորդ load-ը, որից այն կախված է, վերադարձել է, և այն կատարող սարքը՝ FP32 pipe-ը, load/store unit-ը կամ tensor core-ը, այդ ցիկլում ազատ է։ Հակառակ դեպքում warp-ը stalled է։ Ամեն ցիկլ scheduler-ը ընտրում է մեկ warp, որից ուղարկի, և թեկնածու են միայն պատրաստ warp-երը։

**Stall reason** — Profiler-ի դասակարգումը, թե ինչու warp-ը պատրաստ չէր։ NVIDIA-ն առանձնացնում է չորս խումբ՝ warp-ը սպասում է հրամանի կարդացմանը (instruction fetch), հիշողության կախվածությանը (memory dependency), կատարման կախվածությանը (execution dependency) կամ սինխրոնացման barrier-ին։ Հենց այն է անվանում, ինչ պետք է ուղղել։

**Latency hiding** — GPU-ի արագագործության ռազմավարությունը. երբ warp-ը կանգ է առնում, warp scheduler-ը նույն ցիկլում ուղարկում է այլ, պատրաստ warp-ի հրամանը՝ pipeline-ը պարապ թողնելու փոխարեն։ Հենց դրա համար GPU-ն նախընտրում է շատ thread, ոչ թե քիչ ու արագ։

**Divergence (warp divergence)** — Երբ մեկ warp-ի thread-երը branch-ում տարբեր ճանապարհներ են ընտրում։ Քանի որ 32 thread-ը կիսում են հրամանների մեկ հոսք, hardware-ը ամեն ճանապարհ առանձին է անցնում՝ մի մասի lane-երը փակելով, զուգահեռ կատարելու փոխարեն։

**Reconvergence** — Divergent branch-ից հետո այն կետը, որտեղ warp-ի բոլոր lane-երը նորից նույն հրամանն են կատարում։ Volta-ից սկսած lane-երն ունեն անկախ program counter, և branch-ի վերջում reconvergence-ը երաշխավորված չէ. `__syncwarp`-ը այն դարձնում է բացահայտ։

**Loop unrolling** — Ցիկլը փոխարինել իր մարմնի կրկնվող պատճեններով, ինչը հեռացնում է branch-ի ու ինդեքսի հրամանները և scheduler-ին բացում անկախ գործողություններ։ Կառավարվում է `#pragma unroll`-ով։

## Ինչու է latency hiding-ը ճարտարապետության կենտրոնում
CPU-ն հիշողության latency-ն փոքրացնում է խորը cache-երի հիերարխիայով և speculation-ով։ GPU-ն հիմնականում դա չի անում՝ փոխարենը հանդուրժում է latency-ն։ Երբ warp-ը կանգ է առնում load-ի վրա, scheduler-ը նույն ցիկլում ուղարկում է այլ resident warp-ի հրամանը։ Այս մեկ որոշումը բացատրում է occupancy-ն, block size-ի կարևորությունը, divergence-ի բարձր գինը և այն, թե ինչու է arithmetic intensity-ն որոշում արագագործությունը։ Դասընթացի մնացած ամեն ինչը դրա հետևանքն է։

## Պատկեր
![SIMT-ի հրամանների pipeline՝ fetch, decode, warp scheduler, ապա մեկ հրաման՝ ուղարկված միաժամանակ warp-ի բոլոր 32 lane-երին](pipeline.svg)

Մեկ հրաման մեկ անգամ է կարդացվում ու վերծանվում, ապա միաժամանակ ուղարկվում warp-ի բոլոր 32 thread-երին։ Հենց սա է divergence-ի գնի պատճառը. երբ lane-երը branch-ում համաձայն չեն, hardware-ը փակում է lane-երի մի մասը և ամեն ճանապարհ անցնում հերթով։

## Անիմացիա
![Վեց փուլանոց pipeline՝ fetch, decode, register read, execute, memory, writeback, չորս հրաման շարժման մեջ, ամեն մեկը նախորդից մեկ փուլ հետ](pipeline_timeline.svg)

Նույն pipeline-ը՝ ժամանակի մեջ։ Register read-ը և memory-ն առանձին փուլեր են, որովհետև և՛ register file-ը, և՛ global memory-ն սահմանափակ ռեսուրսներ են՝ իրական դիմումի latency-ով։ Երբ warp-ը կանգ է առնում memory փուլում, scheduler-ը այդ ցիկլում ուղարկում է այլ warp-ի հրաման՝ փուլը պարապ թողնելու փոխարեն։

![Warp scheduler-ը շրջանցում է վեց resident warp. token-ը գնում է scheduler, ապա execution unit-ներ, և հենց մեկը պարապում է, ուղարկվում է նորը](warp_scheduling.svg)

Հենց մի warp դադարում է պատրաստ լինելուց, նրա տեղը զբաղեցնում է մյուս resident warp-ը։ Սա է latency hiding-ը, և occupancy-ն չափում է, թե որքան է դրանից հասանելի։

![32 thread-անոց warp-ը բաժանվում է branch-ի վրա. thread 0–15-ը անցնում են A ճանապարհով, 16–31-ը փակ են, ապա հակառակը, ապա բոլոր 32-ը վերամիավորվում են](warp_divergence.svg)

Երկու ճանապարհն անցնում է հերթով։ Divergence-ը սերիականացնում է warp-ը, զուգահեռություն չի ավելացնում։

Քայլ առ քայլ տարբերակը՝ [`warp_animations.html`](warp_animations.html)։ Բացել տեղում, browser-ով. GitHub-ի դիտիչը HTML-ը ցույց է տալիս որպես տեքստ, չի գործարկում։

## Գրականություն
- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/pdf/CUDA_C_Programming_Guide.pdf) — 4. Hardware Implementation · 5. Performance Guidelines
- [Nsight Compute Kernel Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html) — 2.3.1. Hardware Model. Այստեղից են վերևում օգտագործված active, eligible և stalled warp վիճակները
- Oxford-ի CUDA դասընթաց, lecture 3: https://people.maths.ox.ac.uk/~gilesm/cuda/lecs/lec3.pdf
- Using CUDA warp-level primitives: https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/
- Pipeline-ի և warp scheduling-ի գծապատկերները՝ վերևի `Պատկեր` և `Անիմացիա` բաժիններում. ինտերակտիվ տարբերակը՝ [`warp_animations.html`](warp_animations.html)

## Լաբորատոր առաջադրանք
Վեկտորների գումարում՝ ժամանակով համեմատված համարժեք CPU ցիկլի հետ, ապա BGR-ից grayscale փոխակերպման kernel։

## Ինքնուրույն աշխատանք
1. Իրականացնել և չափել վեկտորների գումարումը CPU ցիկլի դեմ։
2. Kernel-ում BGR-ը դարձնել grayscale (`gray = 0.114*B + 0.587*G + 0.299*R`)։
3. Դիտմամբ ներմուծել divergence (`if (threadIdx.x % 2 == 0)`) և չափել գինը՝ առանց divergence-ի տարբերակի դեմ։
4. Կրկնել 3-րդ առաջադրանքը, բայց branch-ը կառուցել `threadIdx.x / 32`-ի վրա։ Բացատրել արդյունքի տարբերությունը։
5. Կիրառել `#pragma unroll` փոքր, ֆիքսված քանակով կրկնությունների ցիկլի վրա և համեմատել ոչ միայն ժամանակը, այլև գեներացված SASS-ը։

## Ինքնաստուգում
Պատասխանները տրված չեն։

1. Ինչու՞ է divergence-ը թանկ, եթե ամեն thread ի վերջո կատարում է իր օգտակար աշխատանքը։
2. Ինչու՞ են register read-ը և memory-ն առանձին pipeline փուլեր, ոչ թե execute-ի մեջ միացված։
3. Warp-ի կեսը գնում է `if`-ով, կեսը՝ `else`-ով։ Ինչպե՞ս է այդ warp-ի կատարման ժամանակը համեմատվում առանց divergence-ի warp-ի հետ, որ նույն ծավալի աշխատանք է անում։
4. 4-րդ առաջադրանքի branch-ը `threadIdx.x / 32`-ի վրա է և ոչինչ չի արժենում։ Ինչու՞։
5. Եթե latency hiding-ը կախված է ուրիշ պատրաստ warp-երի առկայությունից, ի՞նչ է լինում այն kernel-ի հետ, որն այնքան ռեգիստր է օգտագործում, որ ընդամենը մեկ warp է resident։

## Կոդի template
Տես [`template.cu`](template.cu)։
