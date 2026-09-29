# Օր 7. Warp intrinsic-ները, reduction-ը և atomic-ները

## Նպատակներ
- Օգտագործել warp shuffle ֆունկցիաները warp-ի ներսում տվյալներ փոխանակելու համար՝ առանց shared memory-ի և barrier-ների
- Նկարագրել, թե ինչ է նշանակում lane mask-ը, և ինչու է divergent branch-ում կանչված `_sync` intrinsic-ը undefined behaviour, ոչ թե պարզապես դանդաղ կոդ
- Իրականացնել warp-ի մակարդակի reduction և ընդլայնել այն ամբողջ block-ի վրա՝ warp, shared memory, ապա կրկին warp սխեմայով
- Իրականացնել inclusive scan warp-ի ներսում, իսկ բուլյան պայմանների համար օգտագործել `__ballot_sync`-ը `__popc`-ի հետ
- Նկարագրել, թե ինչու է նույն հասցեի համար atomic-ների մրցակցությունը կատարումը դարձնում հաջորդական, և կիրառել privatisation

## Հիմնական հասկացություններ
- `__shfl_down_sync`, `__shfl_up_sync`, `__shfl_xor_sync`, `__shfl_sync`
- Lane mask, divergence և չեզոք արժեքներ սահմանից դուրս գտնվող lane-երի համար
- Հիերարխիկ reduction՝ warp, shared memory, ապա կրկին warp
- Inclusive scan (Kogge-Stone) և stream compaction
- `__syncwarp`, `__activemask`, `__ballot_sync`
- Global memory-ի atomic-ները կատարվում են L2-ում, և նույն հասցեի համար մրցակցությունը դրանք դարձնում է հաջորդական
- Privatisation shared memory-ում և warp-aggregated atomic-ներ

## Սահմանումներ
**Lane** — Thread-ի համարը իր warp-ում՝ 0-ից 31։

**Warp shuffle** — `__shfl_sync`, `__shfl_up_sync`, `__shfl_down_sync`, `__shfl_xor_sync` հրամանների ընտանիքը։ Դրանք lane-ին թույլ են տալիս կարդալ նույն warp-ի մեկ այլ lane-ի ռեգիստրը՝ առանց հիշողության դիմումի և առանց barrier-ի։

**Lane mask** — Ամեն `_sync` intrinsic-ի առաջին արգումենտը՝ 32-բիթանոց թիվ, որում ամեն lane-ին համապատասխանում է մեկ բիթ։ Այն նշում է այն lane-երը, որոնք պետք է մասնակցեն։ `0xffffffff`-ը նշանակում է ամբողջ warp-ը։

**`_sync` վերջածանց** — Այս վերջածանցն ունեն այն intrinsic-ները, որոնք պահանջում են, որ mask-ում նշված lane-երը միասին հասնեն հրամանին։ Եթե նշված lane-երից որևէ մեկը չի հասնում, արդյունքը undefined է, և խնդիրը միայն դանդաղությունը չէ։ Առանց վերջածանցի տարբերակները Volta-ից սկսած architecture-ների համար հասանելի չեն։

**XOR (butterfly) փոխանակում** — `__shfl_xor_sync(mask, v, k)`. lane `i`-ն արժեքներ է փոխանակում lane `i ^ k`-ի հետ։ Մեկ հրամանով ամեն lane և՛ ուղարկում է, և՛ ստանում, ուստի այս ձևն օգտագործվում է, երբ արդյունքը պետք է ունենան բոլոր lane-երը։

**Reduction** — N արժեքի միավորումը մեկ արժեքի մեջ ասոցիատիվ գործողությամբ (օրինակ՝ գումարում կամ max)։ GPU-ի վրա այն կատարվում է ծառի տեսքով. warp-ի ներսում՝ shuffle-ներով, warp-երի միջև՝ shared memory-ով, ապա վերջնականապես՝ մեկ warp-ով։

**Inclusive և exclusive scan** — Նախածանցային գումարներ (prefix sums)։ Inclusive scan-ի i-րդ տարրը 0-ից i համարի տարրերի գումարն է (կամ այլ ասոցիատիվ գործողության արդյունքը), exclusive scan-ինը՝ 0-ից i-1 համարի տարրերինը։

**Kogge-Stone** — Warp-ի ներսում scan կատարելու եղանակ. ամեն քայլում lane-ը իր արժեքին գումարում է իրենից `d` դիրքով ցածր lane-ի արժեքը, և `d`-ն ամեն քայլում կրկնապատկվում է (1, 2, 4, 8, 16)։ 32 lane-անոց warp-ի համար պահանջվում է հինգ քայլ։

**Stream compaction** — Պայմանին չբավարարող տարրերը հեռացնել, իսկ մնացածը դասավորել իրար հետևից՝ առանց բացերի։ Պայմանի արժեքների (0 կամ 1) վրա կատարված scan-ը ամեն մնացող տարրին տալիս է նրա ինդեքսը ելքում։

**`__ballot_sync`** — Վերադարձնում է 32-բիթանոց mask, որում N-րդ բիթը 1 է, եթե N-րդ lane-ի պայմանը ճիշտ է։ Արդյունքը ստանում են բոլոր մասնակից lane-երը։

**`__popc`** — Հաշվում է 32-բիթանոց թվի 1 արժեք ունեցող բիթերը։ Ballot-ի արդյունքի վրա կիրառելիս տալիս է պայմանին բավարարող lane-երի քանակը։

**`__activemask`** — Վերադարձնում է այն lane-երի mask-ը, որոնք միասին են հասել այս հրամանին։ Այն ցույց է տալիս միայն տվյալ պահի վիճակը, որը կարող է պատահական լինել, և չի փոխարինում այն mask-ին, որը պետք է որոշի կոդը։

**`__syncwarp`** — Warp-ի մակարդակի barrier, որը ստիպում է mask-ում նշված lane-երին նորից միավորվել։ Այն անհրաժեշտ է այնտեղ, որտեղ կոդը ենթադրում է, որ lane-երը միասին են, իսկ կոմպիլյատորը չի կարող դա ապացուցել։

**Atomic գործողություն** — Մեկ հասցեի վրա կարդալ-փոխել-գրել գործողություն, որին ոչ մի այլ thread չի կարող միջամտել. `atomicAdd`, `atomicCAS`, `atomicMax` և այլն։

**Atomic-ների մրցակցություն (contention)** — Իրավիճակ, երբ մի քանի thread atomic գործողություն է կատարում նույն հասցեի վրա։ Global memory-ի դեպքում թարմացումները L2-ում կատարվում են հերթով, ուստի գինն աճում է նույն հասցեին դիմող thread-երի քանակի հետ, ոչ թե atomic հրամանների ընդհանուր քանակի։

**Privatisation** — Ամեն block-ին կամ warp-ին տալ ընդհանուր կուտակիչի (օրինակ՝ histogram-ի) սեփական պատճենը, թարմացնել այդ պատճենը և վերջում մեկ անգամ ավելացնել ընդհանուր արդյունքին։ Ամեն մուտքային տարրի համար մեկ global atomic-ի փոխարեն ամբողջ block-ը կատարում է համեմատաբար քիչ global atomic։ Սա atomic-ների մրցակցության ստանդարտ լուծումն է։

**Warp-aggregated atomic-ներ** — Warp-ի բոլոր lane-երի ներդրումը նախ հաշվվում է ballot-ով կամ warp reduction-ով, ապա մեկ lane-ը կատարում է մեկ atomic ամբողջ warp-ի համար։ Atomic-ների քանակը նվազում է մինչև 32 անգամ։ Սա privatisation-ն է warp-ի մակարդակում։

**Cooperative groups** — API, որը բացահայտ է դարձնում, թե thread-երի որ խմբի վրա է կոդը սինխրոնացվում. block, տվյալ պահին միասին գտնվող lane-եր կամ ամբողջ grid։ `__syncthreads()`-ի դեպքում այդ խումբը բացահայտ նշված չէ։

## Ֆունկցիաներ

```c
// Կարդում է srcLane համարով lane-ի var-ը
int   __shfl_sync(unsigned mask, int var, int srcLane, int width = warpSize);
float __shfl_sync(unsigned mask, float var, int srcLane, int width = warpSize);

// Կարդում է իրենից delta դիրքով ցածր lane-ի var-ը
int   __shfl_up_sync(unsigned mask, int var, unsigned int delta, int width = warpSize);
float __shfl_up_sync(unsigned mask, float var, unsigned int delta, int width = warpSize);

// Կարդում է իրենից delta դիրքով բարձր lane-ի var-ը
int   __shfl_down_sync(unsigned mask, int var, unsigned int delta, int width = warpSize);
float __shfl_down_sync(unsigned mask, float var, unsigned int delta, int width = warpSize);

// Կարդում է lane ^ laneMask համարով lane-ի var-ը
int   __shfl_xor_sync(unsigned mask, int var, int laneMask, int width = warpSize);
float __shfl_xor_sync(unsigned mask, float var, int laneMask, int width = warpSize);

// N-րդ բիթը 1 է, եթե N-րդ lane-ի pred-ը զրո չէ
unsigned __ballot_sync(unsigned mask, int pred);

// 1 արժեք ունեցող բիթերի քանակը
int __popc(unsigned int x);

// Այս հրամանին միասին հասած lane-երի mask-ը
unsigned __activemask();

// Warp-ի և block-ի barrier-ներ
void __syncwarp(unsigned mask = 0xFFFFFFFF);
void __syncthreads(void);

// *address += val atomic ձևով։ Վերադարձնում է նախկին արժեքը
int                    atomicAdd(int *address, int val);
unsigned int           atomicAdd(unsigned int *address, unsigned int val);
unsigned long long int atomicAdd(unsigned long long int *address, unsigned long long int val);
float                  atomicAdd(float *address, float val);

// Նույնը, բայց atomic է միայն block-ի ներսում
int          atomicAdd_block(int *address, int val);
unsigned int atomicAdd_block(unsigned int *address, unsigned int val);

// Եթե *address == compare, գրում է val։ Վերադարձնում է նախկին արժեքը
int atomicCAS(int *address, int compare, int val);

// *address = max(*address, val) atomic ձևով։ Վերադարձնում է նախկին արժեքը
int atomicMax(int *address, int val);

// Host-ի ֆունկցիա՝ device-ի հիշողության count բայթը լրացնում է value-ով
cudaError_t cudaMemset(void *devPtr, int value, size_t count);
```

## Ինչու են shuffle-ներն ավելի արագ
Shuffle-ները տվյալները տեղափոխում են ռեգիստրից ռեգիստր՝ ընդհանրապես չդիմելով հիշողությանը։ Այդ պատճառով դրանք ավելի արագ են, քան shared memory-ով reduction-ը։ Նույն պատճառով այդ առավելությունը վերանում է, երբ տվյալներն այլևս չեն տեղավորվում warp-ի ռեգիստրներում։

## Պատկեր
![Warp shuffle reduction 8 lane-ի վրա. ամեն քայլում offset-ը կիսվում է (4, 2, 1)՝ __shfl_down_sync-ով, մինչև lane 0-ում մնում է ընդհանուր գումարը](warp_reduction.svg)

`__shfl_down_sync`-ը թույլ է տալիս lane-ին կարդալ արժեքն ուղիղ այլ lane-ի ռեգիստրից՝ առանց shared memory-ի և առանց `__syncthreads()`-ի։ Ամեն քայլում offset-ը կիսելով՝ 32 արժեքը գումարվում են հինգ քայլում, և արդյունքը մնում է lane 0-ում։ Ամբողջ warp-ի համար offset-ներն են 16, 8, 4, 2, 1։

![__ballot_sync-ը warp-ի 32 բուլյան պայմանները հավաքում է մեկ 32-բիթանոց mask-ում՝ մեկ բիթ ամեն lane-ի համար](warp_ballot.svg)

`__ballot_sync`-ը «որ lane-երն են բավարարում այս պայմանին» հարցի պատասխանը դարձնում է մեկ 32-բիթանոց ամբողջ թիվ, որը ստանում է ամեն lane։ N-րդ բիթը 1 է այն և միայն այն դեպքում, երբ N-րդ lane-ի պայմանը ճիշտ է։ `__popc`-ի հետ միասին այն մեկ հրամանով հաշվում է այդ lane-երի քանակը։ `__activemask`-ը և `__syncwarp`-ը օգտագործում են lane-երի նույն 32-բիթանոց mask-ը։

## Անիմացիա
![8 lane երեք shuffle intrinsic-ի դեպքում. __shfl_down_sync, որտեղ lane i-ն կարդում է lane i+1-ից, __shfl_up_sync, որտեղ lane i-ն կարդում է lane i-1-ից, և __shfl_xor_sync, որտեղ lane-երը զույգերով փոխանակում են արժեքները](warp_shuffle_intrinsics.svg)

Նույն ութ lane-ը երեք տարբեր intrinsic-ի դեպքում։ `__shfl_down_sync`-ը և `__shfl_up_sync`-ը արժեքները տեղաշարժում են մեկ ուղղությամբ՝ ֆիքսված offset-ով։ `__shfl_xor_sync`-ը արժեքները փոխանակում է lane-երի զույգերի միջև (`i` և `i ^ laneMask`), ուստի մեկ հրամանով ամեն lane և՛ ուղարկում է, և՛ ստանում։ Այդ պատճառով այն օգտագործվում է butterfly reduction-ում։

## Գրականություն
- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/pdf/CUDA_C_Programming_Guide.pdf) — 7.22. Warp Shuffle Functions · 7.14. Atomic Functions · 8. Cooperative Groups
- [`INTRINSICS.md`](../INTRINSICS.md) — shuffle, vote, բիթային գործողություններ և atomic-ներ՝ մեկ աղյուսակում
- CUB-ի device-wide reduction և scan՝ https://nvidia.github.io/cccl/cub/

## Լաբորատոր առաջադրանք
Հաշվել իրական պատկերի պիքսելների միջին արժեքը warp reduction-ով և `atomicAdd`-ով։ Ապա կառուցել 256 bin-անոց histogram երկու եղանակով՝ global atomic-ներով և shared memory-ում privatisation-ով։

## Ինքնուրույն աշխատանք
1. Իրականացնել գումարման reduction warp-ի ներսում `__shfl_down_sync`-ով։ Արդյունքը ստուգել host-ի ցիկլով `n = 1024` մեկերով լցված զանգվածի վրա. այն պետք է ճշգրիտ հավասար լինի `n`-ի։
2. Ընդլայնել այն մինչև `block_reduce_sum`՝ warp, shared memory, ապա կրկին warp սխեմայով։ Այնուհետև reduction կատարել ամբողջ grid-ի վրա. ամեն block-ի thread 0-ն իր գումարն ավելացնում է ընդհանուրին `atomicAdd`-ով։ Հաշվել `__syncthreads()`-ի կանչերի քանակը և համեմատել դասական ծառային reduction-ի հետ, որն ամբողջությամբ կատարվում է shared memory-ում։ Չափել երկու տարբերակի ժամանակը։
3. Իրականացնել inclusive scan warp-ի ներսում և դրանով խտացնել շեմից բարձր պիքսելների ինդեքսները։
4. Կրկնել 3-րդ առաջադրանքը `__ballot_sync`-ով և `__popc`-ով և համեմատել երկու տարբերակը։
5. Իրականացնել 256 bin-անոց histogram երկու եղանակով։ Չափել երկուսի ժամանակը մեծ պատկերի վրա, ապա միագույն պատկերի վրա (օրինակ՝ `data/uniform.bmp`)։ Նկարագրել, թե տարբերությունը մեծանում է, թե փոքրանում։
6. Shared memory-ում `atomicAdd`-ը փոխարինել `atomicAdd_block`-ով և չափել ազդեցությունը։

## Ինքնաստուգում
Պատասխանները տրված չեն։

1. Ինչու՞ է `__shfl_down_sync`-ը աշխատում առանց `__syncthreads()`-ի։
2. Reduction-ի հինգ քայլից հետո ինչու՞ է ընդհանուր գումարը երաշխավորված հենց lane 0-ում։
3. Եթե mask-ը `0xFFFFFFFF` է, բայց հրամանին հասնում են lane-երի միայն մի մասը, վարքը undefined է։ Ինչու՞ ձեր GPU-ի վրա ստացված ճիշտ պատասխանը չի ապացուցում, որ կոդը ճիշտ է։
4. Privatisation-ը ավելացնում է զրոյացման անցում, երկու barrier և վերջնական միավորման անցում։ Նկարագրել մուտքային տվյալներ, որոնց դեպքում privatisation-ն ընդհանուր առմամբ դանդաղեցնում է ծրագիրը։
5. Ինչու՞ է shared memory-ի `atomicAdd`-ը global memory-ի `atomicAdd`-ից էժան, եթե բախման դեպքում երկուսն էլ կատարվում են հերթով։

## Կոդի template
Տես [`template.cu`](template.cu)։
