# Օր 4. Հիշողության տեսակները և host-device փոխանցումները

## Նպատակներ
- Տարբերել paged, pinned, page-locked, mapped և unified հիշողությունը, և անվանել ամեն մեկի API-ն
- Բացատրել, թե ինչու է pageable փոխանցումը pinned-ից դանդաղ նույն քանակի բայթերի դեպքում
- Չափել փոխանցման bandwidth-ը և համեմատել device-ի սեփական հիշողության bandwidth-ի հետ
- Nsight Systems-ով տեսնել, թե հիշողության ընտրությունն ինչպես է երևում timeline-ի վրա

## Հիմնական հասկացություններ
- Paged հիշողություն (`malloc`), pinned հիշողություն (`cudaMallocHost`), տեղում page locking (`cudaHostRegister`)
- Mapped, այսինքն՝ zero-copy հիշողություն (`cudaHostAlloc`՝ `cudaHostAllocMapped`-ով)
- Unified հիշողություն (`cudaMallocManaged`, `__managed__`) և page migration
- DMA և ինչու է այն պահանջում pinned էջեր
- PCIe-ի և NVLink-ի bandwidth-ը՝ համեմատած device-ի հիշողության bandwidth-ի հետ

## Սահմանումներ
**Pageable հիշողություն** — Սովորական host հիշողություն՝ `malloc`-ից կամ `new`-ից։ Օպերացիոն համակարգը կարող է տեղափոխել կամ swap անել դրա էջերը, ուստի GPU-ն ուղիղ չի կարող դիմել. փոխանցումը նախ պատճենում է այն driver-ի պահած page-locked միջանկյալ բուֆեր։

**Page-locked հիշողություն** — Host-ի հիշողություն, որի էջերը օպերացիոն համակարգը իրավունք չունի տեղափոխել կամ swap անել։

**Pinned հիշողություն** — `cudaMallocHost`-ով հատկացված page-locked host հիշողություն, որպեսզի GPU-ն այն փոխանցի DMA-ով՝ առանց միջանկյալ պատճենի։ Պարտադիր է, որպեսզի `cudaMemcpyAsync`-ը իրոք ասինխրոն լինի։

**`cudaHostRegister`** — Page-lock է անում սովորական ձևով հատկացված հիշողությունը՝ տալով նրան pinned հիշողության փոխանցման հատկությունները, առանց վերահատկացնելու։ `cudaHostUnregister`-ը հետ է շրջում այս ամենը։

**Mapped (zero-copy) հիշողություն** — Page-locked host հիշողություն, որն ունի նաև device-ի հասցե՝ `cudaHostAlloc`-ից `cudaHostAllocMapped`-ով։ Kernel-ը կարդում և գրում է այն ուղիղ, կապի վրայով, առանց բացահայտ պատճենի՝ ամեն դիմումին վճարելով կապի latency-ն։

**Unified հիշողություն** — Մեկ հատկացում՝ `cudaMallocManaged`, հասցեագրելի և host-ից, և device-ից, և driver-ը էջերը տեղափոխում է նրանց միջև ըստ պահանջի։

**Page migration** — Unified հիշողության էջի տեղափոխումն այն պրոցեսորին, որը դրա վրա fault է տվել։ Երկու ուղղությամբ կրկնվող տեղափոխումը unified հիշողության դանդաղության սովորական պատճառն է. կառավարվում է `cudaMemPrefetchAsync`-ով և `cudaMemAdvise`-ով։

**DMA (Direct Memory Access)** — Փոխանցում, որը կատարում է copy engine-ը՝ առանց CPU-ի տվյալները շարժելու։ Պահանջում է, որ host-ի էջերը page-locked լինեն, և հենց դրա համար են pinned փոխանցումներն ավելի արագ։

**PCIe** — Host-ը և device-ը միացնող bus-ը համակարգերի մեծ մասում։ Նրա bandwidth-ը մեկ կարգով ցածր է device-ի հիշողության bandwidth-ից։

**NVLink** — NVIDIA-ի ուղիղ GPU-GPU կապը, որոշ համակարգերում նաև CPU-GPU, PCIe-ից մի քանի անգամ մեծ bandwidth-ով։

**Հիշողության bandwidth** — Բայթ վայրկյանում SM-երի և device-ի հիշողության միջև։ Տեսական peak-ը bus width-ն է բազմապատկած հիշողության clock-ով և մեկ clock-ի փոխանցումների քանակով. հասած թիվը այն է, ինչին kernel-ն իրականում հասնում է։

## Այն թիվը, որ նշանակություն ունի
Device-ի հիշողության bandwidth-ը հարյուրավոր GB/s-ից մինչև մի քանի TB/s է։ PCIe-ն մեկ կարգ ցածր է։ Այն kernel-ը, որը փոխանցում է իր մուտքը, մեկ անգամ հաշվում է դրա վրայով և արդյունքը հետ փոխանցում, սահմանափակված է կապով, ոչ թե GPU-ով։ Սա առաջին բանն է, որ պետք է ստուգել, երբ GPU տեղափոխված kernel-ը հիասթափեցնում է, և հենց սրա համար են stream-երը գոյություն ունեն (Օր 8)։

## Պատկեր
![Host-ից device փոխանցման չորս ճանապարհ՝ pageable միջանկյալ պատճենով, pinned ուղիղ DMA-ով, mapped, որտեղ GPU-ն ուղիղ կարդում է host-ի հիշողությունը, unified, որտեղ runtime-ը տեղափոխում է էջերը](memory_types.svg)

Հիշողության ամեն տեսակ նույն հարցի այլ պատասխան է՝ ինչպես են տվյալները host-ի RAM-ից հասնում device-ի VRAM։ Pageable հիշողությունը պահանջում է թաքնված միջանկյալ պատճեն, pinned-ը՝ ոչ։ Mapped-ը պատճենն ամբողջությամբ հանում է և փոխարենը վճարում ամեն դիմումի latency-ն։ Unified-ը որոշումը թողնում է runtime-ին։

## Գրականություն
- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/pdf/CUDA_C_Programming_Guide.pdf) — 3.2.2. Device Memory · 3.2.6. Page-Locked Host Memory · 19. Unified Memory Programming
- CUDA C++ Best Practices Guide — Memory Optimizations, Pinned Memory
- Nsight Systems — User Guide

## Լաբորատոր առաջադրանք
Բարելավել Օր 2-ի և Օր 3-ի վեկտորների գումարումը pinned հիշողությամբ, և երկուսը համեմատել Nsight Systems-ում։

## Ինքնուրույն աշխատանք
1. Չափել `cudaMemcpy`-ը pageable և pinned host հիշողությամբ՝ փոխանցման մի քանի չափի համար։ Կառուցել bandwidth-ի կախվածությունը չափից։
2. Գոյություն ունեցող pageable բուֆերը page-lock անել `cudaHostRegister`-ով՝ նախապես pinned հիշողություն հատկացնելու փոխարեն, և համեմատել։
3. Վերագրել վեկտորների գումարումը `cudaMallocManaged`-ով և համեմատել և՛ կոդի բարդությունը, և՛ արագագործությունը։
4. Բոլոր երեք տարբերակը profile անել Nsight Systems-ով և համեմատել փոխանցումների timeline-ները։
5. Հաշվել ձեր վեկտորների գումարման arithmetic intensity-ն և Օր 1-ին հավաքած թվերից որոշել՝ kernel-ն է սահմանափակումը, թե փոխանցումը։

## Ինքնաստուգում
Պատասխանները տրված չեն։

1. Ինչու՞ է pageable-ից device պատճենը pinned-ից դանդաղ, եթե նույն քանակի բայթ է շարժվում։
2. Ի՞նչ արժե համակարգի համար host-ի մեծ ծավալի հիշողություն pin անելը։
3. Ե՞րբ է zero-copy հիշողությունը գերազանցում նախ device պատճենելուն։
4. Unified հիշողությունը սկզբնական կոդից հանում է բացահայտ պատճենը։ Փոխանցո՞ւմն էլ է հանում։

## Կոդի template
Տես [`template.cu`](template.cu)։
