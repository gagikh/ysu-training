# Օր 9. Գրադարանները, tensor core-երը և ճշգրտությունը

## Նպատակներ
- Ձեռքով գրված kernel-ները փոխարինել cuBLAS-ի, cuFFT-ի, cuRAND-ի և cuDNN-ի կանչերով և նկարագրել, թե երբ է սեփական kernel-ը դեռ արդարացված
- Նկարագրել, թե ինչ է անում tensor core-ը և ինչ պահանջներ ունի տվյալների դասավորության և չափերի նկատմամբ
- Համեմատել fp64-ի, fp32-ի, tf32-ի, bf16-ի և fp16-ի throughput-ը կլաստերի GPU-ի վրա և նկարագրել, թե դա ինչ հետևանք ունի թվային հաշվարկների համար
- Նկարագրել, թե ինչու atomic-ներով կուտակման արդյունքը կարող է տարբեր լինել տարբեր գործարկումներում

## Հիմնական հասկացություններ
- cuBLAS, cuSOLVER, cuSPARSE, cuFFT, cuRAND, CUB և Thrust
- cuDNN, NPP և nvJPEG. դրանք են իրականում կանչում deep learning framework-երը
- Tensor core-եր՝ matrix multiply-accumulate մեկ հրամանով, հավասարեցման և չափերի պահանջներ
- Ճշգրտություն՝ fp64, fp32, tf32, bf16, fp16, fused multiply-add
- Atomic-ներով կուտակման ոչ դետերմինիզմը, և ինչ է պետք արդյունքի վերարտադրելիությունը պնդելու համար
- Cooperative groups՝ `tiled_partition`, խմբի տիպի մեթոդներով shuffle-ներ, ամբողջ grid-ի սինխրոնացում

## Սահմանումներ
**Tensor Core** — SM-ի մասնագիտացված hardware միավոր խառը ճշգրտությամբ matrix multiply-accumulate-ի արագ կատարման համար։ Առկա է compute capability 7.0-ից սկսած։ Այն օգտագործում են cuBLAS-ը և cuDNN-ը, իսկ ուղիղ հասանելի է warp matrix ֆունկցիաներով։

**MMA (matrix multiply-accumulate)** — Tensor core-ի գործողությունը՝ `D = A * B + C` փոքր մատրիցային հատվածների վրա, որը warp-ը կատարում է մեկ հրամանով։ Հատվածների չափերը և հավասարեցումը ֆիքսված են hardware-ում, ուստի մատրիցների չափերը պետք է լինեն հատվածի չափի բազմապատիկ։

**fp64, fp32, tf32, bf16, fp16** — Լողացող կետով թվերի ձևաչափեր, որոնք տարբերվում են էքսպոնենտի և մանտիսի բիթերի քանակով. fp64՝ 11 և 52, fp32՝ 8 և 23, tf32՝ 8 և 10 (միայն tensor core-ի մուտքային ձևաչափ), bf16՝ 8 և 7 (նույն տիրույթը, ինչ fp32-ը, բայց ավելի ցածր ճշգրտությամբ), fp16՝ 5 և 10 (ավելի նեղ տիրույթ և ավելի ցածր ճշգրտություն, քան fp32-ը)։

**Խառը ճշգրտություն (mixed precision)** — Հաշվարկը կատարել նեղ ձևաչափով, իսկ կուտակումը՝ ավելի լայնով։ Սովորաբար մուտքերը fp16 կամ bf16 են, իսկ կուտակումը՝ fp32։ Tensor core-երն աշխատում են հենց այդպես։

**FMA (fused multiply-add)** — `a * b + c`-ի հաշվարկ մեկ կլորացմամբ՝ երկուսի փոխարեն։ Սա այն պատճառներից մեկն է, որոնց հետևանքով նույն մուտքերի և գործողությունների նույն հերթականության դեպքում GPU-ի և CPU-ի արդյունքները կարող են տարբերվել վերջին բիթերում։

**Atomic-ներով կուտակման ոչ դետերմինիզմ** — Atomic-ները չեն ամրագրում արժեքների գումարման հերթականությունը, իսկ լողացող կետով գումարումն ասոցիատիվ չէ։ Ուստի `atomicAdd`-ով float-եր կուտակող kernel-ը տարբեր գործարկումներում կարող է տարբեր արդյունք տալ։ Վերարտադրելի արդյունքի համար reduction-ի հերթականությունը պետք է ֆիքսված լինի։

**Ամբողջ grid-ի սինխրոնացում** — Barrier grid-ի բոլոր block-երի համար։ Այն հասանելի է cooperative groups-ով և միայն այն kernel-ների համար, որոնք գործարկված են `cudaLaunchCooperativeKernel`-ով, և որոնց բոլոր block-երը կարող են միաժամանակ resident լինել։

**cuBLAS, cuFFT, cuRAND, cuDNN, NPP, nvJPEG, CUB, Thrust** — NVIDIA-ի գրադարաններ՝ համապատասխանաբար խիտ գծային հանրահաշվի, Ֆուրիեի արագ ձևափոխության, պատահական թվերի ստեղծման, deep learning-ի հիմնական գործողությունների, պատկերների և ազդանշանների մշակման, JPEG-ի վերծանման և կոդավորման, block-ի և device-ի մակարդակի զուգահեռ ալգորիթմների համար։ Thrust-ը STL-ի նման ալգորիթմների շերտ է CUB-ի վրա։

## Ֆունկցիաներ

```c
// cuBLAS։ cublas_v2.h-ը այս անունները փոխարինում է _v2 տարբերակներով (cublasCreate_v2 և այլն)։ Նախատիպերը նույնն են
cublasStatus_t cublasCreate(cublasHandle_t *handle);
cublasStatus_t cublasDestroy(cublasHandle_t handle);

// CUBLAS_TF32_TENSOR_OP_MATH ռեժիմը թույլ է տալիս fp32 հաշվարկները կատարել tf32 tensor core-երով
cublasStatus_t cublasSetMathMode(cublasHandle_t handle, cublasMath_t mode);

// C = alpha * op(A) * op(B) + beta * C։ Մատրիցները column-major են
cublasStatus_t cublasSgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, int k,
                           const float *alpha, const float *A, int lda,
                           const float *B, int ldb,
                           const float *beta, float *C, int ldc);
cublasStatus_t cublasDgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, int k,
                           const double *alpha, const double *A, int lda,
                           const double *B, int ldb,
                           const double *beta, double *C, int ldc);

// NPP։ Box ֆիլտր 8-բիթանոց մեկ ալիքով պատկերի համար՝ եզրերի մշակմամբ
NppStatus nppiFilterBoxBorder_8u_C1R(const Npp8u *pSrc, Npp32s nSrcStep, NppiSize oSrcSize,
                                     NppiPoint oSrcOffset, Npp8u *pDst, Npp32s nDstStep,
                                     NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor,
                                     NppiBorderType eBorderType);

// Kernel-ի cooperative launch։ Առանց դրա ամբողջ grid-ի սինխրոնացումը հնարավոր չէ
cudaError_t cudaLaunchCooperativeKernel(const void *func, dim3 gridDim, dim3 blockDim,
                                        void **args, size_t sharedMem, cudaStream_t stream);

// cooperative_groups։ g խումբը բաժանում է Size thread-անոց մասերի
template <unsigned int Size, typename ParentT>
thread_block_tile<Size, ParentT> tiled_partition(const ParentT &g);

// Օր 7-ի ֆունկցիաներ
int   __shfl_down_sync(unsigned mask, int var, unsigned int delta, int width = warpSize);
float atomicAdd(float *address, float val);
```

## Ճշգրտությունը կլաստերի GPU-ի վրա
Սպառողական և inference-ի համար նախատեսված քարտերի վրա fp64-ի արագությունը fp32-ի արագության 1/32-ից 1/64 մասն է։ Տվյալների կենտրոնների քարտերի վրա այն մոտ է 1/2-ին։ Կրկնակի ճշգրտությամբ որևէ աշխատանք նախագծելուց առաջ պետք է Օր 1-ի թվերով որոշել, թե կլաստերի GPU-ն որ խմբին է պատկանում։

## Պատկեր
![fp64-ի, fp32-ի, tf32-ի, bf16-ի և fp16-ի բիթային կառուցվածքը մասշտաբով. ամեն ձևաչափ բաժանված է նշանի, էքսպոնենտի և մանտիսի դաշտերի](precision_formats.svg)

Էքսպոնենտի դաշտը որոշում է տիրույթը, մանտիսը՝ ճշգրտությունը։ bf16-ը և tf32-ը ունեն fp32-ի 8-բիթանոց էքսպոնենտը, ուստի fp32-ում ներկայացվող արժեքը ներկայացվում է նաև դրանցում (ավելի ցածր ճշգրտությամբ)։ fp16-ի 5-բիթանոց էքսպոնենտի դեպքում այդպես չէ, և այդ պատճառով fp16-ով ուսուցումը պահանջում է loss scaling, իսկ bf16-ով՝ ոչ։ tf32-ը պահպանման ձևաչափ չէ. այն գոյություն ունի միայն որպես tensor core-ի մուտքային ձևաչափ, ուստի cuBLAS-ում այն միացվում է math mode-ով, ոչ թե տվյալների տիպով։

## Գրականություն
- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/pdf/CUDA_C_Programming_Guide.pdf) — 7.24. Warp Matrix Functions · 8. Cooperative Groups
- Train With Mixed Precision՝ https://docs.nvidia.com/deeplearning/performance/
- Micikevicius P. et al. Mixed Precision Training. *ICLR*, 2018. arXiv:1710.03740
- cuBLAS, cuFFT, cuRAND՝ https://docs.nvidia.com/cuda/ · cuDNN՝ https://docs.nvidia.com/deeplearning/cudnn/

## Լաբորատոր առաջադրանք
Ձեռքով գրված երկու kernel փոխարինել գրադարանային կանչերով. Օր 5-ի և Օր 6-ի box ֆիլտրը՝ NPP-ով, իսկ մատրիցների բազմապատկումը՝ cuBLAS-ով։ Naive բազմապատկումը տրված է template-ում, ուստի այս առաջադրանքը կախված չէ Օր 5-ի ընդլայնված առաջադրանքից։ Ապա միացնել tensor core-երը և համեմատել և՛ արագությունը, և՛ արդյունքը։

## Ինքնուրույն աշխատանք
1. Մատրիցների բազմապատկումը կատարել cuBLAS-ով և համեմատել ձեր kernel-ի հետ՝ ըստ ժամանակի և արդյունքի։
2. Նույն բազմապատկման համար միացնել tf32, ապա bf16։ Գրանցել արագությունը և արդյունքի տարբերությունը fp32-ի համեմատ։
3. Profiler-ով ստուգել, որ tensor core-երն իրոք օգտագործվում են։ Սա չի կարելի ենթադրել։
4. Compute-bound kernel-ով չափել fp64-ի throughput-ը՝ համեմատած fp32-ի հետ, կլաստերի GPU-ի վրա։ Արդյունքը համեմատել այդ քարտի համար հրապարակված հարաբերության հետ։
5. Նույն մեծ զանգվածի տարրերը տասն անգամ գումարել `atomicAdd`-ով։ Գրանցել, թե արդյոք արդյունքները բիթ առ բիթ նույնն են, և նկարագրել։
6. cuRAND-ով Monte Carlo եղանակով գնահատել π-ի արժեքը։

## Ինքնաստուգում
Պատասխանները տրված չեն։

1. Volta-ից հին GPU-ները tensor core չունեն։ Ի՞նչ է դա նշանակում այդ GPU-ների վրա cuBLAS-ի համար։
2. Ուսանողը հայտնում է, որ tensor core-երով ստացել է 40 անգամ արագացում՝ առանց ճշգրտության կորստի։ Ի՞նչ կխնդրեիք նրան ցույց տալ։
3. Ինչու՞ է `cg::tiled_partition<32>`-ը նախընտրելի ուղիղ `__shfl_down_sync`-ից, եթե երկուսից էլ ստացվում է նույն հրամանը։
4. Ի՞նչ պայմաններում է սեփական kernel գրելը դեռ ճիշտ որոշում, եթե համապատասխան գրադարանային ֆունկցիա գոյություն ունի։

## Կոդի template
Տես [`template.cu`](template.cu)։
