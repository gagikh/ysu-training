# Օր 8. Stream-երը, event-ները, ասինխրոնությունը և CUDA graph-երը

## Նպատակներ
- Տարբերել սինխրոն և ասինխրոն պատճենումը (`cudaMemcpy` և `cudaMemcpyAsync`) և բացատրել, թե երբ է ասինխրոն պատճենումն աննկատ դառնում սինխրոն
- Stream-երի միջոցով փոխանցումը համատեղել հաշվարկի հետ և համատեղումը ստուգել event-ներով և Nsight Systems-ով
- Ժամանակը չափել device-ի վրա event-ներով, ոչ թե host-ի ժամացույցով
- Գործողությունների հաջորդականությունը գրանցել CUDA graph-ում և բացատրել, թե երբ է graph-ով գործարկումն արդարացված

## Հիմնական հասկացություններ
- Default stream (per-thread), և ինչու են մեկ stream-ի գործողությունները կատարվում հերթով, նույնիսկ եթե անկախ են
- `cudaMemcpyAsync`, stream-երի միջև կախվածություններ և pinned հիշողության պահանջը
- Event-ներ՝ `cudaEventCreate`, `cudaEventRecord`, `cudaEventSynchronize`, `cudaEventElapsedTime`
- Մասերով pipeline. մի մասի պատճենումը, մյուսի հաշվարկը և երրորդի հետ պատճենումը կատարվում են միաժամանակ
- Graph-ի capture, instantiation և launch, launch overhead-ի նվազեցում

## Սահմանումներ
**Stream** — GPU-ի գործողությունների (kernel-ներ, պատճենումներ) կարգավորված հերթ։ Մեկ stream-ի ներսում գործողությունները կատարվում են այն հերթականությամբ, որով ուղարկվել են։ Տարբեր stream-երի գործողությունները կարող են կատարվել միաժամանակ։

**Default stream (per-thread)** — Այն stream-ը, որն օգտագործվում է, երբ stream նշված չէ։ `--default-stream per-thread` flag-ով կոմպիլացնելիս host-ի ամեն thread ստանում է իր default stream-ը, որը չի սպասում մյուս stream-երի գործողությունների ավարտին։

**`cudaMemcpyAsync`** — Պատճենում, որն ուղարկվում է stream և վերադառնում է անմիջապես։ Այն իրոք ասինխրոն է միայն այն դեպքում, երբ host-ի հիշողությունը page-locked է։ Pageable հիշողության դեպքում runtime-ը կատարում է սինխրոն պատճենում և այդ մասին չի հայտնում։

**Event** — Նշիչ, որը `cudaEventRecord`-ով դրվում է stream-ում։ Event-ը համարվում է ավարտված, երբ ավարտվում է այդ stream-ում դրանից առաջ ուղարկված ամբողջ աշխատանքը։ Օգտագործվում է սպասելու (`cudaEventSynchronize`) և ժամանակ չափելու (`cudaEventElapsedTime`) համար։

**Stream-երի կախվածություն** — Տարբեր stream-երի գործողությունների միջև հերթականություն։ Այն սահմանվում է այսպես. մի stream-ում գրանցվում է event, իսկ մյուս stream-ը սպասում է դրան `cudaStreamWaitEvent`-ով։

**Device-ի կողմից ժամանակաչափում** — Ժամանակի չափում event-ներով, ոչ թե host-ի ժամացույցով։ Այդպես չափվում է GPU-ի աշխատանքի ժամանակը, և արդյունքը չի ներառում host-ի կողմից launch-ի ուշացումը։

**Double buffering** — Մուտքը բաժանել մասերի և օգտագործել բուֆերների և stream-երի երկու կամ ավելի հավաքածու, որպեսզի n-րդ մասի հաշվարկի ընթացքում n+1-րդ մասը պատճենվի device, իսկ n-1-րդը՝ host։ Արդյունքում ընդհանուր ժամանակը մոտենում է փոխանցման և հաշվարկի ժամանակներից մեծագույնին, ոչ թե դրանց գումարին։

**CUDA graph** — Գործողությունների (kernel-ներ, պատճենումներ, host callback-ներ) և դրանց միջև կախվածությունների գրանցված ուղղորդված ացիկլիկ գրաֆ, որը գործարկվում է որպես մեկ ամբողջություն։

**Graph capture** — Stream-ի գործողությունները կատարելու փոխարեն դրանք գրանցել graph-ում։ Գրանցվում են `cudaStreamBeginCapture`-ի և `cudaStreamEndCapture`-ի միջև ուղարկված գործողությունները։

**Instantiation** — Գրանցված graph-ից գործարկման պատրաստ graph ստեղծել `cudaGraphInstantiate`-ով։ Սա արվում է մեկ անգամ. launch-երի ստուգման և նախապատրաստման ծախսը կատարվում է այստեղ, ոչ թե ամեն գործարկման ժամանակ։

**Launch overhead** — Host-ի կողմից մեկ kernel-ի launch ուղարկելու ծախսը։ Graph-ը նվազեցնում է հենց այս ծախսը, ինչը կարևոր է, երբ կարճ kernel-ների ֆիքսված հաջորդականությունը կատարվում է շատ անգամ։

## Ֆունկցիաներ

```c
// Ստեղծում է նոր stream
cudaError_t cudaStreamCreate(cudaStream_t *pStream);

// Pinned հիշողություն։ Առանց դրա cudaMemcpyAsync-ը սինխրոն է
cudaError_t cudaMallocHost(void **ptr, size_t size);

// Սինխրոն և ասինխրոն պատճենում
cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind);
cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind,
                            cudaStream_t stream = 0);

// Event-ներ։ cudaEventElapsedTime-ը *ms-ում գրում է start-ի և end-ի միջև ժամանակը միլիվայրկյաններով
cudaError_t cudaEventCreate(cudaEvent_t *event);
cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream = 0);
cudaError_t cudaEventSynchronize(cudaEvent_t event);
cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end);

// stream-ի հետագա գործողությունները սպասում են event-ի ավարտին
cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags = 0);

// Graph-ի capture, instantiation և launch
cudaError_t cudaStreamBeginCapture(cudaStream_t stream, enum cudaStreamCaptureMode mode);
cudaError_t cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t *pGraph);
cudaError_t cudaGraphInstantiate(cudaGraphExec_t *pGraphExec, cudaGraph_t graph, unsigned long long flags = 0);
cudaError_t cudaGraphLaunch(cudaGraphExec_t graphExec, cudaStream_t stream);

// Սպասում է device-ի ամբողջ աշխատանքի ավարտին
cudaError_t cudaDeviceSynchronize(void);
```

## Կապը Օր 4-ի հետ
Օր 4-ում տեսանք, որ արագությունը հաճախ սահմանափակում է host-device կապը։ Stream-երը այս խնդրի լուծումն են. երբ n-րդ մասը հաշվվում է, n+1-րդ մասը պատճենվում է device, իսկ n-1-րդը՝ host։ Այդ դեպքում ընդհանուր ժամանակը մոտենում է փոխանցման և հաշվարկի ժամանակներից մեծագույնին, ոչ թե դրանց գումարին։

## Պատկեր
![Մեկ default stream, որը հերթով կատարում է H2D պատճենումը, kernel-ը և D2H պատճենումը, և կողքին՝ երկու stream, որտեղ մեկի պատճենումը համընկնում է մյուսի kernel-ի հետ](streams_timeline.svg)

Մեկ stream-ում ամեն ինչ կատարվում է հերթով, ուստի պատճենումների ընթացքում SM-ները պարապ են։ Երկու կամ ավելի stream-ի դեպքում copy engine-ը և SM-ները աշխատում են միաժամանակ։ Ստացված խնայողությունը չափվում է event-ներով։

![Չորս stream-ով pipeline. ամեն մասի H2D պատճենումը, kernel-ը և D2H պատճենումը ժամանակում տեղաշարժված են այնպես, որ հաջորդ մասի պատճենումը համընկնի նախորդ մասի հաշվարկի հետ](async_pipeline.svg)

Լաբորատոր առաջադրանքում կառուցվում է հենց սա. մուտքը բաժանվում է մասերի, ամեն մասի պատճենումը, հաշվարկը և հետ պատճենումը ուղարկվում են իր stream, և հարևան մասերի գործողությունները համընկնում են ժամանակում։ Սա աշխատում է միայն pinned host բուֆերներով (Օր 4)։ Հակառակ դեպքում `cudaMemcpyAsync`-ը դառնում է սինխրոն և այդ մասին չի հայտնում։

![Առանց graph-ի ամեն կրկնությունում CPU-ն ամեն launch-ի համար նորից կատարում է launch-ի ծախսը։ Graph-ով հաջորդականությունը մեկ անգամ գրանցվում և instantiate է արվում, ապա ամեն կրկնությունում գործարկվում է մեկ cudaGraphLaunch-ով](cuda_graph.svg)

Graph-երը չեն արագացնում GPU-ի հաշվարկը։ Դրանք նվազեցնում են CPU-ի ծախսը, որը պահանջվում է launch-երի նույն հաջորդականությունն ամեն անգամ նորից ուղարկելու համար։ Սա նկատելի է միայն այն դեպքում, երբ ֆիքսված pipeline-ը կատարվում է շատ անգամ։

## Գրականություն
- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/pdf/CUDA_C_Programming_Guide.pdf) — 3.2.8. Asynchronous Concurrent Execution · 3.2.8.7. CUDA Graphs
- CUDA C++ Best Practices Guide — Asynchronous Transfers and Overlapping Transfers with Computation
- Nsight Systems — User Guide, timeline view

## Լաբորատոր առաջադրանք
Պատկերը մշակել տողերի մասերով՝ մի քանի stream-ում, որպեսզի փոխանցումը և հաշվարկը համընկնեն ժամանակում։ Ապա այդ հաջորդականությունը գրանցել graph-ում և գործարկել graph-ը։

## Ինքնուրույն աշխատանք
1. Երկու stream-ով համատեղել host-ից device ասինխրոն պատճենումը kernel-ի կատարման հետ։ Համատեղումը ստուգել event-ներով և Nsight Systems-ի timeline-ում։
2. Kernel-ի ժամանակը չափել event-ներով և համեմատել host-ի վրա `<chrono>`-ով կատարված չափման հետ։ Բացատրել տարբերությունը։
3. Stream-երի քանակը 2-ից մեծացնել 4-ի, ապա 8-ի և գրանցել, թե որ քանակից սկսած համատեղումն այլևս չի բարելավվում։
4. Մասերով pipeline-ը գրանցել CUDA graph-ում։ Գործարկել graph-ը 1000 անգամ և համեմատել նույն գործողությունների 1000 հաջորդական launch-ի հետ։
5. Ասինխրոն տարբերակում pinned հատկացումը փոխարինել սովորական `malloc`-ով և չափել, թե ինչ է փոխվում։

## Ինքնաստուգում
Պատասխանները տրված չեն։

1. Ինչու՞ է `cudaMemcpyAsync`-ը վարվում սինխրոն, եթե host-ի բուֆերը pinned չէ։
2. Ի՞նչ սխալ կառաջանա, եթե `cudaEventElapsedTime`-ը կանչեք առանց նախապես `cudaEventSynchronize` կանչելու։
3. Ինչու՞ stream-երի քանակի մեծացումը որոշակի պահից այլևս չի օգնում։
4. Ինչու՞ kernel-ները graph-ում գրանցելը չի արագացնում GPU-ի հաշվարկը։
5. `cudaStreamBeginCapture`-ից հետո գործարկված kernel-ը կատարվու՞մ է։

## Կոդի template
Տես [`template.cu`](template.cu)։
