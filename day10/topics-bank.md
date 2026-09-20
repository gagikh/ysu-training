# Topic Examples

Examples for course projects, diploma work and master's theses in AI and computer vision.
Take one as it stands, adapt it, or ignore the list entirely.

Scope decides the category: the first group is implementation and measurement, which
suits a course project; the second group asks a research question and suits diploma work
or a thesis. The supervisor decides which fits which.

Memory and runtime are deliberately not given. Estimate them against the cluster GPU's
own numbers, obtained on Day 1.

---

## Implementation and measurement

### 1. CUDA image filter library
*Days 3–6*

- **Question.** How much do tiling and the separable form gain, from a 3×3 up to a 15×15 kernel, and where does the gain stop?
- **Data.** Any set of images. No specific dataset needed.
- **Sufficient.** Four versions — naive, tiled, separable, NPP. Bandwidth measured for each.
- **Typical mistake.** Timing without warm-up and without `cudaDeviceSynchronize()`.

### 2. Connected components labeling
*Day 7*

- **Question.** When does label propagation beat union-find, as a function of the number and shape of components?
- **Data.** Binary masks, synthetic and from segmentation output.
- **Sufficient.** Both algorithms, correctness checked against a CPU implementation, runtime as a curve against component count rather than a single number.
- **Typical mistake.** Not measuring atomic contention, and so not being able to explain the slowdown.

### 3. Non-maximum suppression
*Day 7*

- **Question.** What fraction of a detection pipeline's time is NMS, and how much of it does the GPU remove?
- **Data.** Output boxes from a pre-trained detector. No training needed.
- **Sufficient.** A CUDA NMS, compared against torchvision's GPU version, with proof the results are identical.
- **Typical mistake.** Using unoptimised CPU code as the baseline and reporting a 100× speedup.

### 4. Decode and resize on the GPU
*Days 4, 9*

- **Question.** What fraction of a training step is JPEG decode and resize, and what does moving them to the GPU with nvJPEG change?
- **Data.** Any large set of JPEGs.
- **Sufficient.** Nsight Systems timeline before and after, GPU utilisation, throughput in images per second for the whole step.
- **Typical mistake.** Measuring decode in isolation and not showing the effect on the full step.

### 5. Histogram equalisation and CLAHE
*Day 7*

- **Question.** How does atomic contention depend on image content — flat sky against rich texture?
- **Data.** Images deliberately chosen to differ in content.
- **Sufficient.** Global and privatised versions, contention measured per image type.
- **Typical mistake.** Averaging over all images and losing the effect being studied.

### 6. Block matching stereo
*Days 5, 6*

- **Question.** How does shared-memory tile size affect disparity computation, and where is the optimum?
- **Data.** Middlebury or KITTI stereo, both with ground truth.
- **Sufficient.** A working disparity map, accuracy against ground truth, a tile-size sweep alongside occupancy.
- **Typical mistake.** Optimising occupancy on a kernel that is bandwidth-bound.

### 7. Precision and accuracy
*Day 9*

- **Question.** How do accuracy and speed change between fp32, tf32 and bf16 for a small network?
- **Data.** A deliberately small dataset, so that repeated runs are possible.
- **Sufficient.** Three modes, at least three runs each, mean and spread of accuracy, and profiler evidence that tensor cores were actually used.
- **Typical mistake.** One run per mode, and a conclusion drawn from random variation.

### 8. Adaptive binarisation of manuscript pages
*Days 4, 5*

- **Question.** How much does Sauvola or Niblack binarisation gain on the GPU using an integral image, over a large page count?
- **Data.** Scanned pages, including digitised Armenian manuscripts.
- **Sufficient.** GPU integral image, speedup measured, visual comparison against OpenCV.
- **Typical mistake.** Not using an integral image, producing an algorithm whose cost depends on window size.

---

## Research question

### 9. Armenian scene text detection and recognition

- **Question.** How much real training data is needed for acceptable quality on Armenian signage, and how far does synthetic data substitute for it?
- **Data.** A collection gathered by the student, plus synthetic generation using Armenian typefaces. Building the dataset is part of the work.
- **Sufficient.** A published dataset, a measurable improvement over a baseline, error analysis by typeface and lighting.
- **Typical mistake.** Most of the time goes to collecting data. The proposal should fix a minimum dataset size and a date after which collection stops.
- No comparable public dataset exists; the result is publishable.

### 10. Layout analysis of Armenian manuscripts

- **Question.** Does line and column segmentation transfer to Armenian manuscripts by fine-tuning a model trained on other writing systems, and how many annotated pages are enough?
- **Data.** Digitised pages with manual annotation.
- **Sufficient.** IoU per line, comparison against existing tools, an error typology by manuscript style.
- **Typical mistake.** Starting from the full page without tiling. The tiling strategy belongs in the proposal.

### 11. High-resolution segmentation under a memory limit

- **Question.** Tiling, gradient checkpointing, or lower resolution — which gives the best accuracy per GB?
- **Data.** Cityscapes or a comparable public dataset.
- **Sufficient.** A curve of accuracy against memory for all three strategies, not three isolated numbers.
- **Typical mistake.** Not fixing a common compute budget, and comparing strategies that were given unequal time.

### 12. Land cover segmentation for Armenia

- **Question.** With what accuracy can land cover be classified from Sentinel-2 multi-spectral imagery, and how does it change over time?
- **Data.** Sentinel-2, public and free. 10–13 channels instead of RGB.
- **Sufficient.** Per-class IoU, analysis of change over time.
- **Typical mistake.** No ground truth. This must be settled at the proposal stage.

### 13. Fused augmentation kernel

- **Question.** How much does fusing an augmentation chain into one kernel raise real training throughput, and for which chains is it justified?
- **Data.** Any classification dataset; the data is a means here, not the subject.
- **Sufficient.** The fused kernel, proof of equivalence against the separate operations, and end-to-end throughput.
- **Typical mistake.** Measuring only kernel time.
- The only topic on the list where the student writes a substantial kernel.

### 14. Object tracking in video

- **Question.** What throughput is available on one GPU for detection plus tracking, and where is the limit — model, decode, or transfers?
- **Data.** MOT or a comparable public video dataset.
- **Sufficient.** FPS by resolution, the effect of stream overlap, and a tracking accuracy metric.
- **Typical mistake.** Counting only inference and calling it real time; including decode halves the figure.

### 15. Inference quantisation

- **Question.** What accuracy does INT8 quantisation cost and what latency does it save, and where is the break-even?
- **Data.** Any existing task with a pre-trained model. Little training is needed.
- **Sufficient.** An accuracy-latency curve rather than a single point, and the effect of calibration.
- **Typical mistake.** Measuring latency at batch 64 and presenting it as an edge-deployment result.

---

## What to ask of a proposal

- The dataset by name and size, the resolution as a number, and where the ground truth comes from.
- A memory estimate compared against one GPU, using the Day 1 numbers.
- A baseline: the best available implementation, not the simplest.
- A definition of success: which metric, above which value, fixed before results are seen.
