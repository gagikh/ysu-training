# Կլաստերին միանալը

## Միացում

```bash
ssh <user>@<cluster>
```

## Միջավայրի կարգավորում (մեկ անգամ)

`~/.bashrc` ֆայլում ավելացնել հետևյալ տողը.

```bash
export PATH="$PATH":/mnt/weka/shared-cache/miniforge3/envs/indoorolo/bin
```

Կամ նույնը կատարել հրամանով և ստուգել արդյունքը.

```bash
echo 'export PATH="$PATH":/mnt/weka/shared-cache/miniforge3/envs/indoorolo/bin' >> ~/.bashrc
source ~/.bashrc
nvcc --version          # release 12.8
```

## Դասընթացի նյութերը

```bash
git clone https://github.com/gagikh/ysu-training.git
cd ysu-training
```

## Առաջին ստուգումը

```bash
./compile.sh common/bmp_example.cu
sbatch submit.sh bmp_example data/camera.bmp
squeue -u $USER                  # job-ը սպասում է հերթում կամ արդեն աշխատում է
cat logs/report_<job id>.log     # OK: 0 of 262144 pixels wrong
```

## Ֆայլերի պատճենումը կլաստերից

`scp`-ն ֆայլերը պատճենում է ssh-ով։ Հրամանը պետք է գործարկել **ձեր համակարգչից**, ոչ թե կլաստերից.

```bash
scp <user>@<cluster>:~/ysu-training/output.bmp .
scp <user>@<cluster>:~/ysu-training/logs/*.log .
```

Windows 10-ում և 11-ում `ssh`-ը և `scp`-ն հասանելի են PowerShell-ից։

## SLURM

```bash
squeue -u $USER         # ձեր job-երը
scancel <job id>        # դադարեցնել job-ը
```
