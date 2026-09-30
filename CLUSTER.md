# Կլաստերին միանալը

## Միացում

```bash
ssh <user>@<cluster>
```

## Դասընթացի նյութերը և միջավայրը (մեկ անգամ)

```bash
git clone https://github.com/gagikh/ysu-training.git
cd ysu-training
source setup.sh         # release 12.8
```

`setup.sh`-ը `~/.bashrc`-ում ավելացնում է CUDA 12.8-ի `PATH`-ը (եթե այն արդեն չկա) և միանգամից փոխում է ընթացիկ terminal-ի `PATH`-ը։ Այն պետք է կանչել `source`-ով, ոչ թե `bash`-ով, այլապես ընթացիկ terminal-ի `PATH`-ը չի փոխվի։

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
