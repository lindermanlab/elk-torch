# elk-torch
Parallelizing nonlinear dynamics--in PyTorch

This repo uses techniques to parallelize nonliner RNNs, such as DEER from Lim et al 2024 ([paper](https://arxiv.org/abs/2309.12252), [code](https://github.com/machine-discovery/deer)) and ELK from Gonzalez et al 2024 ([paper](https://arxiv.org/abs/2407.19115), [code](https://github.com/lindermanlab/elk)). However, unlike these implementations, this repo is in pytorch!

Actively under development. Currently only provides the quasi algorithms. PRs for the full algorithms (with efficient associative parallel scans) welcome!

## Install instructions

### For Stanford Sherlock

Here is the original procedure I followed to be able to run on Sherlock.

Note that I automatically load python 3.12.1 in my Sherlock.
I then set up a miniconda version of python, following the instructions in the lab manual. Note that I requested a compute node in order to actually install miniconda.
```
mkdir -p "$SCRATCH"/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O "$SCRATCH"/miniconda3/miniconda.sh
bash "$SCRATCH"/miniconda3/miniconda.sh -b -u -p "$SCRATCH"/miniconda3
rm -rf "$SCRATCH"/miniconda3/miniconda.sh
"$SCRATCH"/miniconda3/bin/conda init
```
reload the shell
just to double check:
```
which conda  # Expect: /scratch/users/<sunetid>/miniconda3/bin/conda
```
Next, we need to ensure that environment paths are prepended to $PATH; this is not automatically done on Linux systems (see issue). To do this, edit your ~/.bashrc file. At the bottom of the file, you will see a code block starting with # >>> conda initialize >>> (this was added when you ran conda init above). Either
Add `conda deactivate` after this block.
Now, restart your shell by running source ~/.bashrc.
Then, I activated this miniconda environment. I then ran `conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia`
At this point, the python had been upgraded to 3.12.7
Then, I ran `pip install ninja accelerated-scan` (this means that [proger](https://github.com/proger/accelerated-scan) is a dependency).
Finally, I ran `pip install tqdm numpy wandb`.
Note that to get this to run, I also had to `module load cuda/12.4`
