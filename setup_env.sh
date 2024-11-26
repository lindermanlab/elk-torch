# do the installs on a gpu!
conda create -n elk_torch
conda activate elk_torch
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia # note that you need the nvidia channel for pytorch-cuda
pip install -r requirements.txt 
pip install -e elk-torch/