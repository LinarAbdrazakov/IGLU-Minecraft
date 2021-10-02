FROM iglucontest/env:latest

RUN pip3 install jupyter notebook 
RUN pip3 uninstall -y iglu && pip3 install git+https://github.com/iglu-contest/iglu.git
RUN pip3 install torch torchvision
RUN pip3 install wandb
RUN pip3 install 'ray[rllib]'

