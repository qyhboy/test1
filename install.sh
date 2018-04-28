#!/bin/bash
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
#需要yes
bash Miniconda3-latest-Linux-x86_64.sh 
sudo apt-get update
sudo apt-get install git -y
git clone https://github.com/mli/gluon-tutorials-zh
cd glu

bash
conda env create -n nn 
conda env create -f envi
source activate gluon
source deactivate gluon

pip install https://github.com/mli/notedown/tarball/master
jupyter notebook --NotebookApp.contents_manager_class='notedown.NotedownContentsManager' --allow-root

#8008映射远端8888 ssh -L8008:localhost:8888 ubuntu@ip

