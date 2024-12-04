
#before running shell script, make sure to be in h100sh queue for support of updated glibc

WORKING_DIR=$BLACKHOLE/camproj

cd $WORKING_DIR

module load python3/3.10.12

python3 -m venv env_works
source env_works/bin/activate

$BLACKHOLE/camproj/env_works/bin/python -m pip install --upgrade pip

#install packages #we need torch 2.1.0 since prebuilt linux binaries for pytorch3d is limited.
module load cuda/12.1
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip3 install open3d
pip install iopath #dependency for pytorch3d
pip install fvcore #dependency for pytorch3d
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt210/download.html
pip3 install plotly rich plyfile jupyterlab nodejs ipywidgets
pip3 install --upgrade PyMCubes
pip install tensorboard
## for MVG code
pip install pyrender
pip install pymeshfix
pip install trimesh
pip install opencv-python
#dependencies for sugar/3DGS
git clone https://github.com/Anttwo/SuGaR.git --recursive
cd SuGaR/gaussian_splatting/submodules/diff-gaussian-rasterization/
pip install -e .
cd ../simple-knn/
pip install -e .
cd ../../../../
pip install cython PySide6 progressbar

# #SAM
# cd $WORKING_DIR
# git clone https://github.com/facebookresearch/segment-anything.git --recursive
# cd segment-anything; pip install -e .
# wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth


# #XMEM
# cd $WORKING_DIR
# git clone https://github.com/hkchengrex/XMem.git --recursive
# cd XMem
# pip install -r requirements.txt
# mkdir saves
# cd saves
# wget https://github.com/hkchengrex/XMem/releases/download/v1.0/XMem.pth
# cd $WORKING_DIR

# #