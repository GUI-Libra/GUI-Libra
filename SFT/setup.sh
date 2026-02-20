conda env create -n py310 python=3.10
conda activate py310
pip install -r requirements.txt
pip uninstall torch torchvision torchaudio 
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0   --index-url https://download.pytorch.org/whl/cu128
pip install flash-attn==2.8.3 --no-build-isolation
pip install -e .
pip install transformers==4.57.3