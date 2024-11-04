python -m venv venv
venv/Scripts/activate
pip install torch==2.3.1+cu121 torchvision==0.18.1+cu121 torchaudio==2.3.1+cu121 -f https://download.pytorch.org/whl/torch_stable.html
python -m spacy download en_core_web_sm
python -m pip install -r requirements.txt