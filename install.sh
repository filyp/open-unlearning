# git clone https://github.com/filyp/open-unlearning.git && cd open-unlearning && bash install.sh
apt install python3-pip python3-venv -y
python3 -m venv .venv
source .venv/bin/activate
pip install ".[lm_eval]" && pip install --no-build-isolation flash-attn==2.8.3