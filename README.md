wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh
bash Miniforge3-Linux-aarch64.sh -b
~/miniforge3/bin/conda init
source ~/.bashrc
conda create -n myenv python=3.11 -y
conda activate myenv
conda install pytorch cpuonly numpy scipy matplotlib -c pytorch -y
pip install gym

conda activate myenv
python compare_latency.py


sudo apt install -y \
  build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev \
  libsqlite3-dev libffi-dev liblzma-dev xz-utils tk-dev curl git

  git clone https://github.com/pyenv/pyenv.git ~/.pyenv

  echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
exec "$SHELL"

cat /etc/os-release | grep -E 'PRETTY_NAME|VERSION_CODENAME'
