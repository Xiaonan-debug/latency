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


Loading original agent...
/home/xluobd/latency/ql_eye.py:760: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  checkpoint = torch.load(filepath)
Traceback (most recent call last):
  File "/home/xluobd/latency/compare_latency.py", line 334, in <module>
    run_latency_comparison(agent_path, pruning_amount=0.7)
  File "/home/xluobd/latency/compare_latency.py", line 165, in run_latency_comparison
    agent = load_agent(agent_path)
            ^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xluobd/latency/ql_eye.py", line 760, in load_agent
    checkpoint = torch.load(filepath)
                 ^^^^^^^^^^^^^^^^^^^^
  File "/home/xluobd/.pyenv/versions/3.11.14/lib/python3.11/site-packages/torch/serialization.py", line 1384, in load
    return _legacy_load(
           ^^^^^^^^^^^^^
  File "/home/xluobd/.pyenv/versions/3.11.14/lib/python3.11/site-packages/torch/serialization.py", line 1628, in _legacy_load
    magic_number = pickle_module.load(f, **pickle_load_args)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
_pickle.UnpicklingError: invalid load key, 'v'.
