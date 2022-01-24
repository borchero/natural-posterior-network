set -o errexit

#--------------------------------------------------------------------------------------------------
echo "Installing dependencies..."
apt-get update
apt-get install -y build-essential git-lfs libbz2-dev libffi-dev liblzma-dev libncurses5-dev \
    libreadline-dev libssl-dev libsqlite3-dev tk-dev zlib1g-dev

#--------------------------------------------------------------------------------------------------
echo "Installing Python..."
git clone https://github.com/pyenv/pyenv.git ~/.pyenv
cd ~/.pyenv && src/configure && make -C src
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.profile
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.profile
echo 'eval "$(pyenv init --path)"' >> ~/.profile
source ~/.profile
pyenv install 3.9.9
echo "3.9.9" > ~/.python-version
cd ..

#--------------------------------------------------------------------------------------------------
echo "Installing Poetry..."
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
source $HOME/.poetry/env
poetry config virtualenvs.in-project true

#--------------------------------------------------------------------------------------------------
echo "Installing CUDA..."
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
apt-key adv --fetch-keys \
    https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
add-apt-repository \
    "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
apt-get update
apt-get -y install cuda

#--------------------------------------------------------------------------------------------------
echo "Rebooting..."
reboot
