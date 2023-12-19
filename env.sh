#!/bin/bash

# 设置Miniconda的版本和安装路径 (可以根据实际情况进行修改)
VERSION="latest"
INSTALL_PATH="$HOME/miniconda"

# 下载Miniconda的安装包
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-${VERSION}-Linux-x86_64.sh"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-${VERSION}-MacOSX-x86_64.sh"
else
    echo "Unsupported platform"
    exit 1
fi

echo "Downloading Miniconda..."
curl -O "$MINICONDA_URL"

# 安装Miniconda
echo "Installing Miniconda..."
bash Miniconda3*.sh -b -p "$INSTALL_PATH" -f 

# 添加Conda到环境变量中
echo "Configuring Miniconda..."
echo ". \"$INSTALL_PATH/etc/profile.d/conda.sh\"" >> ~/.bashrc
echo "conda activate base" >> ~/.bashrc

# 更新环境变量
source ~/.bashrc

# 打印Miniconda版本号
echo "Miniconda installed successfully!"
conda --version