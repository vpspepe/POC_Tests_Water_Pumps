# Download and install Neovim and LazyGit in Google Colab environment
# # Neovim installation (config + binary)
cd /root/.config
git clone https://github.com/vpspepe/lazy_nvim_conf.git
mv lazy_nvim_conf nvim
cd ~
curl -LO https://github.com/neovim/neovim/releases/download/v0.12.4/nvim-linux-x86_64.appimage
sudo chmod +x nvim-linux-x86_64.appimage
sudo mv nvim-linux-x86_64.appimage /usr/local/bin/nvim
cd ~
# # LazyGit installation
LAZYGIT_VERSION=$(curl -s "https://api.github.com/repos/jesseduffield/lazygit/releases/latest" | \grep -Po '"tag_name": *"v\K[^"]*')
LAZYGIT_ARCH=$(uname -m | sed -e 's/aarch64/arm64/')
curl -Lo lazygit.tar.gz "https://github.com/jesseduffield/lazygit/releases/download/v${LAZYGIT_VERSION}/lazygit_${LAZYGIT_VERSION}_Linux_${LAZYGIT_ARCH}.tar.gz"
tar xf lazygit.tar.gz lazygit
sudo install lazygit -D -t /usr/local/bin/
source ~/.bashrc
