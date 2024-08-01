#!/bin/bash

# Install JAX and Flax
pip install jax[tpu] flax[all] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# Install CPU version of tensorflow
pip install tensorflow[cpu] keras orbax optax clu grain augmax transformers opencv-python pandas tensorflow-datasets jupyterlab python-dotenv scikit-learn termcolor wrapt wandb

wget https://secure.nic.cz/files/knot-resolver/knot-resolver-release.deb
sudo dpkg -i knot-resolver-release.deb
sudo apt update
sudo apt install -y knot-resolver
sudo sh -c 'echo `hostname -I` `hostname` >> /etc/hosts'
sudo sh -c 'echo nameserver 127.0.0.1 > /etc/resolv.conf'
sudo systemctl stop systemd-resolved
sudo systemctl start kresd@1.service
sudo systemctl start kresd@2.service
sudo systemctl start kresd@3.service
sudo systemctl start kresd@4.service
sudo systemctl start kresd@5.service
sudo systemctl start kresd@6.service
sudo systemctl start kresd@7.service
sudo systemctl start kresd@8.service
sudo systemctl start kresd@9.service
sudo systemctl start kresd@10.service
sudo systemctl start kresd@11.service
sudo systemctl start kresd@12.service
sudo systemctl start kresd@13.service
sudo systemctl start kresd@14.service
sudo systemctl start kresd@15.service
sudo systemctl start kresd@16.service

# Backup the original resolv.conf
sudo cp /etc/resolv.conf /etc/resolv.conf.bak

# Define the new nameservers
nameservers=(
  "nameserver 127.0.0.1"
  "nameserver 8.8.8.8"
  "nameserver 8.8.4.4"
  "nameserver 76.76.2.0"
  "nameserver 76.76.10.0"
  "nameserver 9.9.9.9"
  "nameserver 1.1.1.1"
  "nameserver 1.0.0.1"
)

# Clear the existing resolv.conf file
sudo sh -c '> /etc/resolv.conf'

# Add each nameserver to the resolv.conf file
for ns in "${nameservers[@]}"; do
  sudo sh -c "echo \"$ns\" >> /etc/resolv.conf"
done
echo "Nameservers added to /etc/resolv.conf"

# Installing and setting up gcsfuse
export GCSFUSE_REPO=gcsfuse-`lsb_release -c -s`
echo "deb [signed-by=/usr/share/keyrings/cloud.google.asc] https://packages.cloud.google.com/apt $GCSFUSE_REPO main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo tee /usr/share/keyrings/cloud.google.asc
sudo apt update
sudo apt install gcsfuse

# Define the file name
gcsfuse_conf="$HOME/gcsfuse.yml"

# Define the contents of the file
gcsfuse_conf_content=$(cat <<EOF
file-cache:
  max-size-mb: 81920
  cache-file-for-range-read: True
metadata-cache:
  stat-cache-max-size-mb: 4096
  ttl-secs: 60
  type-cache-max-size-mb: 4096
file-system:
  kernel-list-cache-ttl-secs: 60
  ignore-interrupts: True
EOF
)

# Create the file and write the contents
echo "$gcsfuse_conf_content" > $gcsfuse_conf

ulimit -n 65535

# Increase the limits of number of open files to unlimited
# Add the limits to /etc/security/limits.conf
limits_conf="/etc/security/limits.conf"
sudo bash -c "cat <<EOF >> $limits_conf
* soft nofile unlimited
* hard nofile unlimited
EOF"

# Create a systemd override directory if it doesn't exist
systemd_override_dir="/etc/systemd/system.conf.d"
sudo mkdir -p $systemd_override_dir

# Add the limits to the systemd service configuration
systemd_limits_conf="$systemd_override_dir/99-nofile.conf"
sudo bash -c "cat <<EOF > $systemd_limits_conf
[Manager]
DefaultLimitNOFILE=infinity
EOF"

# Reload the systemd configuration
sudo systemctl daemon-reload