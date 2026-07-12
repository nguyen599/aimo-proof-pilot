#!/usr/bin/env bash
set -euo pipefail

DEFAULT_PUBLIC_KEY="ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIAOjr4XeO7SRldvhcTKakqvkdKuqoLSyHLg6vuzWXcD8 pnvmanh2123@gmail.com"
KEY_BLOB="${PUBLIC_KEY:-${SSH_PUBLIC_KEY:-${AUTHORIZED_KEYS:-${SSH_AUTHORIZED_KEYS:-${DEFAULT_PUBLIC_KEY}}}}}"
SSH_PORT="${SSH_PORT:-22}"

mkdir -p /var/run/sshd /root/.ssh
chmod 700 /root/.ssh
printf '%s\n' "${KEY_BLOB}" > /root/.ssh/authorized_keys
chmod 600 /root/.ssh/authorized_keys

ssh-keygen -A >/dev/null 2>&1
sed -i '/^#*Port /d' /etc/ssh/sshd_config
printf 'Port %s\n' "${SSH_PORT}" >> /etc/ssh/sshd_config
sshd -t

printf '[aimo-proof-pilot] starting sshd on port %s\n' "${SSH_PORT}"
exec /usr/sbin/sshd -D -e
