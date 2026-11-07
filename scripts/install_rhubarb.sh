#!/usr/bin/env bash
set -euo pipefail

version="1.14.0"
asset="Rhubarb-Lip-Sync-${version}-Linux.zip"
url="https://github.com/DanielSWolf/rhubarb-lip-sync/releases/download/v${version}/${asset}"
target="$HOME/.local/share/headroom/rhubarb-${version}"

if [[ -x "$target/rhubarb" ]]; then
  echo "Rhubarb Lip Sync ${version} is already installed at $target/rhubarb"
  exit 0
fi
for tool in curl unzip; do
  command -v "$tool" >/dev/null 2>&1 || { echo "Please install $tool first." >&2; exit 1; }
done

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
mkdir -p "$(dirname "$target")"
echo "Downloading the official Rhubarb Lip Sync ${version} Linux release (~84 MB)…"
curl --fail --location --silent --show-error "$url" -o "$work/$asset"
unzip -q "$work/$asset" -d "$work/extracted" \
  "Rhubarb-Lip-Sync-${version}-Linux/rhubarb" \
  "Rhubarb-Lip-Sync-${version}-Linux/LICENSE.md" \
  "Rhubarb-Lip-Sync-${version}-Linux/README.adoc" \
  "Rhubarb-Lip-Sync-${version}-Linux/res/sphinx/acoustic-model/*" \
  "Rhubarb-Lip-Sync-${version}-Linux/res/sphinx/cmudict-en-us.dict" \
  "Rhubarb-Lip-Sync-${version}-Linux/res/sphinx/en-us.lm.bin" \
  "Rhubarb-Lip-Sync-${version}-Linux/res/sphinx/en-us-phone.lm.bin"
rm -rf "$target"
mv "$work/extracted/Rhubarb-Lip-Sync-${version}-Linux" "$target"
chmod +x "$target/rhubarb"
"$target/rhubarb" --version
echo "Installed to $target/rhubarb"
