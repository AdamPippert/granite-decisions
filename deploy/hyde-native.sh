#!/usr/bin/env bash
set -euo pipefail

# Run on Hyde. No containers, sudo, system package changes, or existing-service edits.
GD_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
GD_NATIVE_ROOT="${GD_NATIVE_ROOT:-$GD_PROJECT_ROOT/.native}"
GD_LLAMA_COMMIT="${GD_LLAMA_COMMIT:-aa39d7a3e145a88202793a89462d65e94a5fc25f}"
GD_JOBS="${GD_JOBS:-8}"
mode="${1:-inspect}"

if [[ "$mode" != inspect && "$mode" != build ]]; then
  echo 'Usage: bash deploy/hyde-native.sh [inspect|build]' >&2
  exit 2
fi
hostname
cat /etc/fedora-release
for tool in git cmake gcc g++ hipconfig rocminfo python3; do
  command -v "$tool"
done
hipconfig --version
rocminfo | awk '/Name:.*gfx/ || /Marketing Name:/'
ls -l /dev/kfd /dev/dri/renderD*
[[ -r /dev/kfd && -w /dev/kfd ]] || { echo 'Current user needs read/write access to /dev/kfd.' >&2; exit 2; }
gd_render_access=false
for gd_render in /dev/dri/renderD*; do
  if [[ -r "$gd_render" && -w "$gd_render" ]]; then gd_render_access=true; fi
done
if [[ "$gd_render_access" != true ]]; then
  echo 'Current user needs read/write access to a render device.' >&2
  exit 2
fi
if [[ "$mode" == inspect ]]; then
  exit 0
fi

mapfile -t gd_targets < <(rocminfo | awk '$1 == "Name:" && $2 ~ /^gfx[0-9]+/ && $2 != "gfx000" {print $2}' | sort -u)
if [[ -n "${GD_GPU_TARGET:-}" ]]; then
  gd_target="$GD_GPU_TARGET"
  if [[ ! " ${gd_targets[*]} " == *" $gd_target "* ]]; then
    echo 'GD_GPU_TARGET must match a target reported by rocminfo.' >&2
    exit 2
  fi
elif [[ "${#gd_targets[@]}" == 1 ]]; then
  gd_target="${gd_targets[0]}"
else
  echo 'Set GD_GPU_TARGET to the actual target reported by rocminfo.' >&2
  exit 2
fi
mkdir -p "$GD_NATIVE_ROOT"
gd_source="$GD_NATIVE_ROOT/llama.cpp"
if [[ ! -d "$gd_source" ]]; then
  git init "$gd_source"
  git -C "$gd_source" remote add origin https://github.com/ggml-org/llama.cpp.git
  git -C "$gd_source" fetch --depth 1 origin "$GD_LLAMA_COMMIT"
  git -C "$gd_source" checkout --detach FETCH_HEAD
fi
if [[ "$(git -C "$gd_source" rev-parse HEAD)" != "$GD_LLAMA_COMMIT" ]]; then
  echo 'Existing source differs; use a fresh GD_NATIVE_ROOT to preserve it.' >&2
  exit 2
fi
gd_patch="$GD_PROJECT_ROOT/patches/llama-grammar-probabilities.patch"
if git -C "$gd_source" diff HEAD --quiet && [[ -z "$(git -C "$gd_source" ls-files --others --exclude-standard)" ]]; then
  git -C "$gd_source" apply --check "$gd_patch"
  git -C "$gd_source" apply "$gd_patch"
else
  # Permit only this exact patch on rerun; preserve all other work.
  if ! cmp -s <(git -C "$gd_source" diff HEAD --binary) "$gd_patch" || [[ -n "$(git -C "$gd_source" ls-files --others --exclude-standard)" ]]; then
    echo 'Existing source has other changes; use a fresh GD_NATIVE_ROOT.' >&2
    exit 2
  fi
fi
HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
  cmake -S "$gd_source" -B "$GD_NATIVE_ROOT/build-rocm" \
    -DCMAKE_BUILD_TYPE=Release -DGGML_HIP=ON -DGPU_TARGETS="$gd_target" \
    -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_OPENSSL=OFF
cmake --build "$GD_NATIVE_ROOT/build-rocm" --target llama-server llama-bench -j "$GD_JOBS"
"$GD_NATIVE_ROOT/build-rocm/bin/llama-server" --version
"$GD_NATIVE_ROOT/build-rocm/bin/llama-server" --list-devices
printf 'Native server: %s\n' "$GD_NATIVE_ROOT/build-rocm/bin/llama-server"
