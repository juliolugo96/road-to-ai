#!/bin/bash
# 05-ruby.sh - Install Ruby with OpenSSL 1.1 compatibility
# Ruby 2.6.6 requires OpenSSL 1.1, but Amazon Linux 2023 ships OpenSSL 3.
# This script builds OpenSSL 1.1 from source and compiles Ruby against it,
# ensuring runtime stability independent of OS-level OpenSSL updates.
set -euo pipefail

SCRIPT_NAME="$(basename "$0")"
echo "[$SCRIPT_NAME] Starting Ruby setup..."

# ---------- Configuration ----------
RUBY_VERSION="2.6.6"
OPENSSL_VERSION="1.1.1w"
RBENV_ROOT="/usr/local/rbenv"
OPENSSL_PREFIX="/usr/local/openssl-1.1"

# ---------- Idempotency Check ----------
if [[ -x "${RBENV_ROOT}/versions/${RUBY_VERSION}/bin/ruby" ]]; then
  echo "[$SCRIPT_NAME] Ruby ${RUBY_VERSION} is already installed. Skipping."
  exit 0
fi

# ---------- Build OpenSSL 1.1 ----------
if [[ ! -f "${OPENSSL_PREFIX}/lib/libssl.so" ]]; then
  echo "[$SCRIPT_NAME] Building OpenSSL ${OPENSSL_VERSION} from source..."

  BUILD_DIR=$(mktemp -d)
  cd "$BUILD_DIR"

  curl -fsSL "https://www.openssl.org/source/openssl-${OPENSSL_VERSION}.tar.gz" -o openssl.tar.gz
  tar xzf openssl.tar.gz
  cd "openssl-${OPENSSL_VERSION}"

  ./config \
    --prefix="$OPENSSL_PREFIX" \
    --openssldir="$OPENSSL_PREFIX" \
    shared zlib

  make -j"$(nproc)"
  sudo make install

  cd /
  rm -rf "$BUILD_DIR"

  echo "[$SCRIPT_NAME] OpenSSL ${OPENSSL_VERSION} installed to ${OPENSSL_PREFIX}"
else
  echo "[$SCRIPT_NAME] OpenSSL 1.1 already present at ${OPENSSL_PREFIX}. Skipping build."
fi

# ---------- Install rbenv ----------
if [[ ! -d "$RBENV_ROOT" ]]; then
  echo "[$SCRIPT_NAME] Installing rbenv..."
  sudo git clone https://github.com/rbenv/rbenv.git "$RBENV_ROOT"
  sudo git clone https://github.com/rbenv/ruby-build.git "${RBENV_ROOT}/plugins/ruby-build"
fi

export PATH="${RBENV_ROOT}/bin:${RBENV_ROOT}/shims:$PATH"

# ---------- Install Ruby ----------
echo "[$SCRIPT_NAME] Installing Ruby ${RUBY_VERSION} (this may take several minutes)..."

RUBY_CONFIGURE_OPTS="--with-openssl-dir=${OPENSSL_PREFIX}" \
  sudo -E "${RBENV_ROOT}/bin/rbenv" install "$RUBY_VERSION"

sudo "${RBENV_ROOT}/bin/rbenv" global "$RUBY_VERSION"
sudo "${RBENV_ROOT}/bin/rbenv" rehash

# ---------- Install Bundler ----------
echo "[$SCRIPT_NAME] Installing Bundler..."
sudo "${RBENV_ROOT}/versions/${RUBY_VERSION}/bin/gem" install bundler --no-document

sudo "${RBENV_ROOT}/bin/rbenv" rehash

# ---------- Configure System PATH ----------
PROFILE_SCRIPT="/etc/profile.d/rbenv.sh"
if [[ ! -f "$PROFILE_SCRIPT" ]]; then
  sudo tee "$PROFILE_SCRIPT" > /dev/null <<'PROFILE'
export RBENV_ROOT="/usr/local/rbenv"
export PATH="${RBENV_ROOT}/bin:${RBENV_ROOT}/shims:$PATH"
eval "$(rbenv init -)"
PROFILE
fi

# ---------- Verify ----------
RUBY_BIN="${RBENV_ROOT}/versions/${RUBY_VERSION}/bin/ruby"
echo "[$SCRIPT_NAME] Ruby version: $($RUBY_BIN -v)"
echo "[$SCRIPT_NAME] OpenSSL linked: $($RUBY_BIN -ropenssl -e 'puts OpenSSL::OPENSSL_VERSION')"
echo "[$SCRIPT_NAME] Bundler version: $(${RBENV_ROOT}/versions/${RUBY_VERSION}/bin/bundle -v)"
echo "[$SCRIPT_NAME] Done."
