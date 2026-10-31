#!/usr/bin/env bash
# Build inside the server's android-build:36 container (JDK 21, SDK 36, NDK 29) with Gradle 9.8.2.
# usage: android/build.sh [gradle tasks...]   (default: :mimi:assembleRelease :demo:assembleDebug)
# Mounts the whole neural_audio dir (CMake reads ../phone/libmimi) and the shared Gradle cache.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
WORK=$(cd "$HERE/.." && pwd)
CACHE="$HOME/Lab_notebook/android/build/cache"
[ $# -gt 0 ] || set -- :mimi:assembleRelease :demo:assembleDebug
docker run --rm \
  -e GRADLE_USER_HOME=/cache/gradle \
  -e GRADLE_OPTS="-Djava.net.preferIPv4Stack=true -Dorg.gradle.daemon=false" \
  -v "$CACHE/gradle:/cache/gradle" \
  -v "$CACHE/dists/gradle-9.8.2:/opt/gradle:ro" \
  -v "$WORK:/work" \
  -w /work/android \
  android-build:36 \
  /opt/gradle/bin/gradle --no-daemon --console=plain "$@"
