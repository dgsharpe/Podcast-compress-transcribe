#!/usr/bin/env bash
# Build pocket-casts-android (cloned at neural_audio/pocket-casts-android) in android-build:37.
# usage: android/pc_build.sh [gradle tasks...]   (default: :app:assembleDebugProd)
# The whole neural_audio dir is mounted so the patched app can include android/mimi (+ phone/libmimi).
# cache/m2 keeps Robolectric's android-all jars between runs (re-downloading them got HTTP 429 from Maven Central).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
WORK=$(cd "$HERE/.." && pwd)
CACHE="$HOME/Lab_notebook/android/build/cache"
[ $# -gt 0 ] || set -- :app:assembleDebugProd
docker run --rm \
  -e GRADLE_USER_HOME=/cache/gradle \
  -e GRADLE_OPTS="-Djava.net.preferIPv4Stack=true -Xmx6g" \
  -v "$CACHE/gradle:/cache/gradle" \
  -v "$CACHE/m2:/root/.m2" \
  -v "$CACHE/dists/gradle-9.8.2:/opt/gradle:ro" \
  -v "$WORK:/work" \
  -w /work/pocket-casts-android \
  android-build:37 \
  /opt/gradle/bin/gradle --no-daemon --console=plain -Dorg.gradle.jvmargs=-Xmx6g "$@"
