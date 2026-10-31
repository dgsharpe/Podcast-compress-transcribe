#!/data/data/com.termux/files/usr/bin/bash
# Real-time-paced mimi_dec run (NPU transformer + GPU conv) with temperature logging (~/nac/v2).
#   idle IDLE s -> decode WINDOWS windows, one per 1.52 s (= real time) -> cool COOL s.
# Logs every 10 s to thermal_<tag>.csv: t_s, phase, cpu_max, gpu_max, nsp_max, sys_therm, battery.
# usage: bash thermal_dec.sh TAG [IDLE WINDOWS COOL]
set -u
cd ~/nac/v2
TAG=${1:-dec}; IDLE=${2:-60}; NW=${3:-395}; COOL=${4:-120}
OUT=thermal_$TAG.csv
z() { cat /sys/class/thermal/thermal_zone$1/temp 2>/dev/null || echo 0; }
zmax() { local m=0 v; for i in "$@"; do v=$(z $i); [ "$v" -gt "$m" ] && m=$v; done; echo $m; }
CPU="5 6 7 8 9 10 11 12 13 14 15 16 24 25 26 27"
GPU="36 37 38 39 40 41 42 43 44 45 46"
NSP="28 29 30 31 32 33 34 35"
log() {
  printf '%s,%s,%.1f,%.1f,%.1f,%.1f,%.1f\n' "$(( $(date +%s) - T0 ))" "$1" "$(zmax $CPU)e-3" \
    "$(zmax $GPU)e-3" "$(zmax $NSP)e-3" "$(z 55)e-3" "$(z 68)e-3" >> "$OUT"
}
export LD_LIBRARY_PATH=$PWD/npu:$PWD/lib ADSP_LIBRARY_PATH="$PWD/npu;/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp;/vendor/dsp"
echo "t_s,phase,cpu_max,gpu_max,nsp_max,sys_therm,battery" > "$OUT"
T0=$(date +%s)
end=$((T0 + IDLE)); while [ "$(date +%s)" -lt $end ]; do log idle; sleep 10; done
TIMEFORMAT="wall %R s, user %U s, sys %S s"
{ time ./mimi_dec --codes episode_q32.i16 --tx npu --conv gpu --libdir $PWD/lib --npudir $PWD/npu \
    --cache $PWD/cache --max-windows $NW --pace 1.52 > dec_$TAG.log 2>&1 ; } 2> time_$TAG.txt &
P=$!
while kill -0 $P 2>/dev/null; do log load; sleep 10; done
end=$(( $(date +%s) + COOL )); while [ "$(date +%s)" -lt $end ]; do log cool; sleep 10; done
echo DONE >> "$OUT"
