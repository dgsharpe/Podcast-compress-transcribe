#!/data/data/com.termux/files/usr/bin/bash
# Real-time-paced Mimi decode on the phone with temperature logging (runs in Termux, ~/nac).
#   phase idle (IDLE s) -> load (LOAD s: dec_tx on CPU + deconly on GPU, one 2 s chunk each per PERIOD s)
#   -> cool (COOL s). Logs every 10 s to thermal_<tag>.csv:
#   t_s, phase, cpu_max, gpu_max, sys_therm, battery  (degrees C)
# usage: bash thermal_paced.sh TAG [IDLE LOAD COOL PERIOD]
set -u
cd ~/nac
TAG=${1:-paced}; IDLE=${2:-120}; LOAD=${3:-600}; COOL=${4:-180}; PERIOD=${5:-2.0}
OUT=thermal_$TAG.csv
z() { cat /sys/class/thermal/thermal_zone$1/temp 2>/dev/null || echo 0; }
zmax() { local m=0 v; for i in "$@"; do v=$(z $i); [ "$v" -gt "$m" ] && m=$v; done; echo $m; }
CPU="5 6 7 8 9 10 11 12 13 14 15 16 24 25 26 27"
GPU="36 37 38 39 40 41 42 43 44 45 46"
log() {
  printf '%s,%s,%.1f,%.1f,%.1f,%.1f\n' "$(( $(date +%s) - T0 ))" "$1" \
    "$(zmax $CPU)e-3" "$(zmax $GPU)e-3" "$(z 55)e-3" "$(z 68)e-3" >> "$OUT"
}
echo "t_s,phase,cpu_max,gpu_max,sys_therm,battery" > "$OUT"
T0=$(date +%s)
end=$((T0 + IDLE)); while [ "$(date +%s)" -lt $end ]; do log idle; sleep 10; done
# Pace: run_delay = PERIOD minus the measured per-chunk time (~0.11 s CPU, ~0.05 s GPU)
NR=$(awk -v l=$LOAD -v p=$PERIOD 'BEGIN{print int(l/p)}')
./benchmark_model --graph=mimi_dec_tx_fp16.tflite --use_xnnpack=false --num_threads=4 \
  --num_runs=$NR --warmup_runs=1 --run_delay=$(awk -v p=$PERIOD 'BEGIN{print p-0.11}') \
  --max_secs=$((LOAD + 60)) > bench_cpu_$TAG.txt 2>&1 &
P1=$!
./benchmark_model --graph=mimi_deconly_fp16.tflite --use_gpu=true \
  --num_runs=$NR --warmup_runs=1 --run_delay=$(awk -v p=$PERIOD 'BEGIN{print p-0.05}') \
  --max_secs=$((LOAD + 60)) > bench_gpu_$TAG.txt 2>&1 &
P2=$!
while kill -0 $P1 2>/dev/null || kill -0 $P2 2>/dev/null; do
  log load; sleep 10
  [ $(( $(date +%s) - T0 )) -gt $((IDLE + LOAD + 300)) ] && { kill $P1 $P2 2>/dev/null; break; }
done
end=$(( $(date +%s) + COOL )); while [ "$(date +%s)" -lt $end ]; do log cool; sleep 10; done
echo DONE >> "$OUT"
