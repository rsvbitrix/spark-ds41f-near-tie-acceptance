#!/bin/bash
# curve.sh PREFIX SEEDS SETS SPEC...: one-stream speed at per-request levels on the running engine.
#   SPEC is NAME:TEMPERATURE, e.g. T03:0.997 (typical 0.3), L03:1.003 (lenience 0.3), X:1 (exact).
p=$1; seeds=$2; sets=$3; shift 3; cd ~/spark-night
for spec in "$@"; do
  label=$p-${spec%%:*}; t=${spec##*:}
  python3 nb.py $label --seeds $seeds --sets $sets --extra "{\"temperature\": $t}" </dev/null | grep -E "SUMMARY|busy|Error"
done
echo "curve done $(date +%T)"
