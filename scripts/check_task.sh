#!/bin/bash
# usage: bash scripts/check_task.sh TASK   (on Sparky; used by the progress monitor)
# One line per finished / failed run log under the task workspace, the task's final status, low memory.
T=$1
W=/home/sparky/workspace/$T
if [ -d "$W" ]; then
  while IFS= read -r f; do
    n=${f#$W/}; n=${n%.log}
    grep -qE "^(ref|stbl): text" "$f" && echo "完成:$n"
    { grep -qE "Traceback|out of memory|Killed" "$f" || grep "^stbl:" "$f" | grep -qvE "^stbl: (loaded|prefill|step|generation|text)"; } && echo "出错:$n"
  done < <(find "$W" -name "*.log" | sort)
fi
s=$(~/sparky-toolbox/.venv/bin/stb task list 2>/dev/null | grep "^$T " | grep -oE "complete|failed|timeout|stopped")
[ -n "$s" ] && echo "任务:$T:$s"
a=$(free -g | awk '/Mem:/{print $7}')
[ "$a" -lt 12 ] && echo "内存偏紧:可用${a}G"
true
