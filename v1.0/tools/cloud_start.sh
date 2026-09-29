#!/bin/bash
# Full reset+run: kill old runners, upload exec script + fresh tree, start ONE detached execution.
set -u
K=~/.ssh/agent-root-mgmt
H=root@100.91.25.4
R=/mnt/workspace/alphaproof-learn
echo "[1/4] kill old runners"
ssh -i $K -o BatchMode=yes -o ConnectTimeout=15 $H 'pkill -f "[e]xec_all.sh"; pkill -f "[n]bconvert"; sleep 1; echo KILLED'
echo "[2/4] upload exec script"
ssh -i $K -o BatchMode=yes $H "cat > $R/exec_all.sh" < /tmp/opencode/v1-stage/tools/exec_all_cloud.sh
echo "[3/4] upload fresh tree"
tar czf - -C /tmp/opencode \
  --exclude='v1-stage/__pycache__' --exclude='*.pyc' \
  --exclude='v1-stage/_pipeline_test*' --exclude='v1-stage/.probe' \
  --exclude='test-probe*.txt' v1-stage 2>/dev/null \
  | ssh -i $K -o BatchMode=yes $H "mkdir -p $R && cat > $R/v1-stage.tgz && ls -l $R/v1-stage.tgz"
echo "[4/4] start ONE detached run"
ssh -i $K -o BatchMode=yes $H "cd $R && (setsid bash exec_all.sh > exec_all.nohup 2>&1 < /dev/null &); sleep 1; echo STARTED"
