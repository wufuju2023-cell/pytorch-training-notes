#!/bin/bash
# 【脚本｜F-tools-pipeline_test】v1.0/tools/pipeline_test.sh — 云端 notebook 执行链路冒烟测试
set -e
R=/mnt/workspace/alphaproof-learn/tmp
D=/tmp/opencode/v1-stage
K=~/.ssh/agent-root-mgmt
ssh -i $K -o BatchMode=yes -o ConnectTimeout=15 root@100.91.25.4 "mkdir -p $R"
scp -i $K -o BatchMode=yes "$D/_pipeline_test.ipynb" root@100.91.25.4:$R/
ssh -i $K -o BatchMode=yes root@100.91.25.4 "cd $R && timeout 400 jupyter nbconvert --to notebook --execute --inplace _pipeline_test.ipynb --ExecutePreprocessor.timeout=300 2>&1 | tail -5; echo EXEC_RC=\$?"
scp -i $K -o BatchMode=yes root@100.91.25.4:$R/_pipeline_test.ipynb "$D/_pipeline_test_executed.ipynb"
python3 - <<'PY'
import json
nb=json.load(open("/tmp/opencode/v1-stage/_pipeline_test_executed.ipynb"))
for c in nb["cells"]:
    for o in c.get("outputs",[]):
        print("OUT:", "".join(o.get("text",[])) or o.get("name","<non-text>"))
PY
echo PIPELINE_TEST_DONE
