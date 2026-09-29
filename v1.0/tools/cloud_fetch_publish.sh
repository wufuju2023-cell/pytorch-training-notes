#!/bin/bash
# 【脚本｜F-tools-cloud_fetch_publish】v1.0/tools/cloud_fetch_publish.sh — 下载云端执行结果并发布到 my-new-linux
# Step 3+4: after cloud execution finishes (exec_all.log has ALL_DONE),
# download executed tree and publish to my-new-linux docs.
set -u
K=~/.ssh/agent-root-mgmt
H=root@100.91.25.4
OUT=/tmp/opencode/v1-executed
echo "[3/4] downloading executed tree..."
rm -rf "$OUT" && mkdir -p "$OUT"
ssh -i $K -o BatchMode=yes $H 'tar czf - -C /mnt/workspace/alphaproof-learn/stage v1-stage' > "$OUT/v1-executed.tgz"
tar xzf "$OUT/v1-executed.tgz" -C "$OUT"
echo "[4/4] publishing to my-new-linux v1.0/ ..."
scp -o BatchMode=yes -r "$OUT/v1-stage/." "a@my-new-linux:/home/a/文档/PyTorch和模型训练源码学习/v1.0/"
echo DONE
