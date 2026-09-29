import io, sys

p = "/tmp/opencode/v1-stage/04-价值头/01-价值头原理.md"
s = io.open(p, encoding="utf-8").read()

i7 = s.index("## 7. 与 MCTS")
i8 = s.index("## 8. 源码对照")
assert i8 < i7, "already ordered"

# section 7 block starts at the preceding '---\n\n'
head = s[:i8]
mid = s[i8:i7]           # sections 8 and 9
sec7 = s[i7:]
# trim trailing separator from head (the '---' that preceded section 7)
sep = "---\n\n"
sec7 = sec7
# ensure head ends cleanly (remove trailing whitespace) and re-add separator
new = head.rstrip("\n") + "\n\n---\n\n" + sec7.rstrip("\n") + "\n"
io.open(p, "w", encoding="utf-8").write(new)
print("OK", len(new))
