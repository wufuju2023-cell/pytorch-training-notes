# 【源代码｜F-misc-validate_nb】v1.0/_validate_nb.py — 校验 notebook 合法性与可编译性
import json, sys, py_compile, tempfile, os
for path in sys.argv[1:]:
    nb = json.load(open(path, encoding="utf-8"))
    assert nb.get("nbformat") == 4, path
    assert isinstance(nb.get("cells"), list)
    for i, c in enumerate(nb["cells"]):
        assert c.get("cell_type") in ("code", "markdown")
        assert isinstance(c.get("source"), list)
        if c["cell_type"] == "code":
            src = "".join(c["source"])
            fn = tempfile.mktemp(suffix=".py")
            open(fn, "w", encoding="utf-8").write(src)
            try:
                py_compile.compile(fn, doraise=True)
            except Exception as e:
                print("SYNTAX FAIL", path, "cell", i, e); sys.exit(1)
            os.unlink(fn)
    print("NB OK", path, "cells", len(nb["cells"]))
