#!/usr/bin/env python3
# 【源代码｜F-tools-tag_graph】v1.0/tools/tag_graph.py — Tag 图工具：extract / check / render
# 相关文档：【文档｜DOC-STYLE】（§9）、【文档｜DOC-TAGS】（tags/SCHEMA.md）
"""Tag 图工具（纯标准库）。

用法：
    python3 v1.0/tools/tag_graph.py <extract|check|render> <root>

子命令：
    extract  扫描全部 md/py/ipynb，生成 tags/registry.tsv 与 tags/relations.tsv（去重、稳定排序、幂等）。
    check    校验 SCHEMA.md 的五条规则；打印 `errors: N; warnings: M` 与明细；有 error 时退出码非零。
    render   生成 tags/TAG-INDEX.md（总表按 doccode 分组 + Mermaid 图）、tags/graph.json、tags/graph.mmd。

关系抽取（初版，宁少勿臆造）：
    refs        正文【…】/[...] 引用（md/py 注释/notebook markdown cell）；src 为所在文档/代码/notebook 的 Tag。
    requires    「前置知识」行中的【…】引用 + 大纲 §3「既有位置 → 预备条目」映射（Py 章缺失则留空）。
    corresponds 规范 §3–§5 登记表与代码/notebook 的实际位置：章节目录唯一可判时 doccode ↔ F-/N-，
                另含预备篇 F-pre-pyN/N1x ↔ PY<n> 与 notebook 内显式提及的 DOC 标记。
    adapted_from external.tsv 中登记且被 registry.ext 指向的条目。
"""

from __future__ import annotations

import json
import os
import re
import sys

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

PREFIX2NAME = {
    "D": "定义",
    "T": "定理",
    "P": "命题",
    "L": "引理",
    "C": "推论",
    "E": "例",
    "R": "注",
    "A": "算法",
    "Cd": "代码",
    "Ex": "练习",
    "Pf": "证明",
}
NAME2PREFIX = {v: k for k, v in PREFIX2NAME.items()}

RELATIONS = [
    "requires", "refs", "proves", "corresponds", "impl", "example_of",
    "generalizes", "specializes", "equiv", "version_of", "supersedes",
    "evolves_to", "adapted_from", "seealso",
]

ITEM_TAG_RE = re.compile(r"^(D|T|P|L|C|E|R|A|Cd|Ex|Pf)-([A-Za-z0-9]+)\.([0-9]+)\.([0-9]+)$")
F_TAG_RE = re.compile(r"^F-[A-Za-z0-9]+(?:-[A-Za-z0-9_]+)*(?:\.[A-Za-z0-9_]+)*$")
N_BASE_RE = re.compile(r"^N-[0-9]+$")
N_CELL_RE = re.compile(r"^N-[0-9]+\.(?:c|m)[0-9]+$")
DOC_TAG_RE = re.compile(r"^DOC-[A-Za-z0-9]+$")

SKIP_DIRS = {".git", "__pycache__", ".ipynb_checkpoints", "node_modules", ".venv", ".mypy_cache"}
GENERATED_MD = {"TAG-INDEX.md"}

REGISTRY_HEADER = ["tag", "type", "doccode", "locator", "title", "status", "version", "ext"]
RELATIONS_HEADER = ["src", "relation", "dst", "note", "since"]
EXTERNAL_HEADER = ["ext_id", "url", "license", "usage", "note"]


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------

def natural_key(text: str):
    return tuple((0, int(p)) if p.isdigit() else (1, p.lower()) for p in re.split(r"([0-9]+)", text))


def sanitize(field: str) -> str:
    field = "" if field is None else str(field)
    return field.replace("\t", " ").replace("\n", " ").replace("\r", " ").strip()


def resolve_root(root: str):
    """返回 (root_abs, content_root, tags_dir)。root 可为仓库根或 v1.0 目录。"""
    root_abs = os.path.abspath(root)
    nested = os.path.join(root_abs, "v1.0", "tags")
    plain = os.path.join(root_abs, "tags")
    if os.path.isdir(nested):
        content = os.path.join(root_abs, "v1.0")
        return root_abs, content, nested
    if os.path.isdir(plain):
        return root_abs, root_abs, plain
    # 都缺失：优先视 root 为 v1.0（tags 将新建在 root/tags）
    if os.path.basename(root_abs) == "v1.0":
        return root_abs, root_abs, plain
    return root_abs, root_abs, plain


def iter_source_files(content: str):
    for dirpath, dirnames, filenames in os.walk(content):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for name in sorted(filenames):
            if name in GENERATED_MD:
                continue
            if name.endswith((".md", ".py", ".ipynb")):
                yield os.path.join(dirpath, name)


def code_spans(line: str):
    return [(m.start(), m.end()) for m in re.finditer(r"`[^`\n]*`", line)]


def in_spans(pos: int, spans) -> bool:
    return any(s <= pos < e for s, e in spans)


def iter_md_lines(text: str):
    infence = False
    for i, line in enumerate(text.split("\n"), 1):
        if re.match(r"^\s*```", line):
            infence = not infence
            continue
        if not infence:
            yield i, line


def title_after_paren(line: str, endpos: int) -> str:
    m = re.match(r"（([^）]*)）", line[endpos:])
    return m.group(1).strip() if m else ""


def ref_tag_from_plain(text: str):
    m = re.match(
        r"^(定义|定理|命题|引理|推论|例|注|算法|代码|练习|证明)\s*"
        r"([A-Za-z0-9]+(?:\.[0-9]+){1,3})$",
        text.strip(),
    )
    if not m:
        return None
    return NAME2PREFIX[m.group(1)] + "-" + m.group(2)


def tag_doccode(tag: str) -> str:
    if tag.startswith("DOC-"):
        return tag[4:]
    m = ITEM_TAG_RE.match(tag)
    if m:
        return m.group(2)
    if F_TAG_RE.match(tag):
        return re.split(r"[-.]", tag[2:])[0]
    if N_BASE_RE.match(tag) or N_CELL_RE.match(tag):
        return "N"
    return ""


def tag_type(tag: str) -> str:
    if tag.startswith("DOC-"):
        return "DOC"
    m = ITEM_TAG_RE.match(tag)
    if m:
        return m.group(1)
    if F_TAG_RE.match(tag):
        return "F"
    if N_BASE_RE.match(tag) or N_CELL_RE.match(tag):
        return "N"
    return "?"


# ---------------------------------------------------------------------------
# 抽取
# ---------------------------------------------------------------------------

class ScanResult:
    def __init__(self):
        self.defs = {}              # tag -> record
        self.duplicates = []        # (tag, rel, line)
        self.relations = set()      # (src, rel, dst, note)
        self.files = []             # dict per file
        self.md_without_doc = []    # rel


def add_def(result: ScanResult, rec):
    tag = rec["tag"]
    if tag in result.defs:
        result.duplicates.append((tag, rec["locator"]))
        return
    result.defs[tag] = rec


def add_rel(result: ScanResult, src, relation, dst, note):
    if not src or not dst or src == dst:
        return
    result.relations.add((src, relation, dst, note))


def scan_file(result: ScanResult, path: str, rel: str):
    if path.endswith(".md"):
        scan_md(result, path, rel)
    elif path.endswith(".py"):
        scan_py(result, path, rel)
    elif path.endswith(".ipynb"):
        scan_ipynb(result, path, rel)


def scan_md(result: ScanResult, path: str, rel: str):
    text = open(path, encoding="utf-8").read()
    heading = ""
    for line in text.split("\n"):
        m = re.match(r"^#\s+(.*\S)\s*$", line)
        if m:
            heading = m.group(1).strip()
            break

    info = {"rel": rel, "kind": "md", "doc": None, "f": None, "n": None, "doc_mentions": set()}
    refs = set()
    requires = set()

    for lineno, line in iter_md_lines(text):
        spans = code_spans(line)
        # 文档 Tag
        for m in re.finditer(r"【文档｜(DOC-[A-Za-z0-9]+)】", line):
            if in_spans(m.start(), spans):
                continue
            if info["doc"] is None:
                info["doc"] = m.group(1)
                add_def(result, {
                    "tag": m.group(1), "locator": rel, "title": heading,
                    "status": "active", "version": "v1.0", "ext": "",
                })
            break
        # 条目定义（保留 inline code，用于标题）
        for m in re.finditer(r"【([^】｜]+)｜([^】]+)】", line):
            if in_spans(m.start(), spans):
                continue
            label, tag = m.group(1).strip(), m.group(2).strip()
            parts = label.split()
            if not parts or parts[0] not in NAME2PREFIX:
                continue
            if not ITEM_TAG_RE.match(tag):
                continue
            add_def(result, {
                "tag": tag, "locator": f"{rel}#L{lineno}",
                "title": title_after_paren(line, m.end()),
                "status": "active", "version": "v1.0", "ext": "",
            })
        # 引用（raw line，跳过 inline code）
        for m in re.finditer(r"【([^】]+)】", line):
            if in_spans(m.start(), spans):
                continue
            body = m.group(1)
            if "｜" in body:
                for side in body.split("｜"):
                    side = side.strip()
                    if ITEM_TAG_RE.match(side) or DOC_TAG_RE.match(side):
                        refs.add(side)
            else:
                tag = ref_tag_from_plain(body)
                if tag:
                    refs.add(tag)
        for m in re.finditer(r"\[([A-Za-z]{1,3}-[A-Za-z0-9]+(?:\.[0-9]+){0,3})\]", line):
            if in_spans(m.start(), spans):
                continue
            refs.add(m.group(1))
        # 前置知识行
        if "前置知识" in line:
            stripped = re.sub(r"`[^`\n]*`", "", line)
            for m in re.finditer(r"【([^】]+)】", stripped):
                body = m.group(1)
                if "｜" in body:
                    for side in body.split("｜"):
                        side = side.strip()
                        if ITEM_TAG_RE.match(side) or DOC_TAG_RE.match(side):
                            requires.add(side)
                else:
                    tag = ref_tag_from_plain(body)
                    if tag:
                        requires.add(tag)

    info["refs"] = refs
    info["requires"] = requires
    result.files.append(info)
    if info["doc"] is None:
        result.md_without_doc.append(rel)
    else:
        for dst in sorted(refs):
            add_rel(result, info["doc"], "refs", dst, "auto:refs")
        for dst in sorted(requires):
            add_rel(result, info["doc"], "requires", dst, "auto:requires")


def scan_py(result: ScanResult, path: str, rel: str):
    text = open(path, encoding="utf-8").read()
    info = {"rel": rel, "kind": "py", "doc": None, "f": None, "n": None, "doc_mentions": set()}
    refs = set()
    requires = set()

    for lineno, line in enumerate(text.split("\n"), 1):
        for m in re.finditer(r"【([^】]+)】", line):
            body = m.group(1)
            if "｜" in body:
                a, b = [x.strip() for x in body.split("｜", 1)]
                tag = kind = None
                if F_TAG_RE.match(a):
                    tag, kind = a, b
                elif F_TAG_RE.match(b):
                    tag, kind = b, a
                if tag is not None:
                    after = line[m.end():].strip()
                    if "—" in after:
                        after = after.split("—")[-1].strip()
                    add_def(result, {
                        "tag": tag, "locator": f"{rel}#L{lineno}",
                        "title": after, "status": "active", "version": "v1.0", "ext": "",
                    })
                    if info["f"] is None and "." not in tag:
                        info["f"] = tag
                    continue
                # 不是 F- 定义：按引用处理（取 md 条目 / DOC 一侧）
                for side in (a, b):
                    if ITEM_TAG_RE.match(side) or DOC_TAG_RE.match(side):
                        refs.add(side)
            else:
                tag = ref_tag_from_plain(body)
                if tag:
                    refs.add(tag)
        if "前置知识" in line:
            for m in re.finditer(r"【([^】]+)】", line):
                tag = ref_tag_from_plain(m.group(1))
                if tag:
                    requires.add(tag)

    info["refs"] = refs
    info["requires"] = requires
    result.files.append(info)
    if info["f"]:
        for dst in sorted(refs):
            add_rel(result, info["f"], "refs", dst, "auto:refs")
        for dst in sorted(requires):
            add_rel(result, info["f"], "requires", dst, "auto:requires")


def scan_ipynb(result: ScanResult, path: str, rel: str):
    try:
        nb = json.load(open(path, encoding="utf-8"))
    except Exception:
        result.files.append({"rel": rel, "kind": "ipynb", "doc": None, "f": None, "n": None,
                             "doc_mentions": set(), "refs": set(), "requires": set()})
        return
    info = {"rel": rel, "kind": "ipynb", "doc": None, "f": None, "n": None, "doc_mentions": set()}
    refs = set()

    for cell in nb.get("cells", []):
        ctype = cell.get("cell_type", "")
        source = "".join(cell.get("source", []))
        if ctype == "code":
            for m in re.finditer(r"【(N-[0-9]+\.c[0-9]+)】", source):
                tag = m.group(1)
                add_def(result, {
                    "tag": tag, "locator": f"{rel}#{tag.split('.', 1)[1]}",
                    "title": "", "status": "active", "version": "v1.0", "ext": "",
                })
            continue
        if ctype != "markdown":
            continue
        for m in re.finditer(r"【notebook｜(N-[0-9]+)】", source):
            tag = m.group(1)
            add_def(result, {
                "tag": tag, "locator": rel, "title": "",
                "status": "active", "version": "v1.0", "ext": "",
            })
            if info["n"] is None:
                info["n"] = tag
        for m in re.finditer(r"【(N-[0-9]+\.m[0-9]+)】", source):
            tag = m.group(1)
            add_def(result, {
                "tag": tag, "locator": f"{rel}#{tag.split('.', 1)[1]}",
                "title": "", "status": "active", "version": "v1.0", "ext": "",
            })
        for m in re.finditer(r"【([^】]+)】", source):
            body = m.group(1)
            if "｜" in body:
                for side in body.split("｜"):
                    side = side.strip()
                    if ITEM_TAG_RE.match(side) or DOC_TAG_RE.match(side):
                        refs.add(side)
            else:
                tag = ref_tag_from_plain(body)
                if tag:
                    refs.add(tag)
        for m in re.finditer(r"(DOC-[A-Za-z0-9]+)", source):
            info["doc_mentions"].add(m.group(1))

    info["refs"] = refs
    info["requires"] = set()
    result.files.append(info)
    if info["n"]:
        for dst in sorted(refs):
            add_rel(result, info["n"], "refs", dst, "auto:refs")


def infer_corresponds(result: ScanResult):
    """doc ↔ F/N：目录最长前缀唯一可判 + 预备篇显式编号 + notebook 内显式 DOC 提及。"""
    defs = result.defs
    doc_entries = []
    for info in result.files:
        if info.get("kind") == "md" and info.get("doc"):
            rel = info["rel"]
            doc_entries.append((os.path.dirname(rel), os.path.basename(rel), info["doc"]))

    def match_doc(rel_path):
        d = os.path.dirname(rel_path)
        cand_dirs = {d, d.replace("notebooks/", "", 1)}
        best_len, group = -1, []
        for cd in cand_dirs:
            for dd, bn, doc in doc_entries:
                if cd == dd or cd.startswith(dd + "/"):
                    if len(dd) > best_len:
                        best_len, group = len(dd), [(bn, doc)]
                    elif len(dd) == best_len:
                        group.append((bn, doc))
        group = sorted(set(group))
        non_readme = [doc for bn, doc in group if bn != "README.md"]
        if len(non_readme) == 1:
            return non_readme[0]
        if not non_readme and len(group) == 1:
            return group[0][1]
        return None

    def emit(doc, other, note):
        if doc in defs and other in defs:
            add_rel(result, doc, "corresponds", other, note)

    for info in result.files:
        rel = info.get("rel", "")
        f_tag, n_tag = info.get("f"), info.get("n")
        if f_tag:
            doc = match_doc(rel)
            if doc:
                emit(doc, f_tag, "auto:corresponds(dir)")
            m = re.match(r"^F-pre-py([1-6])$", f_tag)
            if m and ("DOC-PY" + m.group(1)) in defs:
                emit("DOC-PY" + m.group(1), f_tag, "auto:corresponds(pre)")
        if n_tag:
            doc = match_doc(rel)
            if doc:
                emit(doc, n_tag, "auto:corresponds(dir)")
            pre_doc = None
            m = re.match(r"^N-([0-9]+)$", n_tag)
            if m:
                num = int(m.group(1))
                if 18 <= num <= 23 and ("DOC-PY" + str(num - 17)) in defs:
                    pre_doc = "DOC-PY" + str(num - 17)
                    emit(pre_doc, n_tag, "auto:corresponds(pre)")
            for doc_hit in sorted(info.get("doc_mentions", set())):
                if doc_hit == doc or doc_hit == pre_doc:
                    continue  # 与目录推断/预备篇规则同一对端点时只保留一条，保证幂等
                emit(doc_hit, n_tag, "auto:corresponds(mention)")


def infer_requires_outline(result: ScanResult, content: str):
    """大纲 §3「既有位置 → 预备条目」映射；仅当两端都已登记才生成。"""
    path = os.path.join(content, "00-预备", "00-大纲.md")
    if not os.path.exists(path):
        return
    defs = result.defs
    lines = open(path, encoding="utf-8").read().split("\n")
    intable = False
    for line in lines:
        if re.match(r"^##\s*3\b", line):
            intable = True
            continue
        if intable and re.match(r"^##\s", line):
            intable = False
        if not intable or not line.strip().startswith("|"):
            continue
        cols = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cols) < 3:
            continue
        first = cols[0]
        if first in ("既有位置",) or set(first) <= set("-: "):
            continue
        pys = sorted(set(re.findall(r"PY([1-6])", cols[2])))
        if not pys:
            continue
        srcs = []
        m = re.match(r"^第\s*([1-6])\s*章", first)
        if m:
            srcs = ["DOC-" + m.group(1)]
        elif re.search(r"0?2\s*[–\-]\s*0?8", first):
            srcs = sorted(t for t in defs
                          if re.match(r"^DOC-(PT|SFT|VH|LORA|RL|MCTS|V11|SRC[0-9])$", t))
        else:
            continue
        for s in srcs:
            for n in pys:
                dst = "DOC-PY" + n
                if s in defs and dst in defs:
                    add_rel(result, s, "requires", dst, "auto:requires(outline)")


def infer_adapted_from(result: ScanResult, external_rows):
    defs = result.defs
    ext_ids = {r[0] for r in external_rows if r and r[0]}
    for tag, rec in defs.items():
        ext = rec.get("ext", "")
        if ext and ext in ext_ids:
            add_rel(result, tag, "adapted_from", ext, "auto:adapted_from")


def scan(content: str) -> ScanResult:
    result = ScanResult()
    for path in iter_source_files(content):
        rel = os.path.relpath(path, content)
        scan_file(result, path, rel)
    return result


# ---------------------------------------------------------------------------
# TSV 读写
# ---------------------------------------------------------------------------

def read_tsv(path, header):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, encoding="utf-8") as fh:
        first = fh.readline()
        if not first:
            return rows
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            cols = line.split("\t")
            if len(cols) < len(header):
                cols += [""] * (len(header) - len(cols))
            rows.append(cols[:len(header)])
    return rows


def write_tsv(path, header, rows):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write("\t".join(header) + "\n")
        for row in rows:
            fh.write("\t".join(sanitize(c) for c in row) + "\n")
    os.replace(tmp, path)


def read_external(tags_dir):
    return read_tsv(os.path.join(tags_dir, "external.tsv"), EXTERNAL_HEADER)


# ---------------------------------------------------------------------------
# extract
# ---------------------------------------------------------------------------

def build_registry_rows(result: ScanResult, previous):
    rows = []
    for tag, rec in result.defs.items():
        status, version, ext = "active", "v1.0", ""
        if tag in previous:
            prow = previous[tag]
            status = prow[5] or status
            version = prow[6] or version
            ext = prow[7] or ext
        rows.append([
            tag, tag_type(tag), rec.get("doccode") or tag_doccode(tag),
            rec.get("locator", ""), rec.get("title", ""), status, version, ext,
        ])
    rows.sort(key=lambda r: natural_key(r[0]))
    return rows


def build_relation_rows(result: ScanResult, previous_relations):
    auto = {(s, rel, d): note for (s, rel, d, note) in result.relations}
    rows = []
    seen = set()
    # 保留人工边（note 不以 auto: 开头）
    for r in previous_relations:
        src, rel, dst = r[0], r[1], r[2]
        note = r[3] if len(r) > 3 else ""
        since = r[4] if len(r) > 4 else ""
        if not src or not rel or not dst:
            continue
        key = (src, rel, dst)
        if note.startswith("auto:"):
            continue
        if key in seen:
            continue
        seen.add(key)
        rows.append([src, rel, dst, note, since or "v1.0"])
    # 追加自动边
    for (src, rel, dst), note in auto.items():
        key = (src, rel, dst)
        if key in seen:
            continue
        seen.add(key)
        rows.append([src, rel, dst, note, "v1.0"])
    rows.sort(key=lambda r: natural_key(r[0]) + natural_key(r[1]) + natural_key(r[2]))
    return rows


def cmd_extract(root: str) -> int:
    root_abs, content, tags_dir = resolve_root(root)
    rel_content = os.path.relpath(content, root_abs)
    os.makedirs(tags_dir, exist_ok=True)

    result = scan(content)
    # locator 统一为相对 root_abs（含 v1.0/ 前缀，与规范 §3 登记表一致）
    if rel_content != ".":
        for rec in result.defs.values():
            loc = rec.get("locator", "")
            if loc and not loc.startswith(rel_content + "/"):
                rec["locator"] = rel_content + "/" + loc
        for info in result.files:
            info["rel"] = os.path.join(rel_content, info["rel"]) if rel_content != "." else info["rel"]

    registry_path = os.path.join(tags_dir, "registry.tsv")
    relations_path = os.path.join(tags_dir, "relations.tsv")
    previous = {}
    for r in read_tsv(registry_path, REGISTRY_HEADER):
        previous[r[0]] = r
    prev_relations = read_tsv(relations_path, RELATIONS_HEADER)

    external_rows = read_external(tags_dir)
    # ext 只存于 registry：先用旧 registry 的 ext 列回填扫描结果，供 adapted_from 生成
    for tag, rec in result.defs.items():
        prow = previous.get(tag)
        if prow and prow[7]:
            rec["ext"] = prow[7]
    infer_corresponds(result)
    infer_requires_outline(result, content)
    infer_adapted_from(result, external_rows)

    registry_rows = build_registry_rows(result, previous)
    relation_rows = build_relation_rows(result, prev_relations)

    write_tsv(registry_path, REGISTRY_HEADER, registry_rows)
    write_tsv(relations_path, RELATIONS_HEADER, relation_rows)

    by_rel = {}
    for r in relation_rows:
        by_rel[r[1]] = by_rel.get(r[1], 0) + 1
    print("extract: root=%s content=%s" % (root_abs, rel_content))
    print("registry: %d tags -> %s" % (len(registry_rows), os.path.relpath(registry_path, root_abs)))
    print("relations: %d edges -> %s" % (len(relation_rows), os.path.relpath(relations_path, root_abs)))
    if by_rel:
        print("  " + ", ".join("%s=%d" % kv for kv in sorted(by_rel.items())))
    if result.duplicates:
        print("duplicates(%d):" % len(result.duplicates))
        for tag, loc in result.duplicates[:20]:
            print("  [dup] %s at %s" % (tag, loc))
        if len(result.duplicates) > 20:
            print("  ... (%d more)" % (len(result.duplicates) - 20))
    if result.md_without_doc:
        print("md without DOC tag(%d):" % len(result.md_without_doc))
        for rel in result.md_without_doc[:20]:
            print("  [no-doc] %s" % rel)
    return 0


# ---------------------------------------------------------------------------
# check
# ---------------------------------------------------------------------------

def build_graph(registry_rows, relation_rows):
    nodes = {r[0]: {"id": r[0], "type": r[1], "doccode": r[2], "locator": r[3],
                   "title": r[4], "status": r[5], "version": r[6], "ext": r[7]}
             for r in registry_rows}
    edges = [{"src": r[0], "relation": r[1], "dst": r[2],
              "note": r[3] if len(r) > 3 else "", "since": r[4] if len(r) > 4 else ""}
             for r in relation_rows]
    return nodes, edges


def find_requires_cycle(edges):
    graph = {}
    for e in edges:
        if e["relation"] == "requires":
            graph.setdefault(e["src"], []).append(e["dst"])
    WHITE, GREY, BLACK = 0, 1, 2
    color = {}
    stack = []
    cycle = []

    def dfs(u):
        color[u] = GREY
        stack.append(u)
        for v in graph.get(u, []):
            if color.get(v, WHITE) == GREY:
                idx = stack.index(v)
                return stack[idx:] + [v]
            if color.get(v, WHITE) == WHITE:
                got = dfs(v)
                if got:
                    return got
        stack.pop()
        color[u] = BLACK
        return None

    for u in list(graph):
        if color.get(u, WHITE) == WHITE:
            got = dfs(u)
            if got:
                return got
    return None


def run_check(root: str, quiet=False):
    root_abs, content, tags_dir = resolve_root(root)
    registry_path = os.path.join(tags_dir, "registry.tsv")
    relations_path = os.path.join(tags_dir, "relations.tsv")

    fresh = scan(content)
    reg_missing = not os.path.exists(registry_path)
    if reg_missing:
        registry_rows = build_registry_rows(fresh, {})
    else:
        registry_rows = read_tsv(registry_path, REGISTRY_HEADER)
    rel_missing = not os.path.exists(relations_path)
    if rel_missing:
        external_rows = read_external(tags_dir)
        if not reg_missing:
            prev = {r[0]: r for r in registry_rows}
            for tag, rec in fresh.defs.items():
                prow = prev.get(tag)
                if prow and prow[7]:
                    rec["ext"] = prow[7]
        infer_corresponds(fresh)
        infer_requires_outline(fresh, content)
        infer_adapted_from(fresh, external_rows)
        relation_rows = build_relation_rows(fresh, [])
    else:
        relation_rows = read_tsv(relations_path, RELATIONS_HEADER)

    external_rows = read_external(tags_dir)
    ext_ids = {r[0] for r in external_rows if r and r[0]}

    errors, warnings = [], []
    if reg_missing:
        warnings.append("registry.tsv 缺失（%s），已按正文即时抽取比对" % os.path.relpath(registry_path, root_abs))
    if rel_missing:
        warnings.append("relations.tsv 缺失（%s），已按正文即时抽取比对" % os.path.relpath(relations_path, root_abs))

    nodes, edges = build_graph(registry_rows, relation_rows)
    registry_tags = set(nodes)

    # 规则 1：关系端点必须存在
    known = registry_tags | ext_ids
    for e in edges:
        for end, role in ((e["src"], "src"), (e["dst"], "dst")):
            if end not in known:
                errors.append("rule1 悬空端点: %s --%s--> %s（%s 不在 registry/external）"
                              % (e["src"], e["relation"], e["dst"], role))

    # 规则 2：requires 子图无环
    cyc = find_requires_cycle(edges)
    if cyc:
        errors.append("rule2 requires 成环: " + " -> ".join(cyc))

    # 规则 3：registry 与正文扫描一致
    body_tags = set(fresh.defs)
    only_file = sorted(registry_tags - body_tags)
    only_body = sorted(body_tags - registry_tags)
    for tag in only_file:
        errors.append("rule3 registry 有但正文未定义: %s" % tag)
    for tag in only_body:
        errors.append("rule3 正文定义但 registry 缺失: %s" % tag)

    # 规则 4：corresponds 仅 doc条目 ↔ F/N
    for e in edges:
        if e["relation"] != "corresponds":
            continue
        a, b = e["src"], e["dst"]
        a_doc = bool(ITEM_TAG_RE.match(a) or DOC_TAG_RE.match(a))
        b_doc = bool(ITEM_TAG_RE.match(b) or DOC_TAG_RE.match(b))
        a_fn = bool(F_TAG_RE.match(a) or N_BASE_RE.match(a))
        b_fn = bool(F_TAG_RE.match(b) or N_BASE_RE.match(b))
        if not ((a_doc and b_fn) or (b_doc and a_fn)):
            errors.append("rule4 corresponds 两端类型不合法: %s <-> %s（须为 文档条目 ↔ F-/N-）" % (a, b))

    # 规则 5：孤立节点仅告警
    degree = {t: 0 for t in registry_tags}
    for e in edges:
        if e["src"] in degree:
            degree[e["src"]] += 1
        if e["dst"] in degree:
            degree[e["dst"]] += 1
    isolated = sorted(t for t, d in degree.items() if d == 0)
    for tag in isolated:
        warnings.append("rule5 孤立节点: %s" % tag)

    # 额外：md 缺 DOC 头（告警）
    for rel in fresh.md_without_doc:
        warnings.append("md 缺【文档｜DOC-*】头: %s" % rel)

    if not quiet:
        print("check: root=%s" % root_abs)
        print("registry: %d tags; relations: %d edges" % (len(registry_rows), len(relation_rows)))
        for msg in errors:
            print("[ERROR] " + msg)
        for msg in warnings:
            print("[WARN] " + msg)
        print("errors: %d; warnings: %d" % (len(errors), len(warnings)))
    return len(errors), len(warnings), errors, warnings


def cmd_check(root: str) -> int:
    errors, warnings, _, _ = run_check(root)
    return 1 if errors else 0


# ---------------------------------------------------------------------------
# render
# ---------------------------------------------------------------------------

def mermaid_id(tag: str) -> str:
    return "n" + re.sub(r"[^0-9A-Za-z]", "_", tag)


def mermaid_label(text: str) -> str:
    return text.replace('"', "'").replace("[", "（").replace("]", "）").replace("\n", " ")


def emit_mermaid(title: str, nodes, edges, direction="TD"):
    lines = ["flowchart " + direction]
    if not nodes:
        lines.append('  empty["（暂无数据）"]')
        return lines
    for tag in nodes:
        lines.append('  %s["%s"]' % (mermaid_id(tag), mermaid_label(tag)))
    for e in edges:
        lines.append("  %s -->|%s| %s" % (mermaid_id(e["src"]), e["relation"], mermaid_id(e["dst"])))
    return lines


def overview_nodes_edges(nodes, edges):
    keep = {t for t in nodes if DOC_TAG_RE.match(t)
            or (F_TAG_RE.match(t) and "." not in t)
            or N_BASE_RE.match(t)}
    oedges = [e for e in edges
              if e["src"] in keep and e["dst"] in keep
              and e["relation"] in ("requires", "corresponds", "adapted_from",
                                    "version_of", "supersedes", "evolves_to", "refs")]
    onodes = sorted(keep)
    return onodes, oedges


def format_table(rows):
    out = ["| Tag | 类型 | 位置 | 标题 | 状态 | 版本 |",
           "| --- | --- | --- | --- | --- | --- |"]
    for tag, typ, locator, title, status, version in rows:
        t = sanitize(title).replace("|", "\\|")
        l = sanitize(locator).replace("|", "\\|")
        out.append("| `%s` | %s | `%s` | %s | %s | %s |" % (tag, typ, l, t, status, version))
    return out


def cmd_render(root: str) -> int:
    root_abs, content, tags_dir = resolve_root(root)
    registry_path = os.path.join(tags_dir, "registry.tsv")
    relations_path = os.path.join(tags_dir, "relations.tsv")
    if not os.path.exists(registry_path) or not os.path.exists(relations_path):
        cmd_extract(root)
    registry_rows = read_tsv(registry_path, REGISTRY_HEADER)
    relation_rows = read_tsv(relations_path, RELATIONS_HEADER)
    nodes, edges = build_graph(registry_rows, relation_rows)

    errors, warnings, _, _ = run_check(root, quiet=True)

    # graph.json
    graph = {
        "generated_by": "v1.0/tools/tag_graph.py render",
        "root": root_abs,
        "counts": {
            "tags": len(registry_rows),
            "relations": len(relation_rows),
            "errors": errors,
            "warnings": warnings,
        },
        "nodes": [nodes[t] for t in sorted(nodes, key=natural_key)],
        "edges": sorted(edges, key=lambda e: (natural_key(e["src"]), natural_key(e["relation"]), natural_key(e["dst"]))),
    }
    graph_json_path = os.path.join(tags_dir, "graph.json")
    with open(graph_json_path + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(graph, fh, ensure_ascii=False, indent=1)
        fh.write("\n")
    os.replace(graph_json_path + ".tmp", graph_json_path)

    # 分组总表
    groups = {}
    for r in registry_rows:
        groups.setdefault(r[2] or "（无 doccode）", []).append(r)
    table_lines = []
    for doccode in sorted(groups, key=natural_key):
        table_lines.append("### doccode `%s`（%d）" % (doccode, len(groups[doccode])))
        table_lines.append("")
        rows = sorted(groups[doccode], key=lambda r: natural_key(r[0]))
        table_lines.extend(format_table([[r[0], r[1], r[3], r[4], r[5], r[6]] for r in rows]))
        table_lines.append("")

    onodes, oedges = overview_nodes_edges(nodes, edges)
    overview_lines = emit_mermaid("全局概览", onodes, oedges)
    req_edges = [e for e in edges if e["relation"] == "requires"]
    req_nodes = sorted({e["src"] for e in req_edges} | {e["dst"] for e in req_edges}, key=natural_key)
    requires_lines = emit_mermaid("requires", req_nodes, req_edges)
    cor_edges = [e for e in edges if e["relation"] == "corresponds"]
    cor_nodes = sorted({e["src"] for e in cor_edges} | {e["dst"] for e in cor_edges}, key=natural_key)
    cor_lines = emit_mermaid("corresponds", cor_nodes, cor_edges)

    # graph.mmd = 全局概览图
    graph_mmd_path = os.path.join(tags_dir, "graph.mmd")
    with open(graph_mmd_path + ".tmp", "w", encoding="utf-8") as fh:
        fh.write("\n".join(overview_lines) + "\n")
    os.replace(graph_mmd_path + ".tmp", graph_mmd_path)

    by_rel = {}
    for r in relation_rows:
        by_rel[r[1]] = by_rel.get(r[1], 0) + 1
    index_lines = [
        "# Tag 索引（自动生成）",
        "",
        "> 本文件由 `v1.0/tools/tag_graph.py render` 生成，**请勿手改**；数据源为 `tags/registry.tsv` 与 `tags/relations.tsv`。",
        "",
        "## 统计",
        "",
        "- registry: **%d** tags" % len(registry_rows),
        "- relations: **%d** edges（%s）" % (
            len(relation_rows),
            ", ".join("%s=%d" % kv for kv in sorted(by_rel.items())) or "无",
        ),
        "- check: errors=%d, warnings=%d" % (errors, warnings),
        "",
        "## 总表（按 doccode 分组）",
        "",
    ]
    index_lines.extend(table_lines)
    index_lines.extend([
        "## Mermaid 图",
        "",
        "### 全局概览（文档 / 代码 / notebook 顶层）",
        "",
        "```mermaid",
        "\n".join(overview_lines),
        "```",
        "",
        "### requires 依赖图（DAG）",
        "",
        "```mermaid",
        "\n".join(requires_lines),
        "```",
        "",
        "### doc ↔ F ↔ N 对应图",
        "",
        "```mermaid",
        "\n".join(cor_lines),
        "```",
        "",
    ])
    index_path = os.path.join(tags_dir, "TAG-INDEX.md")
    with open(index_path + ".tmp", "w", encoding="utf-8") as fh:
        fh.write("\n".join(index_lines))
    os.replace(index_path + ".tmp", index_path)

    print("render: %d tags, %d edges" % (len(registry_rows), len(relation_rows)))
    print("  %s" % os.path.relpath(index_path, root_abs))
    print("  %s" % os.path.relpath(graph_json_path, root_abs))
    print("  %s" % os.path.relpath(graph_mmd_path, root_abs))
    print("  check: errors=%d, warnings=%d" % (errors, warnings))
    return 0


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

USAGE = "usage: tag_graph.py <extract|check|render> <root>"


def main(argv):
    args = list(argv)
    if len(args) != 2 or args[0] not in ("extract", "check", "render"):
        print(USAGE, file=sys.stderr)
        return 2
    cmd, root = args
    if cmd == "extract":
        return cmd_extract(root)
    if cmd == "check":
        return cmd_check(root)
    return cmd_render(root)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
