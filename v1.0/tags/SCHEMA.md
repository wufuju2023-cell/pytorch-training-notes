# tags/ · Tag 图数据规范

> **【文档｜DOC-TAGS】**（doccode = `TAGS`）｜本目录是 Tag 图的机器可读数据源；schema 与关系词表以《00-风格与编号规范》§9 为准。

数据文件（TSV，UTF-8，首行为表头；由 `tools/tag_graph.py`（F-tools-tag_graph）`extract` 生成 / `check` 校验 / `render` 出图）：

| 文件 | 列 | 说明 |
| --- | --- | --- |
| `registry.tsv` | `tag, type, doccode, locator, title, status, version, ext` | 全部 Tag：`locator` 为相对路径（可带 `#锚点`）；`status` 取 `active/deprecated`；`version` 如 `v1.0`；`ext` 为 `EXT-*`（可空） |
| `relations.tsv` | `src, relation, dst, note, since` | 关系边（有向，src→dst）；`relation` 取自 §9 词表 |
| `external.tsv` | `ext_id, url, license, usage, note` | 外部材料登记；`ext_id` 形如 `EXT-PY5-01` |

关系词表（14 类）：`requires, refs, proves, corresponds, impl, example_of, generalizes, specializes, equiv, version_of, supersedes, evolves_to, adapted_from, seealso`。

校验规则（`tag_graph.py check`）：

1. 所有关系端点必须出现在 registry（或 external）；
2. `requires` 子图无环（DAG）；
3. registry 与正文扫描结果一致（正文中定义的 Tag 均有登记，登记均有定义）；
4. `corresponds` 仅允许 文档条目 ↔ `F-*`/`N-*`；
5. 孤立节点仅告警，不出错。
