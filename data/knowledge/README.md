# 知识库样例资料

本目录中的资料是面向课程演示的公开信息摘要，来源字段用于 RAG 引用展示。内容不包含真实个人隐私，也不代表对具体商品或传承人的认证结论。

## 文件清单

| 文件 | 内容 |
|------|------|
| heritage_paper_cutting.md | 剪纸公开知识摘要（概述） |
| heritage_paper_cutting_detail.md | 剪纸艺术知识详解（历史、流派、纹样寓意、工序） |
| heritage_embroidery.md | 刺绣公开知识摘要（概述） |
| heritage_embroidery_detail.md | 刺绣技艺知识详解（四大名绣、针法、材料、流程） |
| heritage_ceramics.md | 手工制瓷公开知识摘要（概述） |
| heritage_ceramics_detail.md | 手工制瓷知识详解（陶与瓷、工序、釉彩、窑口） |
| live_compliance.md | 直播合规注意事项（监管要求） |
| live_script_skills.md | 非遗直播话术与互动技巧（节奏、互动、合规红线） |

## 如何扩充

1. 把新的 `.md` / `.txt` / `.pdf` 文件放进本目录；
2. md 文件建议参照现有格式书写 front matter（title/category/source/source_url 均为可选）；
3. 在项目根目录执行 `python main.py build` 重建 FAISS 索引（每次修改后都要重新构建）。
