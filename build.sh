#!/usr/bin/env python3
"""Extract all 473 lessons from source repo into VitePress structure."""
import os
import shutil
import sys
import json

SRC = sys.argv[1] if len(sys.argv) > 1 else "/tmp/ai-eng-src"
DEST = os.path.dirname(os.path.abspath(__file__))

PHASE_TITLES = {
    "00-setup-and-tooling": "环境搭建与工具",
    "01-math-foundations": "数学基础",
    "02-ml-fundamentals": "机器学习基础",
    "03-deep-learning-core": "深度学习核心",
    "04-computer-vision": "计算机视觉",
    "05-nlp-foundations-to-advanced": "自然语言处理",
    "06-speech-and-audio": "语音与音频",
    "07-transformers-deep-dive": "Transformer 深入",
    "08-generative-ai": "生成式 AI",
    "09-reinforcement-learning": "强化学习",
    "10-llms-from-scratch": "从零构建 LLM",
    "11-llm-engineering": "LLM 工程",
    "12-multimodal-ai": "多模态 AI",
    "13-tools-and-protocols": "工具与协议",
    "14-agent-engineering": "Agent 工程",
    "15-autonomous-systems": "自主系统",
    "16-multi-agent-and-swarms": "多 Agent 与集群",
    "17-infrastructure-and-production": "基础设施与生产",
    "18-ethics-safety-alignment": "伦理、安全与对齐",
    "19-capstone-projects": "毕业项目",
}

sidebar = []
total_lessons = 0

phases_dir = os.path.join(SRC, "phases")
for phase_name in sorted(os.listdir(phases_dir)):
    phase_path = os.path.join(phases_dir, phase_name)
    if not os.path.isdir(phase_path):
        continue

    phase_num = phase_name.split("-")[0]
    cn_title = PHASE_TITLES.get(phase_name, phase_name)
    print(f"📘 Phase {phase_num}: {cn_title}")

    dest_phase = os.path.join(DEST, "phases", phase_name)
    os.makedirs(dest_phase, exist_ok=True)

    lessons = []
    for lesson_name in sorted(os.listdir(phase_path)):
        lesson_path = os.path.join(phase_path, lesson_name)
        if not os.path.isdir(lesson_path):
            continue
        doc_file = os.path.join(lesson_path, "docs", "en.md")
        if not os.path.isfile(doc_file):
            continue

        # Copy doc
        shutil.copy2(doc_file, os.path.join(dest_phase, f"{lesson_name}.md"))

        # Extract title
        with open(doc_file, "r") as f:
            first_line = f.readline().strip()
        title = first_line.lstrip("# ").strip()

        lessons.append((lesson_name, title))
        total_lessons += 1

    # Create phase index
    with open(os.path.join(dest_phase, "index.md"), "w") as f:
        f.write(f"# Phase {phase_num}: {cn_title}\n\n")
        f.write(f"本阶段包含 **{len(lessons)} 课时**。\n\n")
        f.write("> 原始课程来源：[AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (MIT License)\n\n---\n\n")
        for lesson_name, title in lessons:
            f.write(f"- [{title}](./{lesson_name})\n")

    # Build sidebar items
    phase_items = [
        {"text": title, "link": f"/phases/{phase_name}/{lesson_name}"}
        for lesson_name, title in lessons
    ]
    sidebar.append({
        "text": f"{phase_num}. {cn_title}",
        "collapsed": True,
        "items": phase_items,
    })

    print(f"   → {len(lessons)} lessons")

# Write sidebar config as JSON for VitePress config to import
with open(os.path.join(DEST, ".vitepress", "sidebar.json"), "w") as f:
    json.dump(sidebar, f, ensure_ascii=False, indent=2)

print(f"\n✅ Done! {total_lessons} lessons across {len(sidebar)} phases")

