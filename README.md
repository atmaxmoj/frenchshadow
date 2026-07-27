# Shadow Reader · 发音跟读预研

本地运行的英语发音评估后端 + 网页前端。读一句，停一句，AI 逐词给出发音反馈。

## 依赖

- macOS (已适配 M1/M2/M3) 或 Linux
- Python 3.11+
- espeak-ng (`brew install espeak-ng` on macOS) —— 用于 wav2vec2 音素化后端
- edge-tts (`pip install edge-tts`) —— 用于参考音频；离线时自动回退到 macOS `say`
- ffmpeg —— 用于音频格式转换

## 安装

```bash
cd /Users/wangsijie/Develop/projects/shadow-reader
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

## 启动服务

```bash
./run.sh          # 默认端口 8767
./run.sh 8768     # 自定义端口
```

打开 http://localhost:8767 即可使用。

第一次启动会从 Hugging Face 下载 `facebook/wav2vec2-lv-60-espeak-cv-ft`（约 1.2 GB）。

## 用法

1. 页面加载后显示例句 *The quick brown fox jumps over the lazy dog.*
2. 点击「开始朗读」，允许浏览器使用麦克风。
3. 读出句子后点击「停止」。
4. 等待分析完成：
   - 绿色边框 = 发音良好
   - 黄色边框 = 有改进空间
   - 红色边框 = 明显错误
5. 点击「播放并高亮」回放录音，当前单词会像 KTV 一样被高亮。
6. 下方结果区显示每个单词的 IPA 对比和具体嘴型/舌位/下巴调整建议。

## 项目结构

```
shadow-reader/
├── api/main.py          FastAPI 服务
├── api/static/          网页前端
├── src/analyzer.py      参考 vs 学习者 IPA 对齐与错误分类
├── src/articulatory.py  发音器官调整建议
├── src/models.py        wav2vec2 推理
├── src/audio.py         麦克风录音与文件加载
├── tests/               单元测试
└── demo/                CLI 录音演示
```

## 测试

```bash
./.venv/bin/python -m pytest tests/ -v
```

## 设计说明

- 仅使用 Apache-2.0 许可的 `facebook/wav2vec2-lv-60-espeak-cv-ft` 模型。
- 未使用无 license 的 `slplab` 模型；错误定位依赖参考文本与学习者 IPA 的对齐。
- 针对中文母语者常见英语错误做了专门分类（th/s、v/w、r/l 等）。
- 发音建议来自 `panphon` 特征对比 + 人工整理的中国学习者常见错误 tip 库。
- 嘴形图优先使用 Richard Wright & Dan McCloy 的 [phonetics-teaching-assets](https://github.com/drammock/phonetics-teaching-assets)（CC0），缺失的音用程序生成的示意图兜底。

## 后续方向

- 引入强制对齐（forced alignment）得到更精确的单词时间戳。
- 加入重音、语调、流利度维度。
- 多语言支持（法语、西班牙语等）。
