# 宝宝哭声翻译器

宝宝哭了，点一下，手机听 8 秒，告诉你：

- **是不是宝宝在哭**（谷歌开源的 YAMNet 声音模型来判断）
- **哭得急不急**（看音量、音调和哭声占比）
- **可能是什么原因**：饿了、困了、尿布、胀气、想抱抱（结合你记的喂奶、睡醒、换尿布时间来排）
- **怎么哄**：针对最可能的原因给几条具体做法
- **什么情况要直接看医生**

整个识别都在手机浏览器里跑，**录音不会上传到任何服务器**。一个 HTML 文件 + 一个 3.8MB 的模型文件，不需要后端。

## 先说清楚它能做什么、不能做什么

- 判断「是不是宝宝哭声」比较靠谱。
- 判断「为什么哭」只靠声音很难做准。所以这里用的是**时间记录 + 声音特征**一起推测，页面上写的是「可能原因」，不是「翻译结果」。
- 它不能代替医生。发烧、哭声异常、一直哄不好，请直接就医。

## 实测效果

用真实录音测过（模型和页面里的代码完全一样）：

| 测试集 | 内容 | 结果 |
|---|---|---|
| Donate-a-cry 语料 | 457 段家长用手机录的真实宝宝哭声（8kHz，音质一般） | 84% 判断为哭声 |
| ESC-50 | 40 段宝宝哭声 | 95% 判断为哭声 |
| ESC-50 + LibriSpeech | 600 段家里常见声音（狗叫、猫叫、笑声、咳嗽、吸尘器、洗衣机、闹钟等）+ 73 段大人说话 | 误判为哭声 2.2%，大多数是猫叫 |

猫叫和宝宝哭声本来就很像，所以猫叫分数更高时，页面会提示「可能是宝宝，也可能是猫叫」。

## 部署

需要用 **https** 打开才能用麦克风（浏览器的规定）。

**必须通过网址打开。** 直接打开 `index.html` 文件（比如在微信里收到文件后点开）是用不了的：微信的文件预览不运行程序，浏览器也不允许本地文件读取同目录的模型。页面检测到这种情况会给出提示。

**GitHub Pages**：把整个文件夹推到仓库，Settings → Pages 选分支，几分钟后就有网址。

**Cloudflare Pages**：连 GitHub 仓库或直接上传文件夹，不需要构建命令。

**国内用户**：GitHub Pages 在国内访问不稳定，建议用自己备案过的域名放一份（任何静态网站托管都行，比如对象存储开静态网站）。页面已经把 TF.js 放在 `lib/` 里，不依赖国外 CDN；Google 字体加载不到会自动用手机系统字体。

**本地试**：

```bash
python3 -m http.server 8000
# 打开 http://localhost:8000
```

> 在微信、抖音里直接点开链接，内置浏览器可能用不了麦克风。页面会提示用户「在浏览器打开」。

## 文件结构

```
index.html              页面（样式、脚本都在里面）
model/
  yamnet-weights.bin    YAMNet 权重（int8，3.8MB）
  yamnet-weights.json   权重目录
lib/tf.min.js           TensorFlow.js 4.22.0
sample/cry-demo.wav     示例哭声（Donate-a-cry 语料）
icons/                  图标（加到手机桌面用）
manifest.webmanifest    PWA 配置
tools/export_weights.py 从官方 yamnet.h5 生成上面那两个权重文件的脚本
```

## 常改的配置

打开 `index.html`，搜 `CONFIG`：

```js
var CONFIG = {
  recordSeconds: 8,    // 每次听多久
  ...
};
```

## 原理

1. **哭声识别**：用 Google 开源的 [YAMNet](https://github.com/tensorflow/models/tree/master/research/audioset/yamnet)（在 AudioSet 上训练，能认 521 种声音，其中有「宝宝哭声」）。页面里自己实现了 YAMNet 的 log-mel 特征和网络前向计算，用 TensorFlow.js 跑。电脑浏览器上分析一段 8 秒录音约 1–3 秒，手机上视机型而定。
2. **急不急**：算音量、音调（McLeod 音高算法）和哭声占整段录音的比例。阈值是用上面的真实哭声校准的：宝宝哭声音调中位数大约 430–460 Hz，九成在 350–575 Hz。
3. **可能原因**：按月龄的一般喂奶间隔和清醒时长，加上你记的时间，给每个原因打分排序。规则都写在 `index.html` 的 `rankReasons` 里，可以按需要改。

### 权重文件是怎么来的

`tools/export_weights.py` 读取官方的 `yamnet.h5`，把批归一化层合并进卷积层，再按输出通道做 int8 量化（15MB → 3.8MB）。脚本里还带了一份 NumPy 版前向计算，用来核对：未量化时和 TensorFlow 官方实现结果一致，并通过 YAMNet 官方的三项自检（静音、白噪音、正弦波）。量化后单个分数会有小幅波动，但在上面那套测试里整体效果和原版一样：原版 84% / 92% / 误判 2.2%，量化版 84% / 95% / 误判 2.2%。

```bash
pip install h5py numpy
curl -O https://storage.googleapis.com/audioset/yamnet.h5
python3 tools/export_weights.py yamnet.h5 model/
```

## 声明

本系统免费开源，仅用于帮助宝爸宝妈缓解带娃的困扰。未经作者许可，禁止任何形式的商业使用，侵权必究。

分析结果仅供参考，不能代替专业人士的建议。宝宝情况紧急时，请马上就医。

## 协议与致谢

- 本项目代码：非商用许可，参照 [PolyForm Noncommercial 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0)，详见 `LICENSE`。个人、家庭、学习研究可以免费用；商用需要作者授权。
- YAMNet 模型：Google，Apache-2.0
- TensorFlow.js：Google，Apache-2.0
- 示例哭声：[Donate-a-cry corpus](https://github.com/gveres/donateacry-corpus)，ODbL / DbCL

DEVELOPED by @赛博老爸
