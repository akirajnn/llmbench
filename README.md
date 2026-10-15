# llmbench — Ollama 本地模型評測框架

針對本機 Ollama 跑的多個 LLM，用一組資料驅動（data-driven）的任務自動評分比較，不需要人工看答案。

## 環境需求

- 已安裝 [Ollama](https://ollama.com/) 並至少 pull 好要測的模型
- Python 3.9+

## 安裝

```bash
cd /Users/hsunmaocheng/prg/llmtest
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 快速開始

```bash
source .venv/bin/activate

# 確認要測的模型已經在 Ollama 裡
ollama list
ollama pull <model>   # 若還沒裝

# 用 config.yaml 設定好的模型清單跑評測
python -m llmbench run --config config.yaml

# 或臨時指定模型/任務，不用改檔案
python -m llmbench run --models qwen2.5:7b,ornith:latest --tasks tasks/general.yaml

# 關閉/開啟 thinking 模式（僅對支援的模型有效，如 Qwen3 系列）
python -m llmbench run --config config.yaml --think false

# 每題重複抽樣 10 次，算 pass@10 與信賴區間（見下方「統計嚴謹度」）
python -m llmbench run --config config.yaml --repeat 10
```

跑完會在終端機印出各模型的比較表（通過率、平均/中位數延遲、平均 tokens/sec、最慢的一題），並把詳細結果寫入 `results/<時間戳記>/`：

- `summary.md`：彙總表 + 每題通過/失敗明細
- `raw.jsonl`：每一筆 (model, task) 的完整原始回應、延遲、token 數，方便之後回溯或重新分析

## 專案結構

```
llmtest/
  config.yaml          # 預設的模型清單 + 任務檔路徑 + 執行選項
  tw_hospital_admin_benchmark_v1.yaml  # 醫院行政辦公能力題組原始檔（評分準則格式）
  tasks/
    general.yaml       # 通用任務集（數學／常識問答／推理／格式）
    tw_hospital_admin.yaml  # 由上面的題組用 import-rubric 產生，需評審模型
    med_exam_115.yaml  # 115 年第一次醫師(一) 醫學(一)(二)，197 題（由 data/exams 匯入）
  testpdf/             # 考選部公告的試題／答案／更正答案 PDF
  data/exams/          # moex-to-csv 從 testpdf 轉出的 CSV（人工檢查用）
    mmlu_style.yaml    # MMLU 風格選擇題
    gsm8k_style.yaml   # GSM8K 風格數學應用題（#### 分隔符格式）
  llmbench/
    providers.py       # 模型呼叫層：ollama / openai（相容端點）/ anthropic，ModelSpec 解析
    runner.py          # 透過 providers 跑每個 model x task（支援重複抽樣）
    grading.py         # 評分規則：exact_match / contains / regex / numeric_tolerance /
                        # multiple_choice / gsm8k / checks
    stats.py           # Wilson score interval、pass@k 無偏估計式、McNemar 配對檢定
    importers.py       # 把 TMMLU+ / 選擇題 CSV（如國考題）/ 評分準則題組轉成任務 YAML
    moex.py            # 把考選部試題／答案 PDF 轉成 CSV
    judge.py           # LLM 評審：依 rubric 替開放式回答打分
    report.py          # 彙整分數、相對 baseline 比例表、輸出 console 表格與 summary.md
    cli.py             # `python -m llmbench run ...` 入口
  tests/
    test_grading.py    # grading.py 的單元測試
    test_stats.py      # stats.py 的單元測試
    test_providers_report.py  # ModelSpec 解析、API key 檢查、選擇題中文前綴、相對 baseline 報表
    test_importers_paired.py  # 匯入／分層抽樣、多答案評分、McNemar 配對檢定
    test_rubric_judge.py      # 評審輸出解析、重大錯誤、rubric 匯入、加權總分
    test_moex.py              # PDF 版面解析、更正答案、真實考卷回歸測試
  results/             # 每次執行的輸出（已加入 .gitignore，不會進版控）
```

## 新增/調整要測的模型

編輯 `config.yaml` 的 `models:` 清單，或執行時用 `--models a,b,c` 覆蓋。字串項目一律視為 Ollama 模型；要加雲端模型請見下一節。

## 與雲端模型比較

本地模型的絕對分數本身沒有參照點——「70%」到底算好還是差，取決於題目難度。做法是把雲端模型放進**同一套題目、同一個 prompt、同一套評分**當錨點（baseline），報表就能直接說出「本地模型達到雲端錨點的幾成」。

### 1. 設定模型

`config.yaml` 的 `models:` 除了字串，也可以放 dict：

```yaml
models:
  - qwen2.5:7b                 # 字串 = 本機 Ollama
  - name: gpt-anchor           # 報表顯示名稱
    provider: openai           # ollama | openai | anthropic
    model: gpt-4o-mini         # 請填實際要比較的模型版本
    api_key_env: OPENAI_API_KEY  # 省略時 openai 讀 OPENAI_API_KEY、anthropic 讀 ANTHROPIC_API_KEY
    # base_url: ...            # 省略時用官方端點
    # max_tokens: 2048
  - name: claude-anchor
    provider: anthropic
    model: claude-sonnet-4-5
  - name: gemini-anchor
    provider: openai           # Gemini 走 OpenAI 相容端點
    base_url: https://generativelanguage.googleapis.com/v1beta/openai
    model: gemini-2.5-flash
    api_key_env: GEMINI_API_KEY

baseline: claude-anchor

options:
  temperature: 0               # 跨模型比較建議固定，避免各家預設值不同
```

- `openai` provider 可接任何 OpenAI 相容的 `/chat/completions` 端點（OpenAI、Gemini、Azure OpenAI、vLLM 等）。
- 雲端呼叫只用標準函式庫 `urllib`，不需額外安裝套件；只跑雲端模型時連 `ollama` 套件都可以不裝。
- API key 一律從環境變數讀取，不要寫進 `config.yaml`。找不到時該模型每一題都會記為錯誤，錯誤訊息會指出要設定哪個環境變數。

**為什麼不直接拿 ChatGPT / Copilot 網頁版比？** 網頁版是「產品」，帶有隱藏系統提示、聯網搜尋等工具，版本也會默默更新，結果無法重現。比較時一律用 API 呼叫底層模型，並在 config 記下確切的模型版本。

### 2. 指定 baseline

```bash
# 用 config.yaml 的 baseline 欄位
python -m llmbench run --config config.yaml

# 臨時改用別的錨點
python -m llmbench run --config config.yaml --baseline gpt-anchor
```

baseline 必須是模型清單中某個模型的 `name`（字串項目的 name 就是模型名稱本身），否則會直接報錯結束。

### 3. 解讀報表

設定 baseline 後，終端機與 `summary.md`（在「各題明細」之前）會多一張各分類的比較表：

| model | math | mmlu | ALL |
|---|---|---|---|
| qwen2.5:7b | 70% → 82% | 60% → 75% | 65% → 79% |
| claude-anchor | 85% (基準) | 80% (基準) | 82% (基準) |

- `70% → 82%`：該模型在此分類通過率 70%，約達 baseline 的 82%（70 / 85）。
- `(基準)`：baseline 本身。
- `—`：baseline 在該分類通過率為 0%，無法計算比例。
- 沒有 `category` 的題目歸在 `uncategorized`；`ALL` 是全部題目的總計。
- 比例可能超過 100%，代表該分類本地模型表現比錨點好。題數少時這個比例波動很大，請搭配上方的 Wilson 信賴區間一起看。
- 雲端模型的 `avg_tok/s` 是用「輸出 token 數 / 總延遲」估算，**包含網路延遲與排隊時間**，不能和本地模型的純生成速度直接比較。

### 4. 配對檢定：差距是真的還是運氣？

比例表只有點估計。有設定 baseline 時，報表還會多一張 McNemar 配對檢定表：

```
model     category        n    勝/負  差距(點)          95% CI  McNemar p
local-7b  醫療專業        300  36/90  -18.0  [-25.0, -11.0]   <0.001 *
local-7b  tmmluplus       300  14/22   -2.7    [-6.6, +1.3]      0.243
```

- **勝/負**：本地模型對但 baseline 錯的題數／本地模型錯但 baseline 對的題數。兩邊都對或都錯的題目不提供差異資訊。
- **差距**：配對比例差（勝 − 負）/ n，單位為百分點，負值代表落後 baseline；95% CI 不包含 0 才算有差異。
- **McNemar p**：精確二項檢定，`*` 代表 p < 0.05。

**為什麼要用配對檢定？** 所有模型答的是同一批題目，題目本身的難易會讓兩個模型的對錯高度相關。把兩邊的通過率當成獨立樣本比較（例如看各自的 Wilson CI 有沒有重疊）會白白浪費這個資訊，嚴重低估顯著性。配對檢定只看答案不一致的題目，同樣題數下檢定力高得多。

`repeat > 1` 時只用每題的第一次抽樣配對；同一題的多次抽樣彼此相關，全部放進去會灌水題數、讓 p 值偏小。

### 每個分類需要幾題？

所需題數取決於兩個模型「答案不一致的題目比例」（p_disc），跑完第一輪後可以從勝/負欄算出：p_disc = (勝 + 負) / n。80% 檢定力、雙尾 α = 0.05 時：

```
n ≈ (1.96·√p_disc + 0.84·√(p_disc − δ²))² / δ²      δ = 想偵測的差距（例如 0.10）
```

舉例來說，想偵測 10 個百分點的差距：p_disc = 0.20 時約需 155 題，p_disc = 0.30 時約需 235 題。若用獨立樣本的公式算（80% vs 70%）會得到約 290 題，比配對設計保守。建議先每類跑 100–150 題，用實際的 p_disc 推算還要補多少題。

### 資料安全：送往雲端的題目必須去識別化

設定雲端模型後，**每一題的 prompt 都會原文送到第三方 API**。題目只能使用公開題或去識別化／合成資料，**不可包含真實病歷、個資或任何未經授權外流的資料**。若題庫中有敏感題目，請另建一個只給本地模型跑的任務檔，不要放進含雲端模型的評測。

## 新增題目

在 `tasks/general.yaml` 加一筆，或另外建立新的任務檔（如 `tasks/coding.yaml`），再把路徑加進 `config.yaml` 的 `tasks:` 清單。每筆任務格式：

```yaml
- id: 任務代號（唯一）
  category: 分類（自由命名，只影響報表分組，非必填）
  prompt: "要送給模型的 prompt"
  grading:
    type: exact_match | contains | regex | numeric_tolerance | multiple_choice | gsm8k | checks
    expected: ...        # 依 type 而定的預期值
    tolerance: 0         # numeric_tolerance / gsm8k 用得到，容許誤差
    case_sensitive: false # exact_match / contains 用得到
    choices: "ABCD"      # multiple_choice 用得到，合法選項字母
    delimiter: "####"    # gsm8k 用得到，答案分隔符（預設就是 ####）
    checks: [...]        # checks 用得到，見下方說明
```

評分規則：

| type | 說明 |
|---|---|
| `exact_match` | 回應（去頭尾空白）需與 `expected` 完全相同 |
| `contains` | 回應中需包含 `expected` 這段文字 |
| `regex` | 回應需符合 `expected` 這個正則表達式 |
| `numeric_tolerance` | 從回應抽出「最後一個」數字（heuristic），與 `expected` 的誤差需在 `tolerance` 以內 |
| `multiple_choice` | **MMLU 風格**：抓回應中第一個獨立出現的選項字母（如 B），須與 `expected` 相符 |
| `gsm8k` | **GSM8K 風格**：要求回應用 `#### <數字>` 標示最終答案，只看分隔符之後的數字，格式不合規直接判失敗 |
| `checks` | **IFEval 風格**：`checks` 給一組子條件（格式同上述任一 type），全過才算 strict pass，另外回報部分通過比例（loose score）|

所有文字比對前都會先做 Unicode NFKC 正規化（例如 `H₂O` 的下標數字會轉成一般的 `H2O` 再比對），避免因為字元編碼不同造成假陰性。

### 為什麼要分這麼多種 grading type？

`numeric_tolerance` 是「用 heuristic 猜答案在哪」，遇到模型展示運算過程時容易抓錯數字（例如回應「12 × 7 = 84」，到底哪個是答案？）。`multiple_choice` 和 `gsm8k` 是直接借用業界標準評測（MMLU、GSM8K）的做法：**從源頭設計成沒有歧義的回答格式**（選字母、用分隔符），評分邏輯就能是無歧義的精確比對，不用再靠猜。新題目建議優先用這兩種格式，`numeric_tolerance` 留給不方便要求特定格式的情境當 fallback。

## 匯入大型題庫

手寫 YAML 只適合少量題目。數百題以上的選擇題可以用匯入指令，從 CSV 轉成任務檔，並依學科**按比例分層抽樣**（最大餘數法），讓抽出的子集在學科組成上與母體一致。抽樣用固定的 `--seed`，同樣參數重跑一定得到同一份題組。

匯入時被剔除的題目和原因會寫到 `<out>.skipped.txt`。

### TMMLU+（繁中綜合知識）

```bash
# 第一次加 --download，從 Hugging Face 下載 v1.1 的 test split（66 個學科）到 --src
python -m llmbench import-tmmluplus --src data/tmmluplus --out tasks/tmmluplus.yaml --n 300 --download

# 依官方四大類（stem / social / humanities / other）拆分類，報表可分學科群比較
python -m llmbench import-tmmluplus --src data/tmmluplus --out tasks/tmmluplus.yaml --n 300 --split-by-group
```

- 版本固定為 [TMMLU+ v1.1](https://huggingface.co/datasets/ikala/tmmluplus)（MIT License），v1.1 已修正 197 題答案、移除 539 題有問題的題目。
- 學科分類取自 iKala 官方評測工具 [ievals](https://github.com/iKala/ievals) 的 `subject.tsv`。注意官方把牙醫、藥理、中醫等醫藥學科歸在 `other`，藥劑學和基礎醫學則歸在 `stem`。
- 全集約 2 萬題，抽 300 題平均每個學科只有 4–5 題，**只能解讀整體或四大類，不要拆到單一學科**。
- TMMLU+ 自 2023 年公開，很可能已在模型的訓練資料中，分數可能偏高。

### 考選部 PDF → CSV

考選部公告的是 PDF，要先轉成 CSV。檔名維持考選部下載時的原樣即可：試題 `<考試代碼>_<代號>.pdf`，答案 `<考試代碼>_ANS<代號>.pdf`，更正答案 `<考試代碼>_MOD<代號>.pdf`。有更正答案時會優先使用。

```bash
pip install pymupdf   # 只有這個指令需要
python -m llmbench moex-to-csv --dir testpdf --out data/exams
```

PDF 沒有結構化的題目資訊，只能依版面規則切分，所以採用「**先轉 CSV 人工檢查，再用 import-exam 匯入**」兩段式。轉換時會自動處理以下情況：

- **兩種版面**：「1. / A.」文字標記（如醫師考試），以及用字型私有區字元畫圈選符號、題號獨立一行（如公務人員初等考試）。
- **更正答案**：備註「第 66 題答 A 或 D 者均給分」會轉成 `A,D`，「第 95 題一律給分」會轉成 `一律給分`（`import-exam` 會剔除）。備註無法自動解析時會直接報錯，請人工處理。
- **自動標記剔除（skip 欄）**：
  - 含圖片：包括字型缺字以圖片代替的字，例如「屈拇長肌」的「拇」。
  - 含表格框線或底線：克漏字空格、字詞底線在純文字中會消失。
  - 題幹提及圖表。
  - 複選題：目前評分器只支援單選。
  - 選項數異常。
  - 無法辨識的特殊符號。
- **題組**：「請依下文回答第 46 題至第 50 題」的文章會併入每一題的題幹，並標記待人工確認。

轉出後請打開 CSV 檢查：skip 欄有標記的題目若確認沒問題，把 skip 欄清空就會被匯入。

### 選擇題 CSV（如醫師國考題）

```bash
python -m llmbench import-exam --src data/med_113_1.csv,data/med_113_2.csv \
    --out tasks/med_exam.yaml --category 醫療專業 --min-year 112
```

CSV 欄位（UTF-8，可帶 BOM，Excel 另存「CSV UTF-8」即可）：

| 欄位 | 必填 | 說明 |
|---|---|---|
| `question`、`A`–`D` | ✓ | 題幹與選項；有 `E` 欄時會自動變成五選一 |
| `answer` | ✓ | `C`；多個答案都算對時寫 `A,C`、`A、C`、`A或C`；`一律給分`、`*` 或所有選項都對的題目會被剔除 |
| `id` | | 題號，沒有時用「檔名_列號」自動產生；重複時直接報錯 |
| `year` | | 年度，搭配 `--min-year` 只保留較新的考題（民國或西元皆可，與 `--min-year` 一致即可） |
| `subject` | | 科目，搭配 `--n` 時作為分層抽樣的依據 |
| `skip` | | 有填任何內容就剔除，例如標記「看圖」「表格」 |

國考題注意事項：

- 依《著作權法》第 9 條第 1 項第 5 款，依法令舉行之考試試題不受著作權保護。
- 答案請以考選部**更正後**的答案為準。一律給分的題目無法鑑別模型能力，匯入時會自動剔除。
- 看圖題、依賴表格排版的題目請在 `skip` 欄標記。
- 「最新年度」不等於沒有污染：雲端模型的訓練截止日可能已涵蓋去年的考題。報告中請註明考題年度與各模型的 knowledge cutoff。

## 開放式題目：LLM 評審（rubric）

行政判斷、公文撰寫這類題目沒有唯一正確答案，無法用程式比對。`rubric` 評分類型會把題目、受測回答、預期要點與重大錯誤清單交給**評審模型**，由它依各評分面向打分。

### 匯入評分準則題組

```bash
python -m llmbench import-rubric --src tw_hospital_admin_benchmark_v1.yaml --out tasks/tw_hospital_admin.yaml
```

題組格式為 `benchmark`（題組設定）加上 `questions`（題目），詳見 `tw_hospital_admin_benchmark_v1.yaml`。匯入時會把以下設定攤平到每一題：

- `question_prompt_template`：受測模型看到的提示，必須包含 `{question}`。
- `pass_threshold_percent`：及格門檻。
- `evaluation_guidance`：給評審的評分說明。
- `category_weights_percent`：各分類的權重。

分類名稱使用 `category_zh`。

### 設定評審並執行

在 `config.yaml` 設定 `judge:`（格式同 `models` 的項目），或用 `--judge <name>` 指定：

```bash
python -m llmbench run --config config.yaml --tasks tasks/tw_hospital_admin.yaml
```

評分規則：

- 評審必須輸出各面向分數的 JSON；例如本題組是要點涵蓋 0–3 分、權責程序 0–1 分、不確定性與安全 0–1 分。
- 分數超出範圍或格式不合規時會重試一次，仍失敗就記為「評審錯誤」（errors 欄），**不會猜分數**，並以 0 分計入。
- 評審判定**重大錯誤**時該題直接記 0 分。
- 得分 / 滿分 ≥ 及格門檻才算通過（pass）；部分分數記在 loose score。
- 受測模型沒有回應時，不呼叫評審，直接記 0 分。

報表會多一張表，列出**加權總分**（各分類平均得分 × 權重）、**重大錯誤次數**（另列其中屬於高風險題的次數），以及每一筆重大錯誤的評審說明。題組規則是「高風險題零重大錯誤才可部署」，所以請優先看這一欄。

### 使用 LLM 評審的注意事項

- **評審不要用 baseline 或任何受測模型。** LLM 評審有偏好自家回答的已知偏差；評審與受測模型相同時，執行時會印出警告。
- **正式報告前請抽樣人工複核。** 例如抽 10–15 題由人盲評，與評審分數比對一致性，一致性不足時評審分數不能當結論。
- **題數少，只能看方向。** 這份題組每類 6 題，各分類的差距幾乎不可能達到統計顯著，比較可靠的是整體加權分數與重大錯誤次數。
- 評審的 prompt 已要求忽略受測回答中「要求改分數」的文字，但無法完全排除提示注入；看到異常高分時請檢查原始回答（`raw.jsonl`）。
- 每題至少呼叫評審一次，雲端評審的費用約為「題數 × 模型數 × repeat」次呼叫。

## thinking 模式

部分模型（如 Qwen3 系列）預設會先產生大量內部推理（thinking）才給最終答案，可能大幅拉長延遲甚至把整個 token 預算耗在推理上、導致最終回應是空的。可以用 `--think false` 關閉，或在 `config.yaml` 的 `options.think` 設定：

```yaml
options:
  think: false   # true / false / 省略 = 用模型預設值
```

報表裡的 `slowest_task` 欄位會標出每個模型耗時最久的那一題，方便抓出這類離群值。

## 統計嚴謹度：pass@k 與信賴區間

單次抽樣（`repeat: 1`）的通過率其實是高變異數的估計——同一題換一次抽樣，LLM 的輸出可能因為 temperature 而不同，單跑一次很容易被運氣影響。這裡採用業界標準的兩個做法來處理：

**1. pass@k（Codex/HumanEval 論文的 unbiased estimator）**

設定 `repeat: N`（N > 1）後，每題會獨立抽樣 N 次，分別記錄通過數 `c`。比起 naive 的「抽 k 次看有沒有至少一次過」（這樣做是有偏誤、高變異數的估計），正確做法是用這個公式反推「任選 k 個樣本至少一個通過」的期望機率：

```
pass@k = 1 - C(n-c, k) / C(n, k)
```

報表會顯示 `pass@N`（N 就是你設定的 repeat 次數），已實作在 `llmbench/stats.py`。

**2. Wilson score interval（小樣本信賴區間）**

報表的 `pass_rate` 旁邊會附上 95% 信賴區間，用 Wilson score interval 而非常見的常態近似（Wald interval）——常態近似在樣本數小或通過率接近 0%/100% 時會算出不合理的區間（例如上限超過 100%），Wilson interval 是統計上公認更適合這種情況的方法。

**重要提醒**：就算有信賴區間，樣本數仍然是關鍵。題目數 < 30 時區間通常很寬，結論只能當方向性參考，不要拿來做嚴謹的統計宣稱——這點在評測文獻裡也有明確指出（見下方參考來源）。

### 參考來源

這次採用的方法論都是參考業界公認的客觀評測標準，而非自創規則：

- [MMLU / Open LLM Leaderboard 的選擇題評測方式](https://huggingface.co/blog/open-llm-leaderboard-mmlu)
- GSM8K 資料集的 `####` 答案格式慣例（[openai/grade-school-math](https://github.com/openai/grade-school-math)）
- [pass@k 無偏估計式，出自 Codex/HumanEval 論文](https://arxiv.org/abs/2107.03374)（Chen et al., 2021）
- [Wilson score interval（二項比例信賴區間）](https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval)
- [IFEval：用可驗證的程式化條件評分格式遵循度，並分別回報 strict/loose 準確率](https://arxiv.org/abs/2311.07911)
- [lm-evaluation-harness](https://futureagi.com/blog/lm-evaluation-harness-guide/) / [HELM](https://aisecurityandsafety.org/en/compare/helm-vs-lm-evaluation-harness/)：強調固定 prompt 模板、固定 few-shot 範例，確保跑兩次結果可重現

## 跑單元測試

```bash
source .venv/bin/activate
python -m unittest discover -s tests -v
```

## 後續可擴充方向

- 程式碼生成類任務：重用本機其他 CRUD 專案的需求規格當題目，評分改用「能否成功編譯/執行」當指標
- Few-shot 範例：官方 MMLU 評測用固定的 5-shot 範例（prompt 裡先給幾題示範），目前我們的 `mmlu_style.yaml` 是 zero-shot，分數可能比官方排行榜略低／更不穩定；要跟公開排行榜數字對齊的話，可以在 prompt 裡加上固定的範例題
