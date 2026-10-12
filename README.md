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
  tasks/
    general.yaml       # 通用任務集（數學／常識問答／推理／格式）
    mmlu_style.yaml    # MMLU 風格選擇題
    gsm8k_style.yaml   # GSM8K 風格數學應用題（#### 分隔符格式）
  llmbench/
    runner.py          # 呼叫 Ollama API 跑每個 model x task（支援重複抽樣）
    grading.py         # 評分規則：exact_match / contains / regex / numeric_tolerance /
                        # multiple_choice / gsm8k / checks
    stats.py           # Wilson score interval、pass@k 無偏估計式
    report.py          # 彙整分數、輸出 console 表格與 summary.md
    cli.py             # `python -m llmbench run ...` 入口
  tests/
    test_grading.py    # grading.py 的單元測試
    test_stats.py      # stats.py 的單元測試
  results/             # 每次執行的輸出（已加入 .gitignore，不會進版控）
```

## 新增/調整要測的模型

編輯 `config.yaml` 的 `models:` 清單，或執行時用 `--models a,b,c` 覆蓋。

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
- LLM-as-judge：用模型互評主觀題的流暢度/正確性（目前只做純自動化指標）
- Few-shot 範例：官方 MMLU 評測用固定的 5-shot 範例（prompt 裡先給幾題示範），目前我們的 `mmlu_style.yaml` 是 zero-shot，分數可能比官方排行榜略低／更不穩定；要跟公開排行榜數字對齊的話，可以在 prompt 裡加上固定的範例題
