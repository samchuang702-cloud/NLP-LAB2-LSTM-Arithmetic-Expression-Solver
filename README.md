# NLP LAB2 — 使用 LSTM 解決算術表達式序列生成問題

## 專案簡介

本專案將算術表達式求解視為**字元層級的序列生成問題**：模型讀入包含數字、運算子與括號的算式，再逐字元生成答案。專案以 PyTorch 建立雙層 LSTM，並透過控制實驗分析模型架構、資料分布與標籤品質的影響。

## 想解決的問題

- 循環神經網路能否從字元序列學會產生算術答案？
- LSTM 相較 vanilla RNN 是否能更有效保留長距離資訊？
- 訓練與測試分布不同時，模型能否泛化？
- 訓練資料含錯誤答案時，生成品質受到多大影響？

## 模型與資料處理

- 字元詞彙表包含數字、運算子、括號、等號、`<pad>` 與 `<eos>`。
- 架構：Embedding → 2-layer LSTM → Linear/ReLU/Linear。
- 使用 teacher forcing、packed sequence 與 gradient clipping。
- 使用 CrossEntropyLoss，忽略 padding token。
- 採 Exact Match 評估：整個答案完全一致才算正確。

## 主要超參數

| 項目 | 設定 |
|---|---:|
| Batch size | 64 |
| Epochs | 18 |
| Embedding dimension | 256 |
| Hidden dimension | 256 |
| Learning rate | 0.001 |
| Gradient clipping | 1 |
| Optimizer | Adam |

## 實驗結果

| 實驗 | Exact Match |
|---|---:|
| 最終 2-layer LSTM（18 epochs） | 84.59% |
| 2-layer vanilla RNN（5 epochs） | 15.72% |
| 2-layer LSTM（5 epochs） | 66.84% |
| 僅以三位數資料訓練、原始資料評估 | 0.06% |
| 含 20% 錯誤標籤的訓練資料 | 60.40% |

LSTM 在相同訓練輪數下比 vanilla RNN 高出 51.12 個百分點。分布偏移實驗顯示模型主要學到資料模式，而非可靠的符號計算規則；錯誤標籤也會明顯降低可靠性。

## 使用技術

Python、PyTorch、LSTM、RNN、Sequence Generation、Teacher Forcing、Packed Sequence、Exact Match、Google Colab。

## 專案亮點

- 完成資料生成、字元編碼、模型訓練與自迴歸生成。
- 以量化實驗比較 RNN 與 LSTM。
- 主動測試資料分布偏移與標籤雜訊。
- 展現 PyTorch、序列模型、實驗控制與錯誤分析能力。

## 執行方式

1. 使用 Google Colab 或 Jupyter Notebook 開啟 `NLP_LAB2_LSTM_Arithmetic_Solver.ipynb`。
2. 依序執行資料生成、模型定義及訓練儲存格。
3. 建議使用 GPU 執行環境。

## 已知限制

- 模型不具備保證正確的符號推理能力。
- 對未見過的數字長度或資料分布敏感。
- Exact Match 無法呈現局部答案接近程度。

## 未來改進

- 與 GRU、Transformer 及神經符號方法比較。
- 使用 curriculum learning 逐步提高難度。
- 加入 constrained decoding 限制輸出格式。
