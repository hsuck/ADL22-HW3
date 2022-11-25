# HW3

r11944008 許智凱

## Q1: Model (2%)

### Model (1%)

#### Describe the model architecture and how it works on text summarization.

T5 是一個 Encoder-Decoder 模型，它將所有的 NLP 問題轉換為 text-to-text 的格式。

<img src="C:\Users\kai\AppData\Roaming\Typora\typora-user-images\image-20221122170450129.png" alt="image-20221122170450129" style="zoom: 67%;" />

Encoder-Decoder，是 Seq2Seq 常用模型，分成 Encoder 和 Decoder 兩部分，對於 Encoder 部分，使用 fully-visible attention mask，可以看到全體，之後將結果輸給 Decoder，而 Decoder 使用 causal mask 只能看到自己之前的輸出來預測下一次的結果。

<img src="C:\Users\kai\AppData\Roaming\Typora\typora-user-images\image-20221122173712915.png" alt="image-20221122173712915" style="zoom: 67%;" />


T5 通過為每個任務對應的輸入添加不同的 prefix，可以直接應用在各種任務上，例如:

1. translation task prefix: translate English to German: <s1>
2. summarization task prefix: summarize: <s1>
3. cola (語法可接受性分類 task): cola sentence: <s1>
4. stsb (語義相似度 task): stsb sentence1: <s1> sentence2: <s2>

因為我們的 task 是 text summarization，因此加上 prefix `summarize:`。將 tokenized 的文章 token $x$ 作為 Encoder 的 input；generation 的部分則是先在 Decoder 的開頭輸入一個 start token，接著讓 Decoder 藉由 $y_1,y_2,...,y_{n-1}$ 與之前看過的文章來預測出 $y_n$，也就是計算 $p(y_i|y_1,y_2,...,y_{i-1},x)$，以 greedy 為例，會將當前預測機率最高的 token 當作是下一個時間點的 input，並繼續做預測。

<img src="C:\Users\kai\AppData\Roaming\Typora\typora-user-images\image-20221122180916437.png" alt="image-20221122180916437" style="zoom:67%;" />

model-config:

```json
{
  "_name_or_path": "google/mt5-small",
  "architectures": [
    "MT5ForConditionalGeneration"
  ],
  "d_ff": 1024,
  "d_kv": 64,
  "d_model": 512,
  "decoder_start_token_id": 0,
  "dropout_rate": 0.1,
  "eos_token_id": 1,
  "feed_forward_proj": "gated-gelu",
  "initializer_factor": 1.0,
  "is_encoder_decoder": true,
  "layer_norm_epsilon": 1e-06,
  "model_type": "mt5",
  "num_decoder_layers": 8,
  "num_heads": 6,
  "num_layers": 8,
  "pad_token_id": 0,
  "relative_attention_max_distance": 128,
  "relative_attention_num_buckets": 32,
  "tie_word_embeddings": false,
  "tokenizer_class": "T5Tokenizer",
  "torch_dtype": "float32",
  "transformers_version": "4.18.0.dev0",
  "use_cache": true,
  "vocab_size": 250100
}
```


### Preprocessing (1%)

#### Describe your preprocessing (e.g. tokenization, data cleaning and etc.)

如果 context 的長度超過自訂的長度，會將超過的部分 truncated。

mt5 使用 T5 tokenizer，是基於 SentencePiece 的一種非監督式 text tokenizer 與 detokenizer，和一般非監督式分詞系統認為詞彙量有無窮大不同，其詞彙量是訓練前就會決定好的。

底層使用 subword unit(BPE) 和 unigram language model，藉由統計每一個連續的 subword pair，選擇機率最高的將其合併，直到 vocab size 滿意為止。

在中文方面，還是會將一個中文字當作一個 token。

## Q2: Training (2%)

### Hyperparameter (1%)

#### Describe your hyperparameter you use and how you decide it.
- pretrained model: google/mt5-small
- learning_rate: 0.001
- train_batch_size: 4
- eval_batch_size: 4
-  seed: 42                                                                                                                                                                              
- gradient_accumulation_steps: 4
- total_train_batch_size: 16
- optimizer: Adafactor
- lr_scheduler_type: linear
- lr_scheduler_warmup_ratio: 0.1
- num_epochs: 3.0
- max_source_length: 1024
- max_target_length: 128

參考助教的建議，使用 Adafactor optimizer，與一般的 adam 相比，少了 momentum，因此減少了記憶體用量，另外也有使用 fp16 訓練減少記憶體用量。

在 learning rate 上，使用稍微大一些的 value，因為使用原本 default 的 learning rate: 3e-5 訓練時， loss 下降的很慢，因此有調大一些。

### Learning Curves (1%)

#### Plot the learning curves (ROUGE versus training steps)

<img src="C:\Users\kai\AppData\Roaming\Typora\typora-user-images\image-20221123151436958.png" alt="image-20221123151436958" style="zoom:50%;" />

![image-20221123151549231](C:\Users\kai\AppData\Roaming\Typora\typora-user-images\image-20221123151549231.png)

## Q3: Generation Strategies(6%)

### Stratgies (2%)

#### Describe the detail of the following generation strategies:

- Greedy

  選擇當前 step 機率最大的 token。

  以下圖為例，從 `The` 開始，會選擇機率最大的 `nice` (0.5)，接著基於 `nice` 會選擇機率最大的 `woman` (0.4)，總體的機率為 0.5 X 0.4 = 0.2。因為不需要考慮前面的 step，只需要選擇當前 step 機率最大的 token，因此 greedy 是比較有效率的。

  <img src="C:\Users\kai\Desktop\adl\greedy_search.png" alt="greedy_search" style="zoom: 67%;" />

  

- Beam search

  假設 num_beams = 2，生成第一個 token 的時候，會從詞表選擇機率最大的 2 個 tokens。以下圖為例，從 `The` 開始，會選擇 `nice` (0.5) 和 `dog` (0.4)。

  生成第二個 token 的時候，將  `nice` 和 `dog` 分別與它們的分支進行組合，得到 6 個 sequences，並計算這 6 個 sequences 的機率，留下機率最大的 2 個 sequences，( `nice`, `woman` ) (0.5 x 0.4 = 0.2)、( `dog`, `has` ) (0.4 x 0.9 = 0.36)。

  後面會不斷重複這個過程，直到遇到結束符或者達到最大長度為止。最終輸出機率最高的 2 個 sequences。

  因為 Beam search 是挑選 overall 機率最大的，因此可以解決 greedy 可能錯過 overall 機率最高的 sequence。

  <img src="C:\Users\kai\Desktop\adl\beam_search.png" alt="beam_search" style="zoom: 67%;" />

- Top-k Sampling

  Top-k Sampling 會將機率前 k 高的 tokens 保留下來，並依據原本的機率重新分配它們的採樣機率。

  以下圖為例，在預測下一個 token 時，有 `nice`、`dog`、`car`、`woman`、`guy `、`man`、`people`... 等可能，但如果我們設置 k = 6，此時只剩下  `nice`、`dog`、`car`、`woman`、`guy `、`man ` 這六種可能，去預測下一個 token。

  這種 sampling strategy 可以刪除很多機率很低的 tokens。 但是選擇一個好的 k 值可能很困難，因為每個  step 的機率分佈或 tokens 都不同。 這代表相同的 k 值可能允許在一個 step 中包含奇怪的 tokens，像是左圖的 `down` 和 `a`，同時也可能排除合理的 tokens，像是右圖的 `people`、`big`、`house`、`cat`。 

  <img src="C:\Users\kai\Desktop\adl\top_k_sampling.png" alt="top_k_sampling" style="zoom: 50%;" />

- Top-p Sampling

  Top-p Sampling 是從機率最大的 token 開始累加機率，直到我們所設定的 threshold p，來得到一個 token set。

  以下圖為例，我們設定 p = 0.94，會從機率最大的 `nice` 開始累加機率來擴大 token set，直到 token set 內所有 token 的機率總和 >= 0.94，並依據原本的機率重新分配它們的採樣機率。

  如此以來，便可以解決 Top-k Sampling 無法依據機率分布動態調整 k 值的問題。像是下圖，在左邊較平均的機率分布中，取的範圍就比較寬，而右邊較為集中的機率分布中，取的範圍就比較窄。

  ![top_p_sampling](C:\Users\kai\Desktop\adl\top_p_sampling.png)

- Temperature

  Temperature 是一種可以調整 token 機率分布集中與平均的一個 hyperparameter，較高的 Temperature 會使的機率分布較為平均，decode 出來的結果可能會有較大的 diversity；而較低的 Temperature 會使分布更加集中，diversity 較低。

  $\tau$ 為 Temperature。
  $$
  P(w_t)=\frac{e^{S_w}}{\sum_{w^\prime\in V}e^{S_{w^\prime}}}\Longrightarrow P(w_t)=\frac{e^{S_w/\tau}}{\sum_{w^\prime\in V}e^{S_{w^\prime}/\tau}}
  $$

### Hyperparameters (4%)

#### Try at least 2 settings of each strategies and compare the result. What is your final generation strategy? (you can combine any of them)

- Greedy v.s. Sample

|        |  1-r   |  1-p   |  1-f   |  2-r   |  2-p   |  2-f   |  l-r   |  l-p   |  l-f   |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| Greedy | 0.2277 | 0.2917 | 0.2473 | 0.0856 | 0.1043 | 0.0909 | 0.2052 | 0.2635 | 0.2229 |
| Sample | 0.1893 | 0.2074 | 0.1921 | 0.0604 | 0.0639 | 0.0601 | 0.1683 | 0.1846 | 0.1708 |

​	greedy 比起 sample 來的好，因為 sample 會有機會 sample 到比較不好的選擇，造成後續 decode 的結果也變差。

- Greedy v.s. Beam search

|           |  1-r   |  1-p   |  1-f   |  2-r   |  2-p   |  2-f   |  l-r   |  l-p   |  l-f   |
| :-------: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
|  Greedy   | 0.2277 | 0.2917 | 0.2473 | 0.0856 | 0.1043 | 0.0909 | 0.2052 | 0.2635 | 0.2229 |
| beams = 2 | 0.2416 | 0.2943 | 0.2568 | 0.0954 | 0.1132 | 0.1001 | 0.2175 | 0.2661 | 0.2316 |
| beams = 4 | 0.2490 | 0.2950 | 0.2611 | 0.1007 | 0.1167 | 0.1044 | 0.2248 | 0.2671 | 0.2359 |

可以看到 Beam search 比 Greedy 好，這是因為 Beam Search 較不會漏掉前面機率較低但整體機率較大的答案。

num_beams 越大，結果應該就會越好，因為更不會漏掉整體機率較大的答案，但是 num_beams 越大，inference 的時間就會越長，因為必須維持的資訊較多。

- Top-k v.s. Sample

|            |  1-r   |  1-p   |  1-f   |  2-r   |  2-p   |  2-f   |  l-r   |  l-p   |  l-f   |
| :--------: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
|   Sample   | 0.1893 | 0.2074 | 0.1921 | 0.0604 | 0.0639 | 0.0601 | 0.1683 | 0.1846 | 0.1708 |
| top-k = 10 | 0.2095 | 0.2411 | 0.2176 | 0.0701 | 0.0770 | 0.0710 | 0.1861 | 0.2145 | 0.1933 |
| top-k = 40 | 0.1907 | 0.2132 | 0.1954 | 0.0613 | 0.0664 | 0.0617 | 0.1691 | 0.1892 | 0.1733 |

使用 Top-k Sampling 可以有較好的表現，但如果 k 值太大，結果就和原本差不多，這應該是因為 Top-k Sampling 無法依據機率分布動態調整 k 值的問題，可能包含了奇怪的 tokens。

- Top-p v.s. Top-k v.s. Sample

|             |  1-r   |  1-p   |  1-f   |  2-r   |  2-p   |  2-f   |  l-r   |  l-p   |  l-f   |
| :---------: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
|   Sample    | 0.1893 | 0.2074 | 0.1921 | 0.0604 | 0.0639 | 0.0601 | 0.1683 | 0.1846 | 0.1708 |
| top-k = 10  | 0.2095 | 0.2411 | 0.2176 | 0.0701 | 0.0770 | 0.0710 | 0.1861 | 0.2145 | 0.1933 |
| top-p = 0.6 | 0.2176 | 0.2606 | 0.2301 | 0.0786 | 0.0903 | 0.0814 | 0.1947 | 0.2335 | 0.2059 |
| top-p = 0.8 | 0.2086 | 0.2421 | 0.2177 | 0.0723 | 0.0813 | 0.0742 | 0.1854 | 0.2150 | 0.1933 |

Top-p Sampling 的表現比 Top-k Sampling 和原本的 Sample 更好，這應該是得利於解決 Top-k Sampling 的缺點，可以避免包含奇怪的 tokens 或是排除合理的 tokens。

- w/ and w/o Temperature


|      | top-p = 0.6 | top-p = 0.6, temperature = 0.7 | top-p = 0.6, temperature = 1.3 |
| :--: | :---------: | :----------------------------: | :----------------------------: |
| 1-r  |   0.2176    |             0.2251             |             0.2052             |
| 1-p  |   0.2606    |             0.2781             |             0.2353             |
| 1-f  |   0.2301    |             0.2411             |             0.2126             |
| 2-r  |   0.0786    |             0.0830             |             0.0696             |
| 2-p  |   0.0903    |             0.0980             |             0.0772             |
| 2-f  |   0.0814    |             0.0870             |             0.0708             |
| l-r  |   0.1947    |             0.2020             |             0.1818             |
| l-p  |   0.2335    |             0.2495             |             0.2088             |
| l-f  |   0.2059    |             0.2162             |             0.1884             |

Temperature 在 Top-p Sampling 的影響比較大，因為 Temperature 會影響每個 step 候選 tokens 的數量，Temperature < 1 時，會使機率分布更加集中(sharpen)，機率大的 token 的機率會變得更大，小的變得更小，讓候選的數量變少，更容易 sample 到機率大的 token；Temperature > 1 時，會使機率分布更加平均(smooth)，讓候選的數量變多，因此會有點像是從 Top-p 中去 random sample。

Final generation strategy: Beam search, number of beams = 4