# NewsKoT5
The training data for this T5 model consists of Korean news articles (29GB). However, the performance has not been fine-tuned through the use of small batches and a limited number of training steps, so it may not be fully optimized.

## Quick tour
```python
from transformers import AutoTokenizer, T5ForConditionalGeneration
  
tokenizer = AutoTokenizer.from_pretrained("BM-K/NewsKoT5-small")
model = T5ForConditionalGeneration.from_pretrained("BM-K/NewsKoT5-small")

input_ids = tokenizer("한국형발사체 누리호가 실용급 <extra_id_0> 발사체로서 ‘데뷔’를 성공적으로 <extra_id_1>", return_tensors="pt").input_ids
labels = tokenizer("<extra_id_0> 위성 <extra_id_1> 마쳤다 <extra_id_2>", return_tensors="pt").input_ids

outputs = model(input_ids=input_ids,
                labels=labels)
```

## News Summarization Performance (F1-score)
After restoring the model's tokenized output to the original text, Rouge performance was evaluated by comparing it to the reference and hypothesis tokenized using [mecab](https://konlpy.org/ko/v0.4.0/).

- Dacon 한국어 문서 생성요약 AI 경진대회 [Dataset](https://dacon.io/competitions/official/235673/overview/description)
    - Training: 29,432
    - Validation: 7,358
    - Test: 9,182

| | #Param | rouge-1 |rouge-2|rouge-l|
|-------|--------:|--------:|--------:|--------:|
| pko-t5-small | 95M | 51.48 | 33.18 | 44.96 |
| NewsT5-small | 61M | 52.15 | 33.59 | 45.41 |

- AI-Hub 문서요약 텍스트 [Dataset](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=97)
    - Training: 245,626
    - Validation: 20,296
    - Test: 9,931

| | #Param | rouge-1 |rouge-2|rouge-l|
|-------|--------:|--------:|--------:|--------:|
| pko-t5-small | 95M | 53.44 | 34.03 | 45.36 |
| NewsT5-small | 61M | 53.74 | 34.27 | 45.52 |

- [pko-t5-small](https://github.com/paust-team/pko-t5)
