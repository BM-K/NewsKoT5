# NewsT5
The training data for this T5 model consists of Korean news articles. However, the performance has not been fine-tuned through the use of small batches and a limited number of training steps, so it may not be fully optimized.

## Quick tour
```python
from transformers import AutoTokenizer, T5ForConditionalGeneration
  
tokenizer = AutoTokenizer.from_pretrained("BM-K/")
model = T5ForConditionalGeneration.from_pretrained("BM-K/")

inputs = tokenizer("안녕 세상아!", return_tensors="pt")
outputs = model(**inputs)
```

## News Summarization Performance (F1-score)
- Dacon 한국어 문서 생성요약 AI 경진대회 [Dataset](https://dacon.io/competitions/official/235673/overview/description)
    - Training: -
    - Validation: -
    - Test: -

| | #Param | rouge-1 |rouge-2|rouge-l|
|-------|--------:|--------:|--------:|--------:|
| pko-t5-small | 77M | - | - | - |
| NewsT5-small | 77M | - | - | - |

- AI-Hub 문서요약 텍스트 [Dataset](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=97)
    - Training: -
    - Validation: -
    - Test: -

| | #Param | rouge-1 |rouge-2|rouge-l|
|-------|--------:|--------:|--------:|--------:|
| pko-t5-small | 77M | - | - | - |
| NewsT5-small | 77M | - | - | - |

- [pko-t5-small](https://github.com/paust-team/pko-t5)
