# SSAFY AI Challenge — VQA 1st Place Solution

SSAFY 16기 AI 챌린지 이미지 기반 질의응답(VQA) 개인전 **최종 1위** 솔루션입니다. Qwen3.5-27B를 LoRA로 학습하고, 선택지 순서 TTA와 불확실한 문항에 대한 선택적 확률 결합으로 제출 점수를 개선했습니다.

- 작성자: 김기현 ([kimkihyun1](https://github.com/kimkihyun1))
- 대회: [SSAFY 16-1 AI Challenge](https://www.kaggle.com/competitions/ssafy-16-1-ai)
- 최고 제출 점수: **0.95270**
- 기본 TTA4 제출 점수: **0.95072**

순위와 제출 점수는 참가자의 대회 기록 기준입니다. Public/Private 구분 및 평가 지표 명칭은 이 저장소에서 별도로 확정하지 않았습니다. 점수 차이는 **+0.00198**이며, 재실행 성능을 보장하는 수치는 아닙니다.

## 포함된 노트북

| 노트북 | 역할 | 확인된 제출 점수 |
| --- | --- | --- |
| [A100_FULL5073_TTA4_FINAL](notebooks/vqa_qwen35_27b_A100_FULL5073_TTA4_FINAL.ipynb) | 전체 5,073개 학습 데이터로 LoRA 학습 및 TTA4 추론 | 0.95072 |
| [LASTMILE_BASEOFF_MULTICROP_FINAL](notebooks/vqa_qwen35_27b_LASTMILE_BASEOFF_MULTICROP_FINAL.ipynb) | 기존 캐시 복원, LoRA OFF 추론, 선택적 확률 결합 및 추가 실험 | BASEOFF_BALANCED: **0.95270** |

노트북의 원래 코드와 실험 설명을 보존하고 실행 출력·실행 횟수·Colab 사용자 메타데이터를 제거했습니다. 노트북 끝부분의 후보 추천 순서는 대회 당시 실험 계획이며, 실제 최고 점수 후보는 **BASEOFF_BALANCED**입니다.

## 문제와 접근 방법

이미지, 질문, 네 개의 선택지 `a/b/c/d`를 입력받아 정답 하나를 예측합니다. 학습 시 선택지 순서를 무작위로 섞고 정답 부분에 대한 언어 모델 손실을 사용합니다. 추론에서는 다음 토큰의 `a/b/c/d` logits에 softmax를 적용하고, 바뀐 선택지 순서를 원래 순서로 되돌린 후 확률을 평균합니다.

### 1. 전체 데이터 LoRA 학습과 TTA4

| 설정 | 값 |
| --- | --- |
| 기본 모델 | `Qwen/Qwen3.5-27B` |
| 학습 데이터 | 5,073개 |
| Epoch / learning rate | 1 / `1e-4` |
| LoRA rank / alpha / dropout | 16 / 32 / 0.05 |
| 학습 batch / gradient accumulation | 1 / 8 |
| 추론 batch | 1 |
| 최소 / 최대 visual tokens | 256 / 1,024 |
| Seed | 42 |
| 정밀도 | BF16, `auto` 설정에서 메모리 부족 시 8-bit fallback |
| TTA | 선택지 순서 4가지 |

`v1024`는 이미지 한 변의 픽셀 수가 아니라 **최대 visual token 수**입니다. 기록된 제출은 BF16 실행 결과입니다. 8-bit 또는 다른 설정의 결과를 동일한 제출로 간주하지 않습니다.

### 2. Adaptive TTA와 고해상도 추론 결과 활용

Last-mile 노트북은 기본 TTA4 결과에서 불확실한 1,000문항을 선택한 후, 이전에 저장한 20개 추가 선택지 순서 결과를 불러와 TTA24 확률을 복원합니다. 다시 선정한 400문항에는 최대 visual tokens 1,536으로 추론한 TTA4 결과를 결합합니다.

불확실도에는 상위 두 선택지의 확률 차이, entropy, 최대 확률, TTA 답변 일치율을 사용합니다. 고해상도 대상의 기존 확률은 `0.65 × Adaptive 확률 + 0.35 × HiRes 확률`입니다.

**이 중간 추론을 생성하는 Adaptive 노트북은 이번 두 파일 구성에 포함되지 않습니다.** Last-mile 실행에는 아래 설명의 기존 캐시가 필요합니다.

### 3. 최고 점수: BASEOFF_BALANCED

불확실한 500문항에 대해 학습한 LoRA adapter를 일시적으로 끄고 원본 모델로 TTA4를 실행합니다. 원본 모델의 예측이 기존 예측과 다르고 아래 조건을 모두 만족할 때만 확률을 결합합니다.

| 조건 | 기준 |
| --- | --- |
| 기존 예측의 상위 두 확률 차이 | `< 0.15` |
| 원본 모델이 선택한 답의 평균 확률 | `>= 0.60` |
| 원본 모델 TTA 답변 일치율 | `>= 0.75` |
| FT TTA24에서 기존 답 확률 − 원본 모델 답 확률 | `<= 0.12` |

대상 문항은 `0.50 × 기존 확률 + 0.50 × 원본 모델 확률`로 결합하고 argmax로 정답을 결정합니다. 조건을 만족하지 않는 문항은 기존 확률을 유지합니다. 결합 대상으로 선택되어도 최종 답이 반드시 바뀌는 것은 아닙니다.

**최고 점수 CSV는 Last-mile 노트북의 `9. LoRA OFF 기반 보수적 후보 생성` 단계에서 생성됩니다.** 뒤의 Multi-crop 및 3-way consensus는 별도 후보를 만드는 추가 실험으로, BASEOFF_BALANCED 생성에 필요하지 않습니다.

## 제출 결과

| 생성 노트북 | 제출 파일 | 점수 |
| --- | --- | --- |
| A100_FULL5073_TTA4_FINAL | `submission_qwen35_27b_full5073_bf16_v1024_e1_TTA4.csv` | 0.95072 |
| LASTMILE_BASEOFF_MULTICROP_FINAL | `submission_qwen35_27b_LASTMILE_BASEOFF_BALANCED.csv` | **0.95270** |

Multi-crop 등의 다른 후보 점수는 확인되지 않아 기재하지 않았습니다.

## 실행 환경

첫 번째 원본 노트북의 실행 로그에서 확인한 환경은 다음과 같습니다.

| 항목 | 기록된 환경 |
| --- | --- |
| GPU | NVIDIA A100-SXM4-80GB |
| PyTorch | `2.11.0+cu128` |
| Transformers | `5.16.0.dev0` |
| PEFT | `0.20.0` |

각 노트북의 설치 셀을 먼저 실행하고 Colab 세션을 재시작합니다. 설치 셀은 Transformers GitHub 최신 소스를 설치하도록 되어 있어 과거 실행 환경의 완전한 lockfile은 아닙니다. 당시 Transformers commit SHA와 전체 의존성 버전은 확인되지 않았으므로, 재실행 전 모델 클래스와 버전 호환성을 확인해야 합니다. 대규모 GPU 학습·추론은 저장소 정리 과정에서 재실행하지 않았습니다.

## 데이터 준비

대회 데이터를 별도로 준비합니다. 데이터·가중치·캐시·제출 CSV는 저장소에 포함하지 않습니다.

기본 `ROOT`는 `/content/drive/MyDrive/ssafy-16-1-ai-8-28`입니다. 이 위치 아래 다음 파일과 디렉터리를 배치합니다.

| 경로 | 용도 |
| --- | --- |
| `train.csv`, `train/` | 학습 데이터와 이미지 |
| `test.csv`, `test/` | 테스트 데이터와 이미지 |
| `sample_submission.csv` | 제출 ID 순서 맞추기 |
| `dev.csv`, `dev/` | 선택적 보조 데이터; 첫 노트북의 전체 학습은 train 기준 |

학습 CSV 열은 `id, path, question, a, b, c, d, answer`, 테스트 CSV 열은 정답 열을 제외한 동일 구성입니다. `path`에 해당하는 이미지가 있어야 합니다.

## 실행 순서와 재현 범위

### A. 기본 TTA4 제출 생성

1. 첫 번째 노트북을 Colab A100 80GB 환경에서 엽니다.
2. 패키지 설치 및 세션 재시작 후 Drive를 마운트합니다.
3. `ROOT`를 확인하고 셀을 순서대로 실행합니다.
4. 학습한 adapter, run config, permutation별 `.npz` 캐시 및 제출 결과를 보존합니다.
5. `ROOT/submission/submission_qwen35_27b_full5073_bf16_v1024_e1_TTA4.csv`를 확인합니다.

### B. 기존 캐시를 이용한 최고 점수 후보 생성

두 번째 노트북 실행 전에 다음 산출물이 필요합니다.

| 입력 | 예상 위치 또는 패턴 |
| --- | --- |
| 학습 adapter | `qwen35_27b_FINAL_full5073_bf16_v1024_epoch1/` |
| 정밀도 설정 | `qwen35_27b_FINAL_full5073_v1024_epoch1_run_config.json` |
| 기본 TTA4 캐시 | `tta_cache_qwen35_27b_full5073_bf16_v1024_e1/` 아래 `*_perm1.npz` ~ `*_perm4.npz` |
| Adaptive 추가 20개 캐시 | `adaptive_tta27b/extra_v1024_h1000_<hash>_k20/` |
| HiRes TTA4 캐시 | `adaptive_tta27b/hires_v1536_h400_<hash>_tta4/` |

`<hash>`는 선택된 ID 목록에서 계산됩니다. 임의로 폴더 이름만 바꾸면 안 되며, 같은 데이터 순서·선정 기준으로 생성된 캐시가 필요합니다. 캐시는 본인이 생성한 신뢰할 수 있는 파일만 사용합니다.

1. 두 번째 노트북의 `ROOT`, `FINAL_ADAPTER_DIR`, `RUN_CONFIG_PATH`, `BASE_TTA_CACHE_DIR`를 확인합니다. 경로를 바꾸면 이 네 설정을 함께 수정합니다.
2. 설치·마운트 후 **0~9번 단계**를 순서대로 실행합니다.
3. `ROOT/submission/submission_qwen35_27b_LASTMILE_BASEOFF_BALANCED.csv`를 확인합니다.
4. Multi-crop 실험이 필요할 때만 10번 이후 단계를 실행합니다.

**두 노트북만으로는 Adaptive/HiRes 캐시를 처음부터 생성할 수 없습니다.** 캐시가 없는 새 환경에서 최고 점수 파이프라인을 재현하려면 별도의 `vqa_qwen35_27b_ADAPTIVE_TTA24_HIRES_FINAL.ipynb`가 추가로 필요합니다. 첫 노트북 → Adaptive/HiRes 캐시 준비 → Last-mile 순서이며, 캐시 없이 첫 노트북에서 바로 Last-mile로 진행하면 캐시 확인 단계에서 중단됩니다.

## 저장소 검증 및 공개 범위

- 노트북 JSON 형식, Python 셀의 구문 및 최고 점수 파일 생성 코드를 정적으로 확인했습니다.
- 실행 출력·사용자 메타데이터는 제거했으며, 코드 셀의 로직은 원본과 동일하게 유지했습니다.
- Kaggle 재제출 및 GPU 재학습·재추론은 수행하지 않았습니다.
- 대회 데이터와 모델 가중치는 각 제공처의 이용 조건을 따릅니다. 이 저장소에 별도 코드 라이선스는 부여하지 않았습니다.

## 참고

- [Qwen3.5-27B 모델](https://huggingface.co/Qwen/Qwen3.5-27B)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [Hugging Face PEFT](https://github.com/huggingface/peft)

핵심 경험은 전체 문항에 무조건 추론을 추가하는 대신, 불확실한 문항을 선별하고 fine-tuned 모델과 원본 모델의 예측을 조건부로 결합해 연산을 집중한 것입니다.
