# Building Structure Modeling - Handoff

## 현재 상태
- **로컬 앱**: http://localhost:8580 (polygon15_rl.py)
- **Streamlit Cloud**: https://building-structure-modeling-nzrfqs8zhbvkwqvwo3tkyx.streamlit.app/
  - ⚠️ Main file path를 `polygon15_rl.py`로 변경 필요
- **GitHub**: https://github.com/GeoEmmy/building-structure-modeling
- **TensorBoard**: http://localhost:6006

## 최근 작업

### 2026-03-09: Streamlit Cloud 배포 호환성
- gymnasium/stable-baselines3 조건부 import
- SAC 모델 파일 GitHub에 업로드
- packages.txt 추가 (GDAL 시스템 의존성)
- earcut 삼각화 엔진 우선 사용

### 2026-03-08: 매스라인 방향 축열 정렬
- minimum_rotated_rectangle로 주축 방향 계산
- 그리드 포인트를 원본 좌표계로 역회전
- 기둥/보는 매스라인 방향, 슬라브는 원본 polygon 형태

## 주요 파일

| 파일 | 설명 |
|-----|------|
| `polygon15_rl.py` | 메인 앱 (3가지 모드: 규칙/휴리스틱/SAC) |
| `polygon15.py` | 원본 규칙 기반 버전 (구버전) |
| `sac_mass_stable_v1.zip` | SAC 학습 모델 |
| `sac_mass_general_v1_stable.zip` | SAC 일반화 모델 |
| `merged_address_with_area.parquet` | 지번 데이터 |
| `requirements.txt` | Python 의존성 |
| `packages.txt` | 시스템 의존성 (GDAL) |

## 명령어

```bash
# 로컬 앱 실행
streamlit run polygon15_rl.py --server.port 8580

# TensorBoard (학습 그래프)
tensorboard --logdir=sac_stable_logs --port=6006

# 의존성 설치
pip install -r requirements.txt
```

## 알려진 이슈

1. **Streamlit Cloud 파일 경로**: Main file path가 `polygon15.py`로 설정됨 → `polygon15_rl.py`로 변경 필요
2. **SAC 모델 경고**: numpy 버전 불일치로 lr_schedule 역직렬화 실패 (기능에는 영향 없음)
3. **use_container_width 경고**: Streamlit deprecated → `width='stretch'`로 변경 필요
4. **휴리스틱 vs SAC**: 휴리스틱 최적화가 더 정확한 결과 생성

## TODO

- [x] Streamlit Cloud 배포
- [x] SAC 모델 GitHub 업로드
- [ ] Streamlit Cloud Main file path 변경 (`polygon15_rl.py`)
- [ ] `use_container_width` → `width` 마이그레이션
- [ ] 슬라브 형태 라이노에서 최종 확인
