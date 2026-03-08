# Building Structure Modeling - Handoff

## 현재 상태
- **앱 실행 중**: http://localhost:8570 (polygon15_rl.py)
- **TensorBoard**: http://localhost:6006 (SAC 학습 그래프)
- **GitHub**: https://github.com/GeoEmmy/building-structure-modeling

## 최근 작업

### 2026-03-08: 매스라인 방향 축열 정렬
- minimum_rotated_rectangle로 주축 방향 계산
- polygon을 주축 정렬 후 그리드 생성
- 구조물 전체를 원래 방향으로 회전
- 축열이 매스라인 방향을 따라감

### 2026-03-08: 비정형 polygon 슬라브 지원
- triangle/earcut 삼각화 엔진 적용
- PATCH 형태 → 정확한 polygon 형태 슬라브

## 주요 파일

| 파일 | 설명 |
|-----|------|
| `polygon15_rl.py` | 메인 앱 (RL + 휴리스틱 최적화) |
| `polygon15.py` | 원본 규칙 기반 버전 |
| `sac_mass_stable_v1.zip` | 학습된 SAC 모델 |
| `merged_address_with_area.parquet` | 지번 데이터 |
| `requirements.txt` | 의존성 패키지 |

## 명령어

```bash
# 앱 실행
streamlit run polygon15_rl.py --server.port 8570

# TensorBoard (학습 그래프)
tensorboard --logdir=sac_stable_logs --port=6006

# 의존성 설치
pip install -r requirements.txt
```

## 알려진 이슈

1. **SAC 모델 로드 경고**: numpy 버전 불일치로 lr_schedule 역직렬화 실패 (기능에는 영향 없음)
2. **use_container_width 경고**: Streamlit 2025-12-31 이후 deprecated → `width='stretch'`로 변경 필요
3. **휴리스틱 vs SAC**: 휴리스틱 최적화가 SAC보다 더 정확한 결과 생성

## TODO

- [ ] Streamlit `use_container_width` → `width` 파라미터 마이그레이션
- [ ] SAC 모델 재학습 (현재 numpy 2.x 환경에서)
- [ ] Streamlit Cloud 배포 테스트
- [ ] 슬라브 형태 라이노에서 최종 확인
