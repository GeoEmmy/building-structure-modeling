# Changelog

## [2026-03-09]

### Added
- Streamlit Cloud 배포 지원
  - `packages.txt` 추가 (GDAL 시스템 의존성)
  - SAC 모델 파일 GitHub 업로드 (`sac_mass_stable_v1.zip`, `sac_mass_general_v1_stable.zip`)

### Changed
- gymnasium/stable-baselines3 조건부 import (Cloud 호환)
- `GYMNASIUM_AVAILABLE`, `SAC_AVAILABLE` 플래그로 모드 활성화 결정
- earcut 삼각화 엔진 우선 사용 (Cloud에서 triangle 컴파일 불가)

### Fixed
- 그리드 포인트를 원본 좌표계로 역회전하여 기둥/보가 슬라브 내부에 배치되도록 수정
- 슬라브를 원본 mass polygon 형태로 생성 (회전 없이)

---

## [2026-03-08]

### Added
- 매스라인 방향에 맞춘 기둥/보 축열 생성
  - `minimum_rotated_rectangle`로 주축 방향 계산
  - polygon을 주축 정렬 후 그리드 생성
  - 구조물 전체를 원래 방향으로 회전
- 비정형 polygon 슬라브 지원
  - `create_polygon_slab()` 함수 추가 (triangle/earcut 엔진)
  - `create_polygon_mesh_manual()` fallback 함수 추가
- 삼각화 패키지 requirements 추가 (mapbox-earcut, triangle)

### Changed
- `create_polygon_structure()` 함수 리팩토링
  - World 그리드 → 매스라인 방향 그리드로 변경
  - 슬라브: PATCH 형태 → 정확한 polygon 형태

### Fixed
- 슬라브가 PATCH 느낌으로 나오던 문제 수정
- triangulation 라이브러리 누락으로 인한 extrude 실패 수정

---

## [2026-03-07]

### Added
- `polygon15_rl.py` 생성 (RL + 휴리스틱 최적화 버전)
- 3가지 모드 지원:
  - 규칙 기반 (정사각형 그리드)
  - 휴리스틱 최적화 (랜덤 탐색 + 로컬 서치)
  - 강화학습 SAC (학습된 모델 사용)
- GitHub 저장소 생성: GeoEmmy/building-structure-modeling
- Streamlit Cloud 배포 준비 (requirements.txt, parquet 데이터)

### Changed
- "강화학습" → "휴리스틱 최적화" 명칭 정확화
- 휴리스틱 최적화를 기본 모드로 설정

---

## [초기 버전]

### Added
- `polygon15.py` - 지번 기반 구조 자동 모델링 시스템
- 주소 검색 → 대지 polygon 조회 → 구조물 자동 생성
- Plotly 3D 시각화
- 건폐율/용적률/이격거리 적용
