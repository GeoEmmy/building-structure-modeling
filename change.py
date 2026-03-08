import os
import geopandas as gpd
import pandas as pd
from glob import glob
import re

# 📁 SHP 폴더 경로
folder_path = r"D:\python\lca"

# 🔍 모든 .shp 파일 검색
shp_files = glob(os.path.join(folder_path, "*.shp"))

# 🎯 유지할 컬럼 (PAREA 포함)
columns_to_keep = ["SIDO_NM", "SGG_NM", "EMD_NM", "MNNM", "SLNO", "PAREA", "geometry"]

# 📦 결과 리스트
merged_list = []

for shp_file in shp_files:
    try:
        gdf = gpd.read_file(shp_file)

        if gdf.crs is None:
            gdf.set_crs(epsg=5179, inplace=True)

        gdf = gdf.to_crs(epsg=4326)

        # ✅ 본번에서 지목 제거 (숫자만 추출)
        if "MNNM" in gdf.columns:
            def extract_number(value):
                match = re.match(r"(\d+)", str(value))
                return int(match.group(1)) if match else None
            gdf["MNNM"] = gdf["MNNM"].apply(extract_number)

        # ✅ 부번 숫자화
        if "SLNO" in gdf.columns:
            gdf["SLNO"] = pd.to_numeric(gdf["SLNO"], errors="coerce")



        # ✅ 컬럼 고정 추출
        gdf = gdf[columns_to_keep]

        merged_list.append(gdf)
        print(f"✅ 로드 완료: {os.path.basename(shp_file)} ({len(gdf)}건)")

    except Exception as e:
        print(f"❌ 오류: {os.path.basename(shp_file)} - {e}")

# 💾 저장
if merged_list:
    merged_gdf = pd.concat(merged_list, ignore_index=True)
    geo_merged = gpd.GeoDataFrame(merged_gdf, geometry="geometry", crs="EPSG:4326")

    out_path = os.path.join(folder_path, "merged_address_with_area.parquet")
    geo_merged.to_parquet(out_path, index=False)
    print(f"\n🎉 저장 완료: {out_path}")
else:
    print("❌ 병합할 SHP 없음")