import streamlit as st
import pandas as pd
import os
from PIL import Image, ExifTags
import folium
from folium.plugins import MarkerCluster, Fullscreen
from streamlit_folium import st_folium
from ultralytics import YOLO
import io

# --- 1. 초기 설정 및 모델 로드 ---
st.set_page_config(page_title="쓰레기 위치 맵핑 및 카운터", layout="wide")

@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')

model = load_model()
DATA_FILE = "trash_data.csv"

# --- 2. 헬퍼 함수 ---
def get_decimal_from_dms(dms, ref):
    degrees, minutes, seconds = dms[0], dms[1], dms[2]
    decimal = float(degrees) + float(minutes)/60 + float(seconds)/3600
    if ref in ['S', 'W']:
        decimal = -decimal
    return decimal

# 🌟 개선: 함수가 이미지 바이트 대신 PIL 이미지 객체를 직접 받도록 수정
def extract_gps_info(pil_image):
    try:
        exif = pil_image._getexif()
        if not exif: return None, None

        gps_info = {}
        for key, val in exif.items():
            tag = ExifTags.TAGS.get(key)
            if tag == 'GPSInfo':
                for t in val:
                    sub_tag = ExifTags.GPSTAGS.get(t, t)
                    gps_info[sub_tag] = val[t]
                break
        
        if 'GPSLatitude' in gps_info and 'GPSLongitude' in gps_info:
            lat = get_decimal_from_dms(gps_info['GPSLatitude'], gps_info['GPSLatitudeRef'])
            lon = get_decimal_from_dms(gps_info['GPSLongitude'], gps_info['GPSLongitudeRef'])
            return lat, lon
    except (AttributeError, KeyError, IndexError):
        return None, None
    return None, None

# 🌟 개선: 함수가 이미지 바이트 대신 PIL 이미지 객체를 직접 받도록 수정
def count_trash_in_image(pil_image):
    results = model(pil_image)
    return len(results[0].boxes)

def load_data():
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
        # 데이터 로드 시 trash_count를 정수형으로 변환
        df['trash_count'] = df['trash_count'].astype(int)
        return df
    return pd.DataFrame(columns=["filename", "latitude", "longitude", "trash_count"])

def save_data(new_data_df):
    if os.path.exists(DATA_FILE):
        existing_data = pd.read_csv(DATA_FILE)
        updated_data = pd.concat([existing_data, new_data_df], ignore_index=True)
    else:
        updated_data = new_data_df
    updated_data = updated_data.drop_duplicates(subset=["filename"], keep="last")
    updated_data.to_csv(DATA_FILE, index=False)

# --- 3. Streamlit UI 구성 ---
st.title("🌍 스마트 쓰레기 맵핑 프로그램")
st.write("GPS 정보가 포함된 사진을 업로드하고 버튼을 누르면 AI가 분석을 시작합니다. 잘못 예측된 개수는 오른쪽 표에서 직접 수정할 수 있습니다.")

with st.sidebar:
    st.header("사진 업로드")
    uploaded_files = st.file_uploader("사진을 선택하세요.", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    st.markdown("---")
    if st.button("🗑️ 전체 데이터 초기화"):
        if os.path.exists(DATA_FILE):
            os.remove(DATA_FILE)
            st.success("데이터가 초기화되었습니다.")
            st.rerun()

# --- 4. 데이터 처리 로직 ---
if uploaded_files:
    if st.button(f"🚀 {len(uploaded_files)}개 사진 분석 및 등록하기", type="primary"):
        with st.spinner('이미지를 분석하고 위치 정보를 추출하는 중입니다...'):
            new_records = []
            for file in uploaded_files:
                try:
                    # 🌟 개선: 이미지를 한번만 열어서 처리
                    pil_img = Image.open(file)
                    lat, lon = extract_gps_info(pil_img)
                    
                    if lat is not None and lon is not None:
                        count = count_trash_in_image(pil_img)
                        new_records.append({
                            "filename": file.name, "latitude": lat,
                            "longitude": lon, "trash_count": count
                        })
                    else:
                        st.warning(f"'{file.name}' 파일에 GPS 정보가 없습니다.")
                except Exception as e:
                    st.error(f"'{file.name}' 처리 중 오류 발생: {e}")

            if new_records:
                save_data(pd.DataFrame(new_records))
                st.success("사진 정보가 성공적으로 지도에 등록되었습니다!")
                st.rerun()

# --- 5. 지도 및 데이터 시각화 ---
df = load_data()
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📍 쓰레기 분포 지도")
    if not df.empty:
        m = folium.Map(location=[df["latitude"].mean(), df["longitude"].mean()], zoom_start=13)
        Fullscreen().add_to(m)

        # 🌟 수정: 자바스크립트에서 marker.options.custom_trash_count 대신 marker.options.title을 읽도록 변경
        icon_create_function = """
        function(cluster) {
            var markers = cluster.getAllChildMarkers();
            var sum = 0;
            for (var i = 0; i < markers.length; i++) {
                sum += parseInt(markers[i].options.title) || 0;
            }
            var iconHtml = '<div style="background-color: #CB4335; color: white; border-radius: 50%; width: 36px; height: 36px; display: flex; justify-content: center; align-items: center; font-weight: bold; font-size: 13px; border: 2px solid white; box-shadow: 2px 2px 4px rgba(0,0,0,0.5);">' + sum + '</div>';
            return L.divIcon({ html: iconHtml, className: 'marker-cluster-custom', iconSize: L.point(36, 36) });
        }
        """

        marker_cluster = MarkerCluster(
            name="Trash Cluster",
            icon_create_function=icon_create_function,
            options={'maxClusterRadius': 40} # 반경을 조금 넓혀 클러스터링이 더 잘 되도록 조정
        ).add_to(m)

        for _, row in df.iterrows():
            trash_count = int(row['trash_count'])
            if trash_count <= 2:
    color = "#2ECC71"
elif trash_count <= 5:
    color = "#F1C40F"
elif trash_count <= 10:
    color = "#E67E22"
else:
    color = "#E74C3C"

html_icon = f"""
<div style="
    position:relative;
    width:34px;
    height:34px;
    background:{color};
    border-radius:50% 50% 50% 0;
    transform:rotate(-45deg);
    border:3px solid white;
    box-shadow:0 4px 12px rgba(0,0,0,.35);
">

<div style="
    position:absolute;
    width:100%;
    height:100%;
    display:flex;
    justify-content:center;
    align-items:center;
    transform:rotate(45deg);
    color:white;
    font-weight:bold;
    font-size:15px;
">
{trash_count}
</div>

</div>
"""

            # 🌟 수정: 유실되는 options 대신, 안정적인 'title' 속성에 쓰레기 개수(문자열)를 저장
            folium.Marker(
                [row['latitude'], row['longitude']],
                popup=f"파일명: {row['filename']}<br>쓰레기 개수: {trash_count}개",
                icon=folium.DivIcon(html=html_icon),
                title=str(trash_count) # 여기에 개수를 문자열로 저장
            ).add_to(marker_cluster)

        st_folium(m, use_container_width=True, height=550, key="trash_map")
    else:
        st.info("표시할 데이터가 없습니다. 사진을 업로드해 주세요.")

with col2:
    st.subheader("📋 데이터 목록")
    if not df.empty:
        edited_df = st.data_editor(
            df, use_container_width=True,
            column_config={"trash_count": st.column_config.NumberColumn("쓰레기 개수", min_value=0, step=1)},
            disabled=["filename", "latitude", "longitude"], key="data_editor"
        )
        if not edited_df.equals(df):
            if st.button("💾 수정사항 저장", type="primary"):
                edited_df.to_csv(DATA_FILE, index=False)
                st.success("데이터가 업데이트되었습니다.")
                st.rerun()
