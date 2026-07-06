import streamlit as st
import pandas as pd
import os
from PIL import Image, ExifTags
import folium
from folium.plugins import MarkerCluster, Fullscreen  # 🌟 Fullscreen 플러그인 추가
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
    degrees = dms[0]
    minutes = dms[1]
    seconds = dms[2]
    decimal = float(degrees) + float(minutes)/60 + float(seconds)/3600
    if ref in ['S', 'W']:
        decimal = -decimal
    return decimal

def extract_gps_info(image_bytes):
    try:
        img = Image.open(image_bytes)
        exif = img._getexif()
        if not exif:
            return None, None

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
    except Exception as e:
        return None, None
    return None, None

def count_trash_in_image(image_bytes):
    img = Image.open(image_bytes)
    results = model(img)
    count = len(results[0].boxes) 
    return count

def load_data():
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE)
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
st.write("GPS 정보가 포함된 사진을 업로드하고 버튼을 누르면 AI가 분석을 시작합니다.")

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
    st.info(f"현재 {len(uploaded_files)}개의 사진이 대기 중입니다. 아래 버튼을 눌러 분석을 시작하세요.")
    
    if st.button("🚀 사진 분석 및 지도에 등록하기", type="primary"):
        with st.spinner('이미지를 분석하고 위치 정보를 추출하는 중입니다...'):
            new_records = []
            for file in uploaded_files:
                file_bytes = file.read()
                
                lat, lon = extract_gps_info(io.BytesIO(file_bytes))
                
                if lat is not None and lon is not None:
                    count = count_trash_in_image(io.BytesIO(file_bytes))
                    new_records.append({
                        "filename": file.name,
                        "latitude": lat,
                        "longitude": lon,
                        "trash_count": count
                    })
                else:
                    st.warning(f"'{file.name}' 파일에 GPS 정보가 없습니다.")

            if new_records:
                new_df = pd.DataFrame(new_records)
                save_data(new_df)
                st.success(f"{len(new_records)}개의 사진 정보가 성공적으로 지도에 등록되었습니다!")
                st.rerun()

# --- 5. 지도 및 데이터 시각화 ---
df = load_data()

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📍 쓰레기 분포 지도")
    if not df.empty:
        center_lat = df["latitude"].mean()
        center_lon = df["longitude"].mean()
        m = folium.Map(location=[center_lat, center_lon], zoom_start=14)

        # 🌟 전체화면 버튼 지도에 추가
        Fullscreen(
            position='topright',
            title='전체화면 켜기',
            title_cancel='전체화면 끄기',
            force_separate_button=True
        ).add_to(m)

        marker_cluster = MarkerCluster(name="Trash Cluster").add_to(m)

        for idx, row in df.iterrows():
            popup_text = f"파일명: {row['filename']}<br>쓰레기 개수: {row['trash_count']}개"
            
            html_icon = f"""
            <div style="
                background-color: #E74C3C;
                color: white;
                border-radius: 50%;
                width: 30px;
                height: 30px;
                display: flex;
                justify-content: center;
                align-items: center;
                font-weight: bold;
                font-size: 14px;
                border: 2px solid white;
                box-shadow: 2px 2px 4px rgba(0,0,0,0.5);
            ">
                {row['trash_count']}
            </div>
            """

            folium.Marker(
                [row['latitude'], row['longitude']],
                popup=folium.Popup(popup_text, max_width=300),
                icon=folium.DivIcon(html=html_icon, icon_size=(30, 30), icon_anchor=(15, 15))
            ).add_to(marker_cluster)

        # 지도를 렌더링할 때 반응형으로 꽉 차게 설정 (가로폭 조정)
        st_folium(m, use_container_width=True, height=500, key="trash_map_display")
    else:
        st.info("현재 저장된 데이터가 없습니다. 사진을 업로드하고 분석 버튼을 눌러주세요.")

with col2:
    st.subheader("📋 누적 데이터 목록")
    if not df.empty:
        st.dataframe(df)
    else:
        st.write("데이터 없음")