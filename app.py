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

def count_trash_in_image(pil_image):
    results = model(pil_image)
    return len(results[0].boxes)

def load_data():
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
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
    uploaded_files = st.file_uploader("사진 선택", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    st.markdown("---")
    if st.button("전체 데이터 초기화"):
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
    st.subheader("쓰레기 분포 지도")
    if not df.empty:
        m = folium.Map(location=[df["latitude"].mean(), df["longitude"].mean()], zoom_start=13)
        Fullscreen().add_to(m)

        # 🌟 개선: 구글맵 스타일의 커스텀 클러스터 생성 함수
        icon_create_function = """
        function(cluster){
            var markers = cluster.getAllChildMarkers();
            var sum = 0;

            for(var i=0; i<markers.length; i++){
                sum += parseInt(markers[i].options.title) || 0;
            }

            var color;
            if(sum <= 5)
                color = "#2ECC71";
            else if(sum <= 15)
                color = "#F1C40F";
            else if(sum <= 30)
                color = "#E67E22";
            else
                color = "#E74C3C";

            return L.divIcon({
                html:
                `<div style="
                    width:52px;
                    height:52px;
                    border-radius:50%;
                    background:${color};
                    border:5px solid white;
                    box-shadow:
                        0 6px 18px rgba(0,0,0,.35),
                        inset 0 2px 6px rgba(255,255,255,.3);
                    display:flex;
                    justify-content:center;
                    align-items:center;
                    color:white;
                    font-weight:bold;
                    font-size:19px;
                ">
                    ${sum}
                </div>`,
                className:"",
                iconSize:[52, 52]
            });
        }
        """

        marker_cluster = MarkerCluster(
            name="Trash Cluster",
            icon_create_function=icon_create_function,
            options={'maxClusterRadius': 40}
        ).add_to(m)

        for _, row in df.iterrows():
            trash_count = int(row['trash_count'])

            # 쓰레기 개수에 따른 색상 지정
            if trash_count <= 2:
                color = "#2ECC71"
            elif trash_count <= 5:
                color = "#F1C40F"
            elif trash_count <= 10:
                color = "#E67E22"
            else:
                color = "#E74C3C"

            # 🌟 개선: 구글맵 스타일의 개별 마커 아이콘 HTML
            html_icon = f"""
            <div style="
                width:38px;
                height:38px;
                border-radius:50%;
                background:{color};
                border:4px solid white;
                box-shadow:
                    0 4px 12px rgba(0,0,0,.35),
                    inset 0 2px 4px rgba(255,255,255,.35);
                display:flex;
                justify-content:center;
                align-items:center;
                color:white;
                font-weight:700;
                font-size:16px;
                transition:0.2s;
            ">
                {trash_count}
            </div>
            """

            # 🌟 개선: 깔끔한 팝업 스타일 및 피드백 주신 마커 옵션 적용
            folium.Marker(
                location=[row["latitude"], row["longitude"]],
                popup=folium.Popup(
                    f"""
                    <b>{row['filename']}</b><br>
                    🗑️ 쓰레기 <b>{trash_count}개</b>
                    """,
                    max_width=250
                ),
                icon=folium.DivIcon(
                    html=html_icon,
                    icon_size=(38, 38),
                    icon_anchor=(19, 19)
                ),
                title=str(trash_count)
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
