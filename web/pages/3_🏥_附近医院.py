import streamlit as st
import requests
import logging
import pandas as pd
import pydeck as pdk
from geopy.distance import geodesic
BAIDU_API_AK = "kfLxkGbOE95apSymbmlTBLRjIt4Jsd7U"


st.set_page_config(
    page_title="附近医院",
    page_icon="🏥",
)


# --------------------- Functions ---------------------
def get_client_ip():
    """获取客户端IP（优先国内服务）"""
    services = [
        {'url': 'https://www.taobao.com/help/getip.php', 'pattern': 'ip', 'type': 'text'},
        {'url': 'https://ip.360.cn/IPShare/info', 'key': 'ip'},
        {'url': 'https://ipinfo.io/json', 'key': 'ip'},
    ]
    for service in services:
        try:
            response = requests.get(service['url'], timeout=3)
            response.raise_for_status()
            if 'type' in service and service['type'] == 'text':
                ip = response.text.strip().split('=')[-1].strip("'")
                if ip.count('.') == 3:
                    return ip
            else:
                data = response.json()
                if 'key' in service:
                    return data.get(service['key'], '').split(',')[0].strip()    
        except Exception as e:
            logging.warning(f"Service {service['url']} failed: {str(e)}")
            continue   
    logging.error("All IP services unavailable")
    return None


def get_location(ip):
    """使用百度地图API进行高精度IP定位"""
    try:
        url = f"https://api.map.baidu.com/location/ip?ip={ip}&ak={BAIDU_API_AK}&coor=bd09ll"
        response = requests.get(url, timeout=3)
        data = response.json()
        if data['status'] == 0:
            return (
                data['content']['point']['y'], 
                data['content']['point']['x'],
                data['content'].get('accuracy', '城市级定位')
            )
        return None, None, "定位失败"
    except Exception as e:
        logging.error(f"Location request failed: {str(e)}")
        return None, None, "服务异常"


def geocode_address(address):
    """地址地理编码（使用百度地图API）"""
    try:
        url = f"http://api.map.baidu.com/geocoding/v3/?address={address}&output=json&ak={BAIDU_API_AK}"
        response = requests.get(url, timeout=5)
        data = response.json()
        if data['status'] == 0:
            location = data['result']['location']
            return location['lat'], location['lng']
        return None, None
    except Exception as e:
        logging.error(f"Geocoding failed: {str(e)}")
        return None, None


def get_hospitals(lat, lon, radius=20000):
    """通过Overpass API获取医院数据"""
    overpass_url = "https://overpass-api.de/api/interpreter"
    query = f"""
    [out:json];
    node["amenity"="hospital"](around:{radius},{lat},{lon});
    out body;
    """
    try:
        response = requests.post(overpass_url, data=query, timeout=10)
        response.raise_for_status()
        data = response.json()
        hospitals = []
        for element in data['elements']:
            hospitals.append({
                'name': element['tags'].get('name', '未知医院'),
                'lat': element['lat'],
                'lon': element['lon'],
            })
        return pd.DataFrame(hospitals)
    except Exception as e:
        st.error(f"获取医院数据失败: {str(e)}")
        return pd.DataFrame()


def calculate_distance(row, user_loc):
    """计算医院与用户的距离（千米）"""
    hospital_loc = (row['lat'], row['lon'])
    return geodesic(user_loc, hospital_loc).km


# --------------------- Page Layout ---------------------
st.title("附近医院")
st.write("基于精准定位的医疗资源查询系统")

# Sidebar Settings
with st.sidebar:
    st.header("⚙️ 设置面板")
    user_loc = None
    accuracy = ""
    # Manual positioning toggle
    mode = st.radio("模式",
                   ["自动模式", "手动模式"],
                   index=0,
                   help="选择模式",
                   horizontal=True)  # 水平排列选项
    # Search parameters
    search_radius = st.slider("搜索半径 (km)", 1, 50, 20)
    min_distance = st.slider("显示最近的医院数量", 1, 50, 25)
    st.markdown("------")
    if mode == "手动模式":
        address = st.text_input("请输入详细地址", 
                              placeholder="例如:武汉市洪山区华中科技大学",)
        if st.button("📍 地址地理编码"):
            with st.spinner('正在进行地址地理编码...'):
                lat, lon = geocode_address(address)
                if lat and lon:
                    user_loc = (lat, lon)
                    accuracy = "手动定位"
                    st.success("地址地理编码成功！")
                else:
                    st.error("地理编码失败，请检查输入格式")
    else:
        # Automatic positioning
        with st.spinner('正在获取位置信息...'):
            ip = get_client_ip()
            if ip:
                lat, lon, accuracy = get_location(ip)
                user_loc = (lat, lon) if lat and lon else None
            else:
                user_loc = None
                accuracy = "定位失败"
    # Display location information
    if user_loc:
        try:
            lat = float(user_loc[0])
            lon = float(user_loc[1])
            st.subheader("📍 位置信息")
            st.write(f"""
            - 纬度: `{lat:.6f}`
            - 经度: `{lon:.6f}`
            - 精度: `{accuracy}`
            """)
        except (TypeError, ValueError) as e:
            st.error(f"坐标格式错误: {str(e)}")
            st.stop()
    else:
        st.stop()


# --------------------- Data Fetching ---------------------
with st.spinner(f'正在搜索 {search_radius} 公里范围内的医院...'):
    hospitals_df = get_hospitals(*user_loc, search_radius*1000)
    if hospitals_df.empty:
        st.warning("⚠️ 当前范围内未找到医疗机构")
        st.stop()
    # Calculate distances
    hospitals_df['distance'] = hospitals_df.apply(
        lambda row: calculate_distance(row, user_loc), 
        axis=1
    )
    hospitals_df = hospitals_df.sort_values('distance').head(min_distance)


# --------------------- Content Display ---------------------
if mode == "手动模式":
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("医疗机构数量", len(hospitals_df), border=True)
    with col2:
        st.metric("最近距离", f"{hospitals_df['distance'].min():.2f} 公里", border=True)
    with col3:
        st.metric("最远距离", f"{hospitals_df['distance'].max():.2f} 公里", border=True)
    # Map visualization
    st.subheader("🗺️ 医院地图")
    map_layers = [
        pdk.Layer(
            "ScatterplotLayer",
            data=pd.DataFrame([{'lat': user_loc[0], 'lon': user_loc[1]}]),
            get_position='[lon, lat]',
            get_color='[255, 0, 0, 200]',
            get_radius=200,
            pickable=True
        ),
        pdk.Layer(
            "ScatterplotLayer",
            data=hospitals_df,
            get_position='[lon, lat]',
            get_color='[0, 128, 255, 200]',
            get_radius=150,
            pickable=True
        )
    ]
    st.pydeck_chart(pdk.Deck(
        map_style='road',
        initial_view_state=pdk.ViewState(
            latitude=user_loc[0],
            longitude=user_loc[1],
            zoom=12,
            pitch=50
        ),
        layers=map_layers,
        tooltip={
            'html': '<b>{name}</b><br/>距离: {distance} 公里',
            'style': {'color': 'white'}
        }
    ))
    # Data table
    st.subheader("📋 医院详情")
    display_df = hospitals_df[['name', 'distance']].rename(
        columns={'name': '医院名称', 'distance': '距离 (公里)'}
        )
    display_df['距离 (公里)'] = display_df['距离 (公里)'].round(2)
    st.dataframe(
        display_df,
        height=400,
        use_container_width=True,
        column_config={
            "医院名称": st.column_config.TextColumn(
                width="medium",
                help="医疗机构名称"
            ),
            "距离 (公里)": st.column_config.NumberColumn(
                format="%.2f 公里",
                help="距离您的位置"
            )
        },
        hide_index=True
    )
else:
    # Automatic mode shows table in expander
    with st.expander(f"🏥 附近医院（共找到 {len(hospitals_df)} 家）", expanded=True):
        display_df = hospitals_df[['name', 'distance']].rename(
            columns={'name': '医院名称', 'distance': '距离 (公里)'}
        )
        display_df['距离 (公里)'] = display_df['距离 (公里)'].round(2)
        st.dataframe(
            display_df,
            height=400,
            use_container_width=True,
            column_config={
                "医院名称": st.column_config.TextColumn(
                    width="medium",
                    help="医疗机构名称"
                ),
                "距离 (公里)": st.column_config.NumberColumn(
                    format="%.2f 公里",
                    help="距离您的位置"
                )
            },
            hide_index=True
        )
        st.caption(f"""
        总结: 
        - 平均距离: {display_df['距离 (公里)'].mean():.2f} 公里
        - 最近医院: {display_df['距离 (公里)'].min():.2f} 公里
        - 覆盖半径: {display_df['距离 (公里)'].max():.2f} 公里
        """)
# Refresh button
if st.button("🔄 刷新数据"):
    st.rerun()
