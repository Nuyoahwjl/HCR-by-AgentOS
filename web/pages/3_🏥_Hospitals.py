import streamlit as st
import requests
import logging
import pandas as pd
import folium
from streamlit_folium import st_folium
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
import time
st.set_page_config(
    page_title="Hospitals",
    page_icon="🏥",
)
# --------------------- Functions ---------------------
def get_client_ip():
    """获取客户端IP"""
    services = [
        {"url": "https://api.ip.sb/jsonip", "key": "ip"},
        {"url": "https://ip.user-agent.info/ip", "key": "ip"},
        {"url": "https://httpbin.org/ip", "key": "origin"},
    ]
    for svc in services:
        try:
            resp = requests.get(svc["url"], timeout=5)
            resp.raise_for_status()
            data = resp.json()
            ip = data.get(svc["key"], "")
            if ip and ip.count(".") == 3:
                return ip
        except Exception as e:
            logging.warning(f"IP service {svc['url']} failed: {e}")
            continue
    return None
def ip_to_location(ip):
    """通过 ip-api.com 免费定位（无需密钥）"""
    try:
        url = f"http://ip-api.com/json/{ip}?fields=status,lat,lon,city,regionName,country"
        resp = requests.get(url, timeout=5)
        data = resp.json()
        if data.get("status") == "success":
            return (
                data["lat"],
                data["lon"],
                f"{data.get('city', '')}, {data.get('regionName', '')}",
            )
    except Exception as e:
        logging.error(f"ip-api location failed: {e}")
    return None, None, "Location failed"
def geocode_address(address):
    """使用 Nominatim（OpenStreetMap）地理编码，完全免费"""
    try:
        geolocator = Nominatim(user_agent="hcr-health-check-recommender", timeout=10)
        location = geolocator.geocode(address)
        if location:
            return location.latitude, location.longitude
    except Exception as e:
        logging.error(f"Nominatim geocoding failed: {e}")
    return None, None
def get_hospitals(lat, lon, radius=20000):
    """通过 Overpass API（OpenStreetMap）获取医院数据，完全免费"""
    overpass_url = "https://overpass-api.de/api/interpreter"
    query = f"""
    [out:json];
    node["amenity"="hospital"](around:{radius},{lat},{lon});
    out body;
    """
    try:
        resp = requests.post(overpass_url, data=query, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        hospitals = []
        for el in data.get("elements", []):
            tags = el.get("tags", {})
            hospitals.append({
                "name": tags.get("name", "未知医院"),
                "lat": el["lat"],
                "lon": el["lon"],
                "phone": tags.get("phone", tags.get("contact:phone", "")),
                "addr": tags.get("addr:full", tags.get("addr:street", "")),
            })
        return pd.DataFrame(hospitals)
    except Exception as e:
        st.error(f"Failed to get hospital data: {e}")
        return pd.DataFrame()
def calculate_distance(row, user_loc):
    """计算医院与用户距离（千米）"""
    return geodesic(user_loc, (row["lat"], row["lon"])).km
def build_map(user_loc, hospitals_df):
    """使用 Folium 构建 OpenStreetMap 地图"""
    m = folium.Map(location=user_loc, zoom_start=12, tiles="OpenStreetMap")
    # 用户位置标记
    folium.Marker(
        location=user_loc,
        popup=folium.Popup("📍 Your Location", max_width=200),
        icon=folium.Icon(color="red", icon="user", prefix="fa"),
    ).add_to(m)
    # 医院标记
    for _, row in hospitals_df.iterrows():
        popup_html = f"<b>{row['name']}</b><br>Distance: {row['distance']:.2f} km"
        if row.get("phone"):
            popup_html += f"<br>Tel: {row['phone']}"
        if row.get("addr"):
            popup_html += f"<br>{row['addr']}"
        folium.Marker(
            location=[row["lat"], row["lon"]],
            popup=folium.Popup(popup_html, max_width=250),
            icon=folium.Icon(color="blue", icon="plus-sign"),
        ).add_to(m)
    # 搜索半径圆
    if "search_radius" in st.session_state:
        folium.Circle(
            location=user_loc,
            radius=st.session_state["search_radius"] * 1000,
            color="blue",
            fill=True,
            fill_opacity=0.05,
        ).add_to(m)
    return m
# --------------------- Page Layout ---------------------
st.title("Nearby Hospital")
st.write("A Medical Resource Query System Based on OpenStreetMap")
with st.sidebar:
    st.header("⚙️ Settings Panel")
    user_loc = None
    accuracy = ""
    mode = st.radio(
        "Mode",
        ["Auto_Mode", "Manual_Mode"],
        index=0,
        help="Select Mode",
        horizontal=True,
    )
    search_radius = st.slider("Search Radius (km)", 1, 50, 20)
    min_distance = st.slider("Number of Nearest Hospitals to Display", 1, 50, 25)
    st.session_state["search_radius"] = search_radius
    st.markdown("------")
    if mode == "Manual_Mode":
        address = st.text_input(
            "Please enter a detailed address",
            placeholder="eg: 武汉市洪山区华中科技大学",
        )
        if st.button("📍 Geocode Address"):
            with st.spinner("Geocoding address via OpenStreetMap..."):
                lat, lon = geocode_address(address)
                if lat and lon:
                    user_loc = (lat, lon)
                    accuracy = "Manual positioning (Nominatim)"
                    st.success("Address geocoding successful!")
                else:
                    st.error("Geocoding failed, please check input format")
    else:
        with st.spinner("Fetching location information..."):
            ip = get_client_ip()
            if ip:
                lat, lon, accuracy = ip_to_location(ip)
                if lat and lon:
                    user_loc = (lat, lon)
                else:
                    user_loc = None
                    accuracy = "Location failed"
            else:
                user_loc = None
                accuracy = "Location failed"
    if user_loc:
        try:
            lat = float(user_loc[0])
            lon = float(user_loc[1])
            st.subheader("📍 Location Information")
            st.write(f"""
            - Latitude: `{lat:.6f}`
            - Longitude: `{lon:.6f}`
            - Accuracy: `{accuracy}`
            """)
        except (TypeError, ValueError) as e:
            st.error(f"Coordinate format error: {e}")
            st.stop()
    else:
        st.warning("Unable to locate automatically. Please switch to Manual_Mode.")
        st.stop()
# --------------------- Data Fetching ---------------------
with st.spinner(f"Searching for hospitals within {search_radius} km radius..."):
    hospitals_df = get_hospitals(*user_loc, search_radius * 1000)
    if hospitals_df.empty:
        st.warning("⚠️ No medical institutions found within current range")
        st.stop()
    hospitals_df["distance"] = hospitals_df.apply(
        lambda row: calculate_distance(row, user_loc), axis=1
    )
    hospitals_df = hospitals_df.sort_values("distance").head(min_distance)
# --------------------- Content Display ---------------------
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Medical Institutions", len(hospitals_df), border=True)
with col2:
    st.metric("Nearest Distance", f"{hospitals_df['distance'].min():.2f} km", border=True)
with col3:
    st.metric("Farthest Distance", f"{hospitals_df['distance'].max():.2f} km", border=True)
# Map
st.subheader("🗺️ Hospital Map")
map_obj = build_map(user_loc, hospitals_df)
st_folium(map_obj, width=None, height=500, returned_objects=[])
# Data table
st.subheader("📋 Hospital Details")
display_df = hospitals_df[["name", "distance"]].rename(
    columns={"name": "Hospital Name", "distance": "Distance (km)"}
)
display_df["Distance (km)"] = display_df["Distance (km)"].round(2)
st.dataframe(
    display_df,
    height=400,
    width='stretch',
    column_config={
        "Hospital Name": st.column_config.TextColumn(
            width="medium",
            help="Medical institution name",
        ),
        "Distance (km)": st.column_config.NumberColumn(
            format="%.2f km",
            help="Distance from your location",
        ),
    },
    hide_index=True,
)
st.caption(
    f"Average: {display_df['Distance (km)'].mean():.2f} km | "
    f"Nearest: {display_df['Distance (km)'].min():.2f} km | "
    f"Coverage: {display_df['Distance (km)'].max():.2f} km"
)
if st.button("🔄 Refresh Data"):
    st.rerun()