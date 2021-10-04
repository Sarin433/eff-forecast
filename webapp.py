### Import Library ###
import base64

import pandas as pd
import streamlit as st
import numpy as np

from fbprophet import Prophet
from fbprophet.plot import plot_plotly, plot_components_plotly

from plotly import graph_objs as go

### Setup file location ###
maeNgat = 'เขื่อนแม่งัดสมบูรณ์ชล'
maeKuang = 'เขื่อนแม่กวงอุดมธารา'
url_MaeNgat = 'MaeNgat_2.csv'
url_MaeKuang = 'MaeKuang_2.csv'

dam = (maeNgat, maeKuang)

st.set_page_config(page_title ="Forecast App",
                    initial_sidebar_state="collapsed",
                    page_icon="🔮")
st.header("โปรแกรมพยากรณ์ปริมาณน้ำที่ถูกระบายจากเขื่อนแม่กวงอุดมธาราและเขื่อนแม่งัดสมบูรณ์ชล")
select_dam = st.selectbox('โปรดเลือกอ่างเก็บน้ำ', dam)

### Load Data ###
@st.cache
def load_data(r_data):
    if r_data == maeKuang:
        r_data = pd.read_csv(url_MaeKuang)
        r_data = r_data[['date', 'outflow_m3']]
        r_data['date'] = pd.to_datetime(r_data['date'])
        return r_data

    if r_data == maeNgat:
        r_data = pd.read_csv(url_MaeNgat)
        r_data = r_data[['date', 'outflow_m3']]
        r_data['date'] = pd.to_datetime(r_data['date'])
        r_data.fillna(min(r_data['outflow_m3']))
        return r_data

# data_load_state = st.text('กำลังดาวน์โหลดข้อมูล')
df = load_data(select_dam)
# data_load_state.text('ดาวน์โหลดข้อมูลเสร็จสิ้น')

### If you want to see the raw data ###
### Remove code_raw_data and '''........''' ###
code_raw_data = '''
### Show Raw Data ###
st.subheader('Daily Raw data')
st.write(df)

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['date'], y=df['outflow_m3'],
                             name='Effluent of the dam'))
    fig.layout.update(title_text='Time Series Data', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

df_monthly = df.set_index('date').groupby(pd.Grouper(freq='M'))['outflow_m3'].sum().reset_index()
st.write(df_monthly)

def plot_raw_monthly_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_monthly['date'], y=df_monthly['outflow_m3'],
                             name='Effluent of the dam'))
    fig.layout.update(title_text='Time Series Data', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_monthly_data()
'''

### Prepare for Fitting Model ###
df_train = df[['date', 'outflow_m3']]
df_train = df_train.rename(columns={'date': 'ds', 'outflow_m3': 'y'})

model = Prophet(changepoint_prior_scale=0.001)
model.fit(df_train)
future = model.make_future_dataframe(periods=730, freq='D')
forecast = model.predict(future)
forecast["yhat"] = np.where(forecast["yhat"]<0,min(df_train['y']),forecast["yhat"])

### Daily Forecast ###
#st.subheader('การพยากรณ์น้ำระบายรายวัน')
daily_forecast = forecast
daily_forecast = daily_forecast.rename(columns={'ds':'วันที่', 'yhat':'ค่าพยากรณ์', 'yhat_lower':'ค่าต่ำสุด', 'yhat_upper':'ค่าสูงสุด'})
# st.write('ตารางแสดงผลการพยากรณ์น้ำระบายรายวัน')
# st.write(daily_forecast[['วันที่', 'ค่าพยากรณ์', 'ค่าต่ำสุด', 'ค่าสูงสุด']].tail(730))

# st.write('กราฟแสดงผลของการพยากรณ์')
fig1 = plot_plotly(model, forecast)
# st.plotly_chart(fig1)

fig2 = plot_components_plotly(model, forecast)


### Monthly Operation ###
dt = forecast[['ds','yhat']]
dt = dt.rename(columns={'ds':'วันที่', 'yhat':'ปริมาณน้ำที่ถูกระบาย (ลูกบาศก์เมตร)'})
dt = dt.set_index('วันที่').groupby(pd.Grouper(freq='M'))['ปริมาณน้ำที่ถูกระบาย (ลูกบาศก์เมตร)'].sum().reset_index()

def monthly_plot():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dt['วันที่'], y=dt['ปริมาณน้ำที่ถูกระบาย (ลูกบาศก์เมตร)'],
                             name='Effluent of the dam'))
    fig.layout.update(xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

### Web Template ###
st.subheader('กราฟแสดงการพยากรณ์น้ำออกรายเดือน')
monthly_plot()
st.subheader('กราฟแสดงส่วนประกอบต่างๆ (เทรน, การเปลี่ยนแปลงรายปี, การเปลี่ยนแปลงรายสัปดาห์)')
st.write(fig2)
st.subheader('ตารางแสดงผลของการพยากรณ์น้ำออกรายเดือน')
st.write(dt.tail(24))
st.subheader('ดาวน์โหลดตารางข้อมูลพยากรณ์น้ำออกรายเดือน')
col1, col2 = st.columns(2)
with col1:
    if st.button('ตารางน้ำที่ถูกระบายรายเดือน(.csv)'):
        with st.spinner('Exporting...'):
            export_forecast = dt.rename(columns={'วันที่':'Date', 'ปริมาณน้ำที่ถูกระบาย (ลูกบาศก์เมตร)':'Effluents (m3)'})
            st.write(export_forecast.tail(24))
            export_forecast = export_forecast.to_csv()
            b64 = base64.b64encode(export_forecast.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (click Download > ตั้งชื่อไฟล์ **forecast.csv**)'
            st.markdown(href, unsafe_allow_html=True)
with col2:
    if st.button('ตารางน้ำที่ถูกระบายรายวัน(.csv)'):
        with st.spinner('Exporting...'):
            ex_dt = daily_forecast[['วันที่', 'ค่าพยากรณ์']]
            export_forecast = ex_dt.rename(columns={'วันที่':'Date', 'ค่าพยากรณ์':'Effluents (m3)'})
            st.write(export_forecast.tail(730))
            export_forecast = export_forecast.to_csv()
            b64 = base64.b64encode(export_forecast.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (click Download > ตั้งชื่อไฟล์ **forecast.csv**)'
            st.markdown(href, unsafe_allow_html=True)
#===============================================================================================================================
