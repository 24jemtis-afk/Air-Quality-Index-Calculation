from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title='Air Quality Monitoring', page_icon='🌍',layout='wide')
ROOT=Path(__file__).resolve().parent
POLL=['PM2.5','PM10','NO2','SO2','CO','O3']

@st.cache_data
def load_data():
    p=ROOT/'air_quality_cleaned.csv'
    return pd.read_csv(p) if p.exists() else None

@st.cache_resource
def load_model():
    p=ROOT/'aqi_model.pkl'
    if not p.exists(): return None
    with p.open('rb') as f:return pickle.load(f)

def category(aqi):
    if aqi<=50:return 'Good'
    if aqi<=100:return 'Satisfactory'
    if aqi<=200:return 'Moderate'
    if aqi<=300:return 'Poor'
    if aqi<=400:return 'Very Poor'
    return 'Severe'

def profile(bundle,city,season):
    key=f'{city}||{season}'
    return (bundle.get('group_profiles',{}).get(key)
            or bundle.get('city_profiles',{}).get(city)
            or bundle.get('season_profiles',{}).get(season)
            or bundle.get('pollutant_medians',{}))

def prediction(bundle,inputs,city,season):
    row=pd.DataFrame([{**inputs,'City':city,'Season':season}])
    base=float(bundle['pipeline'].predict(row)[0])
    adjusted=base+float(bundle.get('city_offsets',{}).get(city,0))+float(bundle.get('season_offsets',{}).get(season,0))
    return max(0.0,adjusted),row

try:
    df=load_data()
    bundle=load_model()
except Exception as e:
    st.error(f'Loading error: {e}')
    st.stop()

st.sidebar.title('🌍 Air Quality Monitoring')
page=st.sidebar.radio('Navigation',['Dashboard','Data Analysis','AQI Prediction','Model Comparison','About Project'])
st.sidebar.caption('Statistical Inference Laboratory')

if page=='Dashboard':
    st.title('🌍 Air Quality Monitoring Dashboard')
    if df is None or 'AQI' not in df:
        st.warning('Place air_quality_cleaned.csv next to app.py')
    else:
        aqi=pd.to_numeric(df['AQI'],errors='coerce')
        m=st.columns(4)
        for col,title,value in zip(m,['Records','Average AQI','Maximum AQI','Cities'],[len(df),round(aqi.mean(),2),round(aqi.max(),2),df['City'].nunique() if 'City' in df else 0]):col.metric(title,value)
        left,right=st.columns(2)
        with left:
            fig,ax=plt.subplots();sns.histplot(aqi.dropna(),bins=30,kde=True,ax=ax);ax.set_title('AQI Distribution');st.pyplot(fig);plt.close(fig)
        with right:
            st.subheader('Air Quality Categories')
            st.bar_chart(aqi.dropna().map(category).value_counts())
        st.dataframe(df.head(30),use_container_width=True)

elif page=='Data Analysis':
    st.title('📊 Air Quality Analysis')
    if df is None: st.warning('CSV missing');st.stop()
    filtered=df
    if 'City' in df:
        cities=sorted(df['City'].dropna().astype(str).unique().tolist())
        city=st.selectbox('City',['All Cities']+cities)
        if city!='All Cities':filtered=df[df['City'].astype(str)==city]
    present=[c for c in POLL if c in filtered]
    st.dataframe(filtered.describe(include='all'),use_container_width=True)
    if present:
        col=st.selectbox('Pollutant',present)
        fig,ax=plt.subplots();sns.histplot(pd.to_numeric(filtered[col],errors='coerce').dropna(),kde=True,ax=ax);st.pyplot(fig);plt.close(fig)
        correlation=filtered[[c for c in POLL+['AQI'] if c in filtered]].apply(pd.to_numeric,errors='coerce').corr()
        fig,ax=plt.subplots(figsize=(8,5));sns.heatmap(correlation,annot=True,cmap='coolwarm',ax=ax);st.pyplot(fig);plt.close(fig)
    if 'Date' in filtered and 'AQI' in filtered:
        trends=filtered.copy();trends['Date']=pd.to_datetime(trends['Date'],errors='coerce');trends['AQI']=pd.to_numeric(trends['AQI'],errors='coerce')
        st.line_chart(trends.groupby('Date')['AQI'].mean().sort_index())

elif page=='AQI Prediction':
    st.title('🤖 City and Season Aware AQI Prediction')
    if bundle is None:
        st.error('aqi_model.pkl not found. Run the last cells of the UPDATED notebook first.');st.stop()
    if not bundle.get('group_profiles'):
        st.error('Old PKL detected. Run the updated notebook including the export cell to regenerate aqi_model.pkl.');st.stop()
    cities=bundle.get('cities',[]);seasons=bundle.get('seasons',[])
    if not cities or not seasons: st.error('No city or season found in model.');st.stop()
    c1,c2=st.columns(2)
    city=c1.selectbox('Select City',cities,key='selected_city')
    season=c2.selectbox('Select Season',seasons,key='selected_season')
    choice=(city,season)
    # Auto-update input values if city OR season changed; inputs still editable by hand.
    if st.session_state.get('last_city_season')!=choice:
        recommended=profile(bundle,city,season)
        for col in POLL:
            st.session_state['gas_'+col]=max(0.0,float(recommended.get(col,bundle['pollutant_medians'].get(col,0))))
        st.session_state['last_city_season']=choice
    st.caption('Gas inputs are automatically filled with typical values for the selected city and season. You may edit them manually. Changing either dropdown refreshes these defaults.')
    cols=st.columns(3);measurements={}
    for i,col in enumerate(POLL):
        with cols[i%3]:
            measurements[col]=st.number_input(col,min_value=0.0,step=0.1,format='%.2f',key='gas_'+col)
    aqi,row=prediction(bundle,measurements,city,season)
    st.metric('Predicted AQI',f'{aqi:.2f}')
    st.subheader('AQI Category: '+category(aqi))
    st.caption('Predictions use saved ML model + estimated city/season residual calibration. These are not live pollution measurements.')
    with st.expander('Input details'):
        st.dataframe(row,use_container_width=True)
        st.write('Regression model:',bundle.get('name','Unknown'))
        st.write('City adjustment:',round(bundle.get('city_offsets',{}).get(city,0),3))
        st.write('Season adjustment:',round(bundle.get('season_offsets',{}).get(season,0),3))

elif page=='Model Comparison':
    st.title('📈 Model Comparison')
    if bundle is None:st.warning('Run notebook to export the model first.');st.stop()
    st.write('Best regression model:',bundle.get('name','Unknown'))
    tab=bundle.get('table')
    if isinstance(tab,pd.DataFrame) and not tab.empty:
        st.dataframe(tab,use_container_width=True)
        if 'Model' in tab and 'CV RMSE' in tab:st.bar_chart(tab.set_index('Model')['CV RMSE'])
    st.write('Best classifier:',bundle.get('classifier_name','Unknown'))
    st.caption('See Practical 8 of your notebook for classification comparison scores.')

else:
    st.title('📚 About This Project')
    st.write('Statistical Inference project: air-quality analysis using pollutant concentrations, city and season.')
    st.write('Regression predicts numerical AQI; classification evaluates AQI categories.')
    st.write('The notebook keeps the same taught algorithms as the friend-style practicals.')
    st.info('Changing city or season updates typical gas levels from the dataset and uses city/season residual corrections. Predictions are estimates, not live sensor readings.')
