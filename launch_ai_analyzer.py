import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# 📝 إعدادات الصفحة
st.set_page_config(page_title="AutoDiag AI Pro", layout="wide", page_icon="")

# 🎨 تخصيص CSS بسيط
st.markdown("""
<style>
    .main { background-color: #f5f5f5; }
    h1 { color: #004d40; text-align: center; }
    .stDataFrame { border-radius: 10px; }
    .metric-card { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center; }
</style>
""", unsafe_allow_html=True)

# 🧠 قاعدة البيانات (DTC Database)
DTC_DB = {
    'P0171': {'fr': 'Système trop pauvre (Banque 1)', 'ar': 'نظام فقير جداً (بنك 1)', 'action_fr': 'Vérifier fuites vide, MAF, pression carburant', 'action_ar': 'تفقد تسريب الهواء، حساس MAF، ضغط الوقود'},
    'P0300': {'fr': 'Ratés d\'allumage multiples', 'ar': 'احتراق عشوائي متعدد', 'action_fr': 'Tester bougies, bobines, compression', 'action_ar': 'افحص البوجيات، الكويلات، والكومبراس'},
    'P0420': {'fr': 'Efficacité catalyseur faible', 'ar': 'كفاءة الكاتاليزر منخفضة', 'action_fr': 'Vérifier capteurs O2, fuites échappement', 'action_ar': 'تفقد حساسات O2 وتسريب العادم'},
    'P0128': {'fr': 'Thermostat température basse', 'ar': 'ترموستات حرارة منخفضة', 'action_fr': 'Remplacer thermostat', 'action_ar': 'بدل الترموستات'},
    'P0101': {'fr': 'Débitmètre d\'air (MAF) performance', 'ar': 'حساس تدفق الهواء (MAF)', 'action_fr': 'Nettoyer ou remplacer MAF', 'action_ar': 'نظف أو بدل حساس MAF'},
}

# 🧠 تدريب النموذج (Cached)
@st.cache_resource
def train_model():
    data = {
        'DTC': ['P0171', 'P0300', 'P0420', 'P0128', 'P0101'],
        'RPM': [800, 2200, 1500, 750, 2800],
        'Load': [15, 45, 35, 10, 60],
        'Temp': [90, 95, 105, 70, 110],
        'Severity': ['Medium', 'High', 'Medium', 'Low', 'High']
    }
    df = pd.DataFrame(data)
    le = LabelEncoder()
    df['DTC_Enc'] = le.fit_transform(df['DTC'])
    X = df[['DTC_Enc', 'RPM', 'Load', 'Temp']]
    y = df['Severity']
    model = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42)
    model.fit(X, y)
    return model, le

def get_severity(model, le, dtc, rpm, load, temp):
    try:
        dtc_enc = le.transform([dtc])[0]
    except:
        dtc_enc = 0
    return model.predict([[dtc_enc, rpm, load, temp]])[0]

# 🌐 اختيار اللغة
lang = st.sidebar.selectbox("🌐 Langue / Language", ["Français", "العربية"])

# 🖥️ الواجهة الرئيسية
if lang == "العربية":
    st.title("🚗 محلل الأعطال الذكي للسيارات")
    st.info("ارفع ملف CSV من جهاز الفحص (Launch/Autel) ليحلل لك الأعطال.")
    upload_label = "📤 رفع ملف CSV"
    col1, col2, col3 = "عالي 🔴", "متوسط ", "منخفض 🟢"
    search_ph = "بحث عن كود العطل..."
else:
    st.title(" AI Car Diagnostic Analyzer")
    st.info("Upload CSV from your scanner (Launch/Autel) for AI analysis.")
    upload_label = "📤 Upload CSV Report"
    col1, col2, col3 = "High 🔴", "Medium 🟡", "Low 🟢"
    search_ph = "Search DTC code..."

# 📂 رفع الملف
uploaded_file = st.sidebar.file_uploader(upload_label, type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # تكييف الأعمدة تلقائياً
    col_mapping = {
        'DTC_Code': 'DTC_Code', 'Code': 'DTC_Code', 'DTC': 'DTC_Code',
        'RPM': 'RPM', 'Engine RPM': 'RPM',
        'Load_%': 'Load_%', 'Load': 'Load_%', 'Calculated Load': 'Load_%',
        'Temp_C': 'Temp_C', 'Coolant Temp': 'Temp_C', 'Temperature': 'Temp_C'
    }
    df.rename(columns={k: v for k, v in col_mapping.items() if k in df.columns}, inplace=True)

    if 'DTC_Code' in df.columns:
        st.subheader("📊 Raw Data Preview")
        st.dataframe(df.head())
        
        # 🔍 البحث
        search_term = st.sidebar.text_input(search_ph)
        if search_term:
            df = df[df['DTC_Code'].astype(str).str.contains(search_term, case=False)]

        # 🤖 تحليل AI
        model, le = train_model()
        results = []
        for _, row in df.iterrows():
            dtc = str(row['DTC_Code']).strip()
            sev = get_severity(model, le, dtc, row.get('RPM', 0), row.get('Load_%', 0), row.get('Temp_C', 0))
            
            info = DTC_DB.get(dtc, {'fr': 'Unknown DTC', 'ar': 'كود غير معروف', 'action_fr': 'Check manual', 'action_ar': 'راجع الدليل'})
            
            results.append({
                'DTC': dtc,
                'Description': info['ar'] if lang == "العربية" else info['fr'],
                'Severity': sev,
                'Action': info['action_ar'] if lang == "العربية" else info['action_fr'],
                'RPM': row.get('RPM', 0),
                'Load': row.get('Load_%', 0)
            })

        res_df = pd.DataFrame(results)
        
        # 📈 لوحة القيادة (Dashboard)
        c1, c2, c3 = st.columns(3)
        c1.metric(col1, len(res_df[res_df['Severity'] == 'High']))
        c2.metric(col2, len(res_df[res_df['Severity'] == 'Medium']))
        c3.metric(col3, len(res_df[res_df['Severity'] == 'Low']))
        
        #  الرسم البياني
        if len(res_df) > 0:
            fig = px.scatter(res_df, x='RPM', y='Load', color='Severity', 
                             title="RPM vs Load Analysis", 
                             labels={'RPM': 'Engine Speed', 'Load': 'Load %'})
            st.plotly_chart(fig, use_container_width=True)

        # 📋 جدول النتائج
        st.subheader("🔍 Analysis Results")
        st.dataframe(res_df)
        
        # 📥 تحميل
        csv = res_df.to_csv(index=False).encode('utf-8-sig')
        st.download_button("📥 Download Report (CSV)", csv, "report_ai.csv", "text/csv")

    else:
        st.error(f"❌ Colonnes manquantes. Attendu: DTC_Code, RPM, Load_%, Temp_C")
        st.code("Colonnes trouvées: " + ", ".join(df.columns))

else:
    st.info("⬅️ Commencez par uploader un fichier CSV dans le menu latéral.")
