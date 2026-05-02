import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import re

# Configuration
st.set_page_config(
    page_title="AutoDiag Pro",
    layout="wide",
    page_icon="🚗"
)

# Gestion langue
if 'language' not in st.session_state:
    st.session_state.language = 'fr'

# Traductions
TRANSLATIONS = {
    'fr': {
        'title': '🚗 AutoDiag Pro - Diagnostic Intelligent',
        'sidebar_title': '⚙️ Options',
        'import_data': '📁 Importer Données',
        'method': 'Méthode:',
        'csv': 'CSV',
        'text_ocr': 'Texte/OCR',
        'manual': 'Saisie Manuelle',
        'choose_csv': 'Choisir fichier CSV',
        'search_dtc': '🔍 Recherche DTC',
        'dtc_code': 'Code DTC',
        'placeholder_dtc': 'Ex: P0171',
        'upload': 'Upload',
        'paste_text': 'Coller le texte du scanner',
        'analyze': 'Analyser',
        'add': 'Ajouter',
        'rpm': 'RPM',
        'load': 'Load %',
        'temp': 'Température °C',
        'file_loaded': '✅ Fichier chargé avec succès!',
        'columns_detected': 'Colonnes détectées:',
        'dtc_col_not_found': 'Colonne DTC non trouvée!',
        'select_dtc_col': 'Sélectionner colonne DTC',
        'data_preview': '📊 Aperçu des données',
        'analysis_results': '📊 Analyse des Résultats',
        'high': '🔴 Élevée',
        'medium': '🟡 Moyenne',
        'low': '🟢 Faible',
        'details': '🔧 Détails des défauts',
        'severity': 'Sévérité',
        'price': 'Prix',
        'cause': 'Cause',
        'solution': 'Solution',
        'download': '📥 Télécharger Rapport',
        'no_dtc_found': '⚠️ Aucun code DTC reconnu',
        'error': '❌ Erreur:',
        'footer': '© 2026 AutoDiag Pro - Diagnostic Automobile',
        'distribution': 'Répartition des défauts'
    },
    'ar': {
        'title': '🚗 أوتو دياج برو - التشخيص الذكي',
        'sidebar_title': '⚙️ الخيارات',
        'import_data': '📁 استيراد البيانات',
        'method': 'الطريقة:',
        'csv': 'CSV',
        'text_ocr': 'نص/مسح ضوئي',
        'manual': 'إدخال يدوي',
        'choose_csv': 'اختر ملف CSV',
        'search_dtc': '🔍 البحث عن عطل',
        'dtc_code': 'كود العطل',
        'placeholder_dtc': 'مثال: P0171',
        'upload': 'رفع',
        'paste_text': 'الصق النص من جهاز الفحص',
        'analyze': 'تحليل',
        'add': 'إضافة',
        'rpm': 'دورة المحرك',
        'load': 'الحمل %',
        'temp': 'درجة الحرارة °C',
        'file_loaded': '✅ تم تحميل الملف بنجاح!',
        'columns_detected': 'الأعمدة المكتشفة:',
        'dtc_col_not_found': 'لم يتم العثور على عمود DTC!',
        'select_dtc_col': 'اختر عمود DTC',
        'data_preview': '📊 معاينة البيانات',
        'analysis_results': '📊 تحليل النتائج',
        'high': '🔴 عالي',
        'medium': '🟡 متوسط',
        'low': '🟢 منخفض',
        'details': '🔧 تفاصيل الأعطال',
        'severity': 'الخطورة',
        'price': 'السعر',
        'cause': 'السبب',
        'solution': 'الحل',
        'download': '📥 تحميل التقرير',
        'no_dtc_found': '⚠️ لم يتم العثور على أي عطل',
        'error': '❌ خطأ:',
        'footer': '© 2026 أوتو دياج برو - التشخيص الذكي للسيارات',
        'distribution': 'توزيع الأعطال'
    }
}

# Base de données DTC (bilingue)
DTC_DB = {
    'P0171': {
        'fr': {'desc': 'Système trop pauvre (Banque 1)', 'cause': 'Fuite vide, MAF, O2', 'solution': 'Vérifier fuites, nettoyer MAF', 'prix': '50-300€'},
        'ar': {'desc': 'نظام الوقود فقير جداً', 'cause': 'تسريب هواء، حساس MAF، حساس O2', 'solution': 'تفقد تسريب الهواء، نظف حساس MAF', 'prix': '50-300€'}
    },
    'P0300': {
        'fr': {'desc': 'Ratés d\'allumage multiples', 'cause': 'Bougies, bobines', 'solution': 'Tester bougies et bobines', 'prix': '100-500€'},
        'ar': {'desc': 'احتراق عشوائي متعدد', 'cause': 'البوجيات، الكويلات', 'solution': 'افحص البوجيات والكويلات', 'prix': '100-500€'}
    },
    'P0420': {
        'fr': {'desc': 'Efficacité catalyseur faible', 'cause': 'Catalyseur, O2', 'solution': 'Vérifier capteurs O2', 'prix': '200-1500€'},
        'ar': {'desc': 'كفاءة الكاتاليزر منخفضة', 'cause': 'الكاتاليزر، حساسات O2', 'solution': 'تفقد حساسات الأكسجين', 'prix': '200-1500€'}
    },
    'P0128': {
        'fr': {'desc': 'Thermostat température basse', 'cause': 'Thermostat ouvert', 'solution': 'Remplacer thermostat', 'prix': '50-150€'},
        'ar': {'desc': 'الترموستات درجة حرارة منخفضة', 'cause': 'الترموستات مفتوح', 'solution': 'استبدل الترموستات', 'prix': '50-150€'}
    },
    'P0101': {
        'fr': {'desc': 'MAF performance', 'cause': 'MAF sale', 'solution': 'Nettoyer MAF', 'prix': '80-300€'},
        'ar': {'desc': 'حساس تدفق الهواء MAF', 'cause': 'حساس MAF وسخ', 'solution': 'نظف حساس MAF', 'prix': '80-300€'}
    },
    'P0442': {
        'fr': {'desc': 'Petite fuite EVAP', 'cause': 'Bouchon carburant', 'solution': 'Serrer bouchon', 'prix': '30-200€'},
        'ar': {'desc': 'تسريب صغير EVAP', 'cause': 'غطاء البنزين', 'solution': 'أحكم غطاء البنزين', 'prix': '30-200€'}
    },
}

@st.cache_resource
def train_model():
    data = {
        'DTC': ['P0171', 'P0300', 'P0420', 'P0128', 'P0101', 'P0442'],
        'RPM': [800, 2200, 1500, 750, 2800, 900],
        'Load': [15, 45, 35, 10, 60, 20],
        'Temp': [90, 95, 105, 70, 110, 88],
        'Severity': ['Medium', 'High', 'Medium', 'Low', 'High', 'Low']
    }
    df = pd.DataFrame(data)
    le = LabelEncoder()
    df['DTC_Enc'] = le.fit_transform(df['DTC'])
    X = df[['DTC_Enc', 'RPM', 'Load', 'Temp']]
    y = df['Severity']
    model = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42)
    model.fit(X, y)
    return model, le

# Sélecteur de langue dans sidebar
with st.sidebar:
    lang_choice = st.selectbox("🌐 Language / اللغة", ["Français 🇫", "العربية 🇦"])
    st.session_state.language = 'ar' if 'العربية' in lang_choice else 'fr'

# Texte courant
t = TRANSLATIONS[st.session_state.language]

# CSS avec support RTL
if st.session_state.language == 'ar':
    st.markdown("""
    <style>
        .main { direction: rtl; text-align: right; }
        h1 { text-align: center; background: linear-gradient(90deg, #3b82f6, #1e3a8a); color: white; padding: 20px; border-radius: 10px; }
        .stSidebar { direction: rtl; }
        div[data-testid="stMetricValue"] { direction: ltr; }
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
        h1 { text-align: center; background: linear-gradient(90deg, #1e3a8a, #3b82f6); color: white; padding: 20px; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

st.title(t['title'])

# Sidebar content
with st.sidebar:
    st.header(t['sidebar_title'])
    
    st.subheader(t['import_data'])
    upload_method = st.radio(t['method'], [t['csv'], t['text_ocr'], t['manual']])
    
    if upload_method == t['csv']:
        uploaded_file = st.file_uploader(t['choose_csv'], type=["csv", "txt"])
    else:
        uploaded_file = None
    
    if upload_method == t['manual']:
        st.text_input(t['dtc_code'], key="manual_dtc")
        st.number_input(t['rpm'], value=800, key="manual_rpm")
        st.number_input(t['load'], value=15, key="manual_load")
        st.number_input(t['temp'], value=90, key="manual_temp")
    
    st.subheader(t['search_dtc'])
    search = st.text_input(t['dtc_code'], placeholder=t['placeholder_dtc'])
    if search and search.upper() in DTC_DB:
        lang_key = 'ar' if st.session_state.language == 'ar' else 'fr'
        info = DTC_DB[search.upper()][lang_key]
        st.info(f"**{info['desc']}**\n\n💰 {info['prix']}")

# Main content
if upload_method == t['csv'] and uploaded_file is not None:
    try:
        try:
            df = pd.read_csv(uploaded_file, encoding='utf-8')
        except:
            try:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding='latin-1')
            except:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding='cp1252', sep=';')
        
        st.success(t['file_loaded'])
        st.write(f"**{t['columns_detected']}** {', '.join(df.columns)}")
        
        # Détection colonnes
        dtc_col = rpm_col = load_col = temp_col = None
        for col in df.columns:
            col_lower = col.lower().strip()
            if 'dtc' in col_lower or 'code' in col_lower:
                dtc_col = col
            if 'rpm' in col_lower:
                rpm_col = col
            if 'load' in col_lower:
                load_col = col
            if 'temp' in col_lower:
                temp_col = col
        
        if dtc_col is None:
            st.error(t['dtc_col_not_found'])
            dtc_col = st.selectbox(t['select_dtc_col'], df.columns.tolist())
        
        st.subheader(t['data_preview'])
        st.dataframe(df.head())
        
        # Analyse
        model, le = train_model()
        results = []
        lang_key = 'ar' if st.session_state.language == 'ar' else 'fr'
        
        for idx, row in df.iterrows():
            dtc = str(row[dtc_col]).strip().upper()
            match = re.search(r'P\d{4}', dtc)
            if match:
                dtc = match.group()
            
            rpm = float(row[rpm_col]) if rpm_col and pd.notna(row[rpm_col]) else 0
            load = float(row[load_col]) if load_col and pd.notna(row[load_col]) else 0
            temp = float(row[temp_col]) if temp_col and pd.notna(row[temp_col]) else 0
            
            if dtc in DTC_DB:
                info = DTC_DB[dtc][lang_key]
                try:
                    dtc_enc = le.transform([dtc])[0]
                except:
                    dtc_enc = 0
                
                severity = model.predict([[dtc_enc, rpm, load, temp]])[0]
                
                results.append({
                    t['dtc_code']: dtc,
                    t['search_dtc'].split()[1] if st.session_state.language == 'ar' else 'Description': info['desc'],
                    t['cause']: info['cause'],
                    t['solution']: info['solution'],
                    t['price']: info['prix'],
                    t['severity']: severity
                })
        
        if not results:
            st.warning(t['no_dtc_found'])
            st.stop()
        
        results_df = pd.DataFrame(results)
        
        # Dashboard
        st.subheader(t['analysis_results'])
        col1, col2, col3 = st.columns(3)
        high = len(results_df[results_df[t['severity']] == 'High'])
        med = len(results_df[results_df[t['severity']] == 'Medium'])
        low = len(results_df[results_df[t['severity']] == 'Low'])
        
        col1.metric(t['high'], high)
        col2.metric(t['medium'], med)
        col3.metric(t['low'], low)
        
        # Graphique
        fig = px.pie(results_df, names=t['severity'], title=t['distribution'],
                     color_discrete_map={'High': '#ef4444', 'Medium': '#f59e0b', 'Low': '#10b981'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Détails
        st.subheader(t['details'])
        for idx, row in results_df.iterrows():
            sev = row[t['severity']]
            box_class = "error-box" if sev == 'High' else "warning-box" if sev == 'Medium' else "success-box"
            
            st.markdown(f"""
            <div class="{box_class}" style="padding: 15px; margin: 10px 0; border-radius: 5px; border-right: 5px solid {'#ef4444' if sev == 'High' else '#f59e0b' if sev == 'Medium' else '#10b981'}; background: {'#fee2e2' if sev == 'High' else '#fef3c7' if sev == 'Medium' else '#d1fae5'}">
                <h4>🔧 {row[list(row.keys())[0]]}</h4>
                <p><strong>{t['severity']}:</strong> {sev} | <strong>{t['price']}:</strong> {row[t['price']]}</p>
                <p><strong>{t['cause']}:</strong> {row[t['cause']]}</p>
                <p><strong>{t['solution']}:</strong> {row[t['solution']]}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Export
        csv = results_df.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(t['download'], csv, f"diag_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")
        
    except Exception as e:
        st.error(f"{t['error']} {str(e)}")

# Footer
st.markdown("---")
st.markdown(t['footer'])
