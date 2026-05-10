import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import re

# Configuration
st.set_page_config(page_title="AutoDiag Pro", layout="wide", page_icon="🚗")

if 'language' not in st.session_state:
    st.session_state.language = 'fr'
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'results_df' not in st.session_state:
    st.session_state.results_df = None

BACKGROUND_IMAGE = "https://images.unsplash.com/photo-1618843479313-895a43d5a593?w=1920"

TRANSLATIONS = {
    'fr': {
        'title': 'AutoDiag Pro', 'subtitle': 'Diagnostic Intelligent & Rapide',
        'sidebar_title': 'Navigation', 'dashboard': '📊 Dashboard',
        'live_data': '📈 Live Data', 'dtcs': '🔧 DTC Codes',
        'clear_codes': '❌ Clear Codes', 'reports': '📄 Reports',
        'import_data': '📁 Importer Données', 'method': 'Méthode:',
        'csv': 'CSV', 'text_ocr': 'Texte/OCR', 'manual': 'Saisie Manuelle',
        'choose_csv': 'Choisir fichier CSV', 'upload': 'Upload',
        'dtc_code': 'Code DTC', 'rpm': 'RPM', 'load': 'Charge %', 'temp': 'Température °C',
        'file_loaded': '✅ Fichier chargé avec succès!',
        'dtc_col_not_found': 'Colonne DTC non trouvée!',
        'analysis_results': '📊 Analyse des Résultats', 'details': '🔧 Détails des défauts',
        'severity': 'Sévérité', 'price': 'Prix estimé', 'cause': 'Cause probable',
        'solution': 'Solution', 'download': '📥 Télécharger Rapport',
        'no_dtc_found': '⚠️ Aucun code DTC reconnu', 'error': '❌ Erreur:',
        'category': 'Catégorie', 'status': 'Statut', 'known': '✅ Connu',
        'category_detected': 'ℹ️ Catégorie détectée', 'critical': 'Critical',
        'warning': 'Warning', 'info': 'Info', 'clear_data': '🗑️ Effacer les données'
    },
    'ar': {
        'title': 'أوتو دياج برو', 'subtitle': 'التشخيص الذكي والسريع',
        'sidebar_title': 'القائمة', 'dashboard': '📊 لوحة التحكم',
        'live_data': '📈 البيانات المباشرة', 'dtcs': '🔧 أكواد الأعطال',
        'clear_codes': '❌ مسح الأعطال', 'reports': '📄 التقارير',
        'import_data': '📁 استيراد البيانات', 'method': 'الطريقة:',
        'csv': 'CSV', 'text_ocr': 'نص/مسح', 'manual': 'إدخال يدوي',
        'choose_csv': 'اختر ملف CSV', 'upload': 'رفع',
        'dtc_code': 'كود العطل', 'rpm': 'دورة المحرك', 'load': 'الحمل %', 'temp': 'درجة الحرارة °C',
        'file_loaded': '✅ تم تحميل الملف بنجاح!',
        'dtc_col_not_found': 'لم يتم العثور على عمود DTC!',
        'analysis_results': '📊 تحليل النتائج', 'details': '🔧 تفاصيل الأعطال',
        'severity': 'الخطورة', 'price': 'السعر المقدر', 'cause': 'السبب المحتمل',
        'solution': 'الحل', 'download': '📥 تحميل التقرير',
        'no_dtc_found': '⚠️ لم يتم العثور على أي عطل', 'error': '❌ خطأ:',
        'category': 'الفئة', 'status': 'الحالة', 'known': '✅ معروف',
        'category_detected': 'ℹ️ الفئة مكتشفة', 'critical': 'حرج',
        'warning': 'تحذير', 'info': 'معلومات', 'clear_data': '🗑️ مسح البيانات'
    }
}

def extract_valid_dtc(value):
    if not value: return None
    value = str(value).strip().upper()
    match = re.search(r'\b(P|B|C|U)(\d{4})\b', value)
    return match.group(0) if match else None

def clean_numeric(val):
    if val is None or str(val).lower() in ['none', 'nan', 'error', '--', '', 'n/a']: return 0.0
    clean = re.sub(r'[^\d.]', '', str(val))
    try: return float(clean) if clean else 0.0
    except ValueError: return 0.0

def get_dtc_info(dtc_code):
    dtc = dtc_code.strip().upper()
    specific_db = {
        'P0171': {'desc_fr': 'Système trop pauvre (Banque 1)', 'desc_ar': 'نظام الوقود فقير جداً', 'cause_fr': 'Fuite d\'air, MAF, O2', 'cause_ar': 'تسريب هواء، MAF، O2', 'solution_fr': 'Vérifier fuites, nettoyer MAF', 'solution_ar': 'تفقد التسريب، نظف MAF', 'prix': '50-300€', 'categorie': 'Injection'},
        'P0300': {'desc_fr': 'Ratés d\'allumage multiples', 'desc_ar': 'احتراق عشوائي متعدد', 'cause_fr': 'Bougies, bobines', 'cause_ar': 'البوجيات، الكويلات', 'solution_fr': 'Tester bougies et bobines', 'solution_ar': 'افحص البوجيات والكويلات', 'prix': '100-500€', 'categorie': 'Allumage'},
        'C0035': {'desc_fr': 'Capteur vitesse roue avant gauche', 'desc_ar': 'حساس سرعة العجلة الأمامية اليسرى', 'cause_fr': 'Capteur ABS AVG, câblage', 'cause_ar': 'حساس ABS الأمامي الأيسر، أسلاك', 'solution_fr': 'Tester capteur AVG', 'solution_ar': 'افحص حساس العجلة اليسرى', 'prix': '100-300€', 'categorie': 'ABS'},
        'B0000': {'desc_fr': 'Airbag conducteur circuit', 'desc_ar': 'دائرة وسادة السائق', 'cause_fr': 'Airbag, câblage', 'cause_ar': 'الوسادة، الأسلاك', 'solution_fr': 'Tester airbag', 'solution_ar': 'افحص الوسادة', 'prix': '200-800€', 'categorie': 'Airbag'},
        'U0100': {'desc_fr': 'Perte communication ECU', 'desc_ar': 'فقدان اتصال الكمبيوتر', 'cause_fr': 'ECU ne répond pas', 'cause_ar': 'الكمبيوتر لا يستجيب', 'solution_fr': 'Tester ECU et CAN', 'solution_ar': 'افحص الكمبيوتر وشبكة CAN', 'prix': '200-1000€', 'categorie': 'Réseau'},
    }
    if dtc in specific_db: return specific_db[dtc], True

    rules = {
        'P01': {'desc_fr': f'Problème injection ({dtc})', 'desc_ar': f'مشكل حقن ({dtc})', 'cause_fr': 'Système fuel/air', 'cause_ar': 'نظام الوقود', 'solution_fr': 'Vérifier injection', 'solution_ar': 'تفقد الحقن', 'prix': '80-400€', 'categorie': 'Injection'},
        'P03': {'desc_fr': f'Problème allumage ({dtc})', 'desc_ar': f'مشكل إشعال ({dtc})', 'cause_fr': 'Bougies/bobines', 'cause_ar': 'بوجيات/كويلات', 'solution_fr': 'Tester allumage', 'solution_ar': 'افحص الإشعال', 'prix': '80-450€', 'categorie': 'Allumage'},
        'P07': {'desc_fr': f'Problème transmission ({dtc})', 'desc_ar': f'مشكل علبة سرعة ({dtc})', 'cause_fr': 'Boîte auto', 'cause_ar': 'علبة السرعة', 'solution_fr': 'Diagnostiquer boîte', 'solution_ar': 'شخص العلبة', 'prix': '200-2000€', 'categorie': 'Transmission'},
        'C': {'desc_fr': f'Problème châssis ({dtc})', 'desc_ar': f'مشكل شاسيه ({dtc})', 'cause_fr': 'ABS/ESP/freins', 'cause_ar': 'ABS/ESP/فرامل', 'solution_fr': 'Diagnostiquer freins', 'solution_ar': 'شخص الفرامل', 'prix': '100-1000€', 'categorie': 'Châssis'},
        'B': {'desc_fr': f'Problème carrosserie ({dtc})', 'desc_ar': f'مشكل هيكل ({dtc})', 'cause_fr': 'Airbag/habitacle', 'cause_ar': 'وسائد/مقصورة', 'solution_fr': 'Vérifier airbag', 'solution_ar': 'تفقد الوسائد', 'prix': '100-800€', 'categorie': 'Carrosserie'},
        'U': {'desc_fr': f'Problème réseau ({dtc})', 'desc_ar': f'مشكل شبكة ({dtc})', 'cause_fr': 'CAN Bus', 'cause_ar': 'شبكة CAN', 'solution_fr': 'Vérifier CAN', 'solution_ar': 'تفقد شبكة CAN', 'prix': '150-600€', 'categorie': 'Réseau'},
    }
    for prefix, info in rules.items():
        if dtc.startswith(prefix): return info, False
    return {'desc_fr': f'Code {dtc} inconnu', 'desc_ar': f'كود {dtc} غير معروف', 'cause_fr': 'Consulter manuel', 'cause_ar': 'راجع الدليل', 'solution_fr': 'Diagnostic nécessaire', 'solution_ar': 'تشخيص ضروري', 'prix': 'N/A', 'categorie': 'Inconnue'}, False

@st.cache_resource
def train_model():
    data = {'DTC': ['P0171', 'P0300', 'P0420', 'P0128', 'P0101', 'P0442', 'P0172', 'P0301'], 'RPM': [800, 2200, 1500, 750, 2800, 900, 2000, 2400], 'Load': [15, 45, 35, 10, 60, 20, 40, 50], 'Temp': [90, 95, 105, 70, 110, 88, 92, 98], 'Severity': ['Medium', 'High', 'Medium', 'Low', 'High', 'Low', 'Medium', 'High']}
    df = pd.DataFrame(data)
    le = LabelEncoder()
    df['DTC_Enc'] = le.fit_transform(df['DTC'])
    X = df[['DTC_Enc', 'RPM', 'Load', 'Temp']]
    y = df['Severity']
    model = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42)
    model.fit(X, y)
    return model, le

def process_file(file):
    """Process uploaded CSV and return results DataFrame"""
    try:
        try: df = pd.read_csv(file, encoding='utf-8')
        except:
            try: df = pd.read_csv(file, encoding='latin-1')
            except: df = pd.read_csv(file, encoding='cp1252', sep=';')
        
        dtc_col = rpm_col = load_col = temp_col = None
        for col in df.columns:
            cl = col.lower()
            if 'dtc' in cl or 'كود' in cl: dtc_col = col
            if 'rpm' in cl: rpm_col = col
            if 'load' in cl: load_col = col
            if 'temp' in cl: temp_col = col
        
        if not dtc_col:
            for col in df.columns:
                if df[col].dtype == 'object' and df[col].str.contains(r'P\d{4}|C\d{4}|B\d{4}|U\d{4}', regex=True, na=False).any():
                    dtc_col = col
                    break
        
        if not dtc_col:
            st.error("Colonne DTC non trouvée!")
            return None
        
        df['RPM'] = df[rpm_col].apply(clean_numeric) if rpm_col else 0.0
        df['Load'] = df[load_col].apply(clean_numeric) if load_col else 0.0
        df['Temp'] = df[temp_col].apply(clean_numeric) if temp_col else 0.0
        
        model, le = train_model()
        results = []
        lang_key = st.session_state.language
        
        for idx, row in df.iterrows():
            raw_val = str(row[dtc_col]).strip()
            if not raw_val or raw_val.lower() in ['none', 'nan', '-', '']: continue
            
            dtc = extract_valid_dtc(raw_val)
            if not dtc: continue
            
            rpm = float(df['RPM'].iloc[idx]) if isinstance(df['RPM'].iloc[idx], (int, float)) else 0.0
            load = float(df['Load'].iloc[idx]) if isinstance(df['Load'].iloc[idx], (int, float)) else 0.0
            temp = float(df['Temp'].iloc[idx]) if isinstance(df['Temp'].iloc[idx], (int, float)) else 0.0
            
            info, is_known = get_dtc_info(dtc)
            try: dtc_enc = le.transform([dtc])[0] if dtc in le.classes_ else 0
            except: dtc_enc = 0
            severity = model.predict([[dtc_enc, rpm, load, temp]])[0]
            
            # Use CSV columns if they exist, otherwise use get_dtc_info
            if lang_key == 'ar':
                description = row.get('Description', info.get('desc_ar', 'Unknown'))
                cause = row.get('السبب المحتمل', info.get('cause_ar', 'Unknown'))
                solution = row.get('الحل', info.get('solution_ar', 'Unknown'))
                category = row.get('الفئة', info.get('categorie', 'Unknown'))
                price = row.get('السعر المقدر', info.get('prix', 'N/A'))
                status = row.get('الحالة', '✅ Connu' if is_known else 'ℹ️ Détecté')
            else:
                description = row.get('Description', info.get('desc_fr', 'Unknown'))
                cause = row.get('cause_fr', info.get('cause_fr', 'Unknown'))
                solution = row.get('solution_fr', info.get('solution_fr', 'Unknown'))
                category = row.get('Catégorie', info.get('categorie', 'Unknown'))
                price = row.get('Prix', info.get('prix', 'N/A'))
                status = row.get('Statut', '✅ Connu' if is_known else 'ℹ️ Détecté')
            
            results.append({
                'Code': dtc, 'Description': description,
                'Catégorie': category, 'Cause': cause,
                'Solution': solution, 'Prix': price,
                'Sévérité': severity, 'Statut': status
            })
        return pd.DataFrame(results) if results else None
    except Exception as e:
        st.error(f"Erreur: {str(e)}")
        return None

# CSS
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
* {{ font-family: 'Inter', sans-serif; }}
.main {{ background-image: url('{BACKGROUND_IMAGE}'); background-size: cover; background-position: center; background-attachment: fixed; min-height: 100vh; }}
.stApp {{ background: rgba(15, 23, 42, 0.4); backdrop-filter: blur(10px); }}
[data-testid="stSidebar"] {{ background: rgba(255, 255, 255, 0.95); backdrop-filter: blur(20px); border-right: 1px solid rgba(0, 0, 0, 0.1); }}
.metric-card {{ background: linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(240,248,255,0.95) 100%); border-radius: 15px; padding: 20px; text-align: center; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08); border: 1px solid rgba(59, 130, 246, 0.2); }}
.glass-card {{ background: rgba(255, 255, 255, 0.9); backdrop-filter: blur(20px); border-radius: 20px; padding: 25px; box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1); border: 1px solid rgba(255, 255, 255, 0.5); }}
.dtc-item {{ background: rgba(255, 255, 255, 0.95); border-left: 4px solid #3b82f6; padding: 15px; margin: 10px 0; border-radius: 10px; box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05); }}
.badge-critical {{ background: #ef4444; color: white; padding: 5px 15px; border-radius: 20px; font-size: 0.85em; font-weight: 600; }}
.badge-warning {{ background: #f59e0b; color: white; padding: 5px 15px; border-radius: 20px; font-size: 0.85em; font-weight: 600; }}
.badge-info {{ background: #10b981; color: white; padding: 5px 15px; border-radius: 20px; font-size: 0.85em; font-weight: 600; }}
h1, h2, h3 {{ color: #1e293b !important; }}
.footer {{ text-align: center; padding: 20px; color: rgba(255, 255, 255, 0.8); margin-top: 40px; }}
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    lang_options = {"Français 🇫🇷": "fr", "العربية 🇲": "ar"}
    selected_label = st.selectbox("🌐 Language / اللغة", list(lang_options.keys()), index=list(lang_options.values()).index(st.session_state.language))
    new_lang = lang_options[selected_label]
    
    if new_lang != st.session_state.language:
        st.session_state.language = new_lang
        st.rerun()
    
    t = TRANSLATIONS[st.session_state.language]
    
    st.markdown("""<div style="text-align: center; padding: 20px; margin-bottom: 20px;"><h2 style="color: #3b82f6; margin: 0; font-size: 1.8em;">🚗 AutoDiag Pro</h2><p style="color: #64748b; font-size: 0.9em;">Mercedes Edition</p></div>""", unsafe_allow_html=True)
    
    st.radio("Navigation", [t['dashboard'], t['live_data'], t['dtcs'], t['clear_codes'], t['reports']], label_visibility="collapsed")
    st.markdown("---")
    st.subheader(t['import_data'])
    upload_method = st.radio(t['method'], [t['csv'], t['text_ocr'], t['manual']])
    
    if upload_method == t['csv']:
        uploaded_file = st.file_uploader(t['choose_csv'], type=["csv", "txt"])
        if uploaded_file is not None:
            st.session_state.uploaded_file = uploaded_file
            st.success(t['file_loaded'])
            if st.session_state.results_df is None:
                st.session_state.results_df = process_file(uploaded_file)
    elif upload_method == t['manual']:
        st.text_input(t['dtc_code'], key="manual_dtc")
        st.number_input(t['rpm'], value=800, key="manual_rpm")
        st.number_input(t['load'], value=15, key="manual_load")
        st.number_input(t['temp'], value=90, key="manual_temp")
    
    if st.session_state.results_df is not None:
        if st.button(t['clear_data'], type="secondary"):
            st.session_state.uploaded_file = None
            st.session_state.results_df = None
            st.rerun()

# Main Content
t = TRANSLATIONS[st.session_state.language]

st.markdown(f"""<div class="glass-card" style="margin-bottom: 30px;"><div style="display: flex; align-items: center; justify-content: space-between;"><div><h1 style="margin: 0; color: #1e293b;">{t['title']}</h1><p style="margin: 5px 0 0 0; color: #64748b;">{t['subtitle']}</p></div><div style="font-size: 3em;">🔧</div></div></div>""", unsafe_allow_html=True)

if st.session_state.results_df is not None:
    res_df = st.session_state.results_df
    
    st.subheader(t['analysis_results'])
    c1, c2, c3, c4 = st.columns(4)
    high = len(res_df[res_df['Sévérité'] == 'High'])
    med = len(res_df[res_df['Sévérité'] == 'Medium'])
    low = len(res_df[res_df['Sévérité'] == 'Low'])
    health_score = max(0, 100 - (high * 20) - (med * 10))
    
    with c1: st.markdown(f'<div class="metric-card"><div style="font-size:2.5em">🔴</div><div style="color:#64748b">{t["critical"]}</div><div style="font-size:2em;font-weight:bold;color:#ef4444">{high}</div></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="metric-card"><div style="font-size:2.5em">⚠️</div><div style="color:#64748b">{t["warning"]}</div><div style="font-size:2em;font-weight:bold;color:#f59e0b">{med}</div></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="metric-card"><div style="font-size:2.5em">✅</div><div style="color:#64748b">{t["info"]}</div><div style="font-size:2em;font-weight:bold;color:#10b981">{low}</div></div>', unsafe_allow_html=True)
    with c4: st.markdown(f'<div class="metric-card"><div style="font-size:2.5em">🏥</div><div style="color:#64748b">Health</div><div style="font-size:2em;font-weight:bold;color:#3b82f6">{health_score}%</div></div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        fig_pie = px.pie(res_df, names='Catégorie', title='📊 Répartition', hole=0.4)
        st.plotly_chart(fig_pie, use_container_width=True)
    with col_chart2:
        severity_counts = res_df['Sévérité'].value_counts()
        fig_bar = px.bar(x=severity_counts.index, y=severity_counts.values, title='📈 Sévérité', color=severity_counts.index)
        st.plotly_chart(fig_bar, use_container_width=True)
    
    st.subheader(t['details'])
    for _, r in res_df.iterrows():
        sev = r['Sévérité']
        badge_class = "badge-critical" if sev == "High" else "badge-warning" if sev == "Medium" else "badge-info"
        st.markdown(f'''<div class="dtc-item"><h3 style="margin:0;color:#3b82f6">🔧 {r["Code"]}</h3><p><strong>📂 {t["category"]}:</strong> {r["Catégorie"]} | <strong>💰 {t["price"]}:</strong> {r["Prix"]}</p><p><strong>📝 Description:</strong> {r["Description"]}</p><p><strong>🔍 {t["cause"]}:</strong> {r["Cause"]}</p><p><strong>🔧 {t["solution"]}:</strong> {r["Solution"]}</p><span class="{badge_class}">{r["Sévérité"]}</span></div>''', unsafe_allow_html=True)
    
    csv = res_df.to_csv(index=False, encoding='utf-8-sig')
    st.download_button(label=t['download'], data=csv, file_name=f"diag_{datetime.now().strftime('%Y%m%d')}.csv", mime="text/csv")

elif upload_method == t['manual'] and st.session_state.get('manual_dtc'):
    dtc = extract_valid_dtc(st.session_state.manual_dtc)
    if dtc:
        info, _ = get_dtc_info(dtc)
        lang_key = st.session_state.language
        st.info(f"**{dtc}**: {info.get(f'desc_{lang_key}', 'Unknown')}\n\n💰 {info['prix']}")
else:
    st.markdown('''<div class="glass-card" style="text-align: center; padding: 60px 40px;"><div style="font-size: 4em; margin-bottom: 20px;">🚗</div><h2 style="color: #1e293b; margin-bottom: 15px;">Bienvenue sur AutoDiag Pro</h2><p style="color: #64748b; font-size: 1.1em; max-width: 600px; margin: 0 auto 30px auto;">Système intelligent de diagnostic automobile avec analyse AI des codes DTC</p></div>''', unsafe_allow_html=True)

st.markdown('<div class="footer"><p>© 2026 AutoDiag Pro - Mercedes Edition</p></div>', unsafe_allow_html=True)
