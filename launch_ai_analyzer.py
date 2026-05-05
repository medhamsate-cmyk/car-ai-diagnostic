import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import re

# ============================================================================
# CONFIGURATION & TRANSLATIONS
# ============================================================================
st.set_page_config(page_title="AutoDiag Pro", layout="wide", page_icon="🚗")

if 'language' not in st.session_state:
    st.session_state.language = 'fr'

TRANSLATIONS = {
    'fr': {
        'title': 'AutoDiag Pro - Diagnostic Intelligent',
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
        'high': 'Élevée',
        'medium': 'Moyenne',
        'low': 'Faible',
        'details': '🔧 Détails des défauts',
        'severity': 'Sévérité',
        'price': 'Prix estimé',
        'cause': 'Cause probable',
        'solution': 'Solution',
        'download': '📥 Télécharger Rapport',
        'no_dtc_found': '⚠️ Aucun code DTC reconnu',
        'error': '❌ Erreur:',
        'footer': '© 2026 AutoDiag Pro - Diagnostic Automobile',
        'distribution': 'Répartition des défauts',
        'category': 'Catégorie',
        'status': 'Statut',
        'known': '✅ Connu',
        'category_detected': 'ℹ️ Catégorie détectée'
    },
    'ar': {
        'title': 'أوتو دياج برو - التشخيص الذكي',
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
        'high': 'عالي',
        'medium': 'متوسط',
        'low': 'منخفض',
        'details': '🔧 تفاصيل الأعطال',
        'severity': 'الخطورة',
        'price': 'السعر المقدر',
        'cause': 'السبب المحتمل',
        'solution': 'الحل',
        'download': '📥 تحميل التقرير',
        'no_dtc_found': '⚠️ لم يتم العثور على أي عطل',
        'error': '❌ خطأ:',
        'footer': '© 2026 أوتو دياج برو - التشخيص الذكي للسيارات',
        'distribution': 'توزيع الأعطال',
        'category': 'الفئة',
        'status': 'الحالة',
        'known': '✅ معروف',
        'category_detected': 'ℹ️ الفئة مكتشفة'
    }
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
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

# ============================================================================
# SMART DATABASE (ALL CODES COVERED)
# ============================================================================
def get_dtc_info(dtc_code):
    dtc = dtc_code.strip().upper()
    
    # --- SPECIFIC CODES (Examples) ---
    specific_db = {
        'P0171': {'desc_fr': 'Système trop pauvre (Banque 1)', 'desc_ar': 'نظام الوقود فقير جداً (بنك 1)', 'cause_fr': 'Fuite d\'air, MAF, O2', 'cause_ar': 'تسريب هواء، MAF، O2', 'solution_fr': 'Vérifier fuites, nettoyer MAF', 'solution_ar': 'تفقد التسريب، نظف MAF', 'prix': '50-300€', 'categorie': 'Injection'},
        'P0300': {'desc_fr': 'Ratés d\'allumage multiples', 'desc_ar': 'احتراق عشوائي متعدد', 'cause_fr': 'Bougies, bobines', 'cause_ar': 'البوجيات، الكويلات', 'solution_fr': 'Tester bougies et bobines', 'solution_ar': 'افحص البوجيات والكويلات', 'prix': '100-500€', 'categorie': 'Allumage'},
        'P0420': {'desc_fr': 'Efficacité catalyseur faible', 'desc_ar': 'كفاءة الكاتاليزر منخفضة', 'cause_fr': 'Catalyseur usé, O2', 'cause_ar': 'الكاتاليزر تالف، O2', 'solution_fr': 'Vérifier catalyseur et O2', 'solution_ar': 'تفقد الكاتاليزر و O2', 'prix': '200-1500€', 'categorie': 'Échappement'},
        'C0035': {'desc_fr': 'Capteur vitesse roue avant gauche', 'desc_ar': 'حساس سرعة العجلة الأمامية اليسرى', 'cause_fr': 'Capteur ABS AVG, câblage', 'cause_ar': 'حساس ABS الأمامي الأيسر، أسلاك', 'solution_fr': 'Tester capteur AVG', 'solution_ar': 'افحص حساس العجلة اليسرى', 'prix': '100-300€', 'categorie': 'ABS'},
        'C0121': {'desc_fr': 'Circuit alimentation valves ABS', 'desc_ar': 'دائرة تغذية صمامات ABS', 'cause_fr': 'Valves ABS, câblage', 'cause_ar': 'صمامات ABS، أسلاك', 'solution_fr': 'Tester valves ABS', 'solution_ar': 'افحص صمامات ABS', 'prix': '300-800€', 'categorie': 'ABS'},
        'B0000': {'desc_fr': 'Airbag conducteur circuit', 'desc_ar': 'دائرة وسادة السائق', 'cause_fr': 'Airbag, câblage', 'cause_ar': 'الوسادة، الأسلاك', 'solution_fr': 'Tester airbag', 'solution_ar': 'افحص الوسادة', 'prix': '200-800€', 'categorie': 'Airbag'},
        'U0100': {'desc_fr': 'Perte communication ECU', 'desc_ar': 'فقدان اتصال الكمبيوتر', 'cause_fr': 'ECU ne répond pas', 'cause_ar': 'الكمبيوتر لا يستجيب', 'solution_fr': 'Tester ECU et CAN', 'solution_ar': 'افحص الكمبيوتر وشبكة CAN', 'prix': '200-1000€', 'categorie': 'Réseau'},
    }
    
    if dtc in specific_db: return specific_db[dtc], True

    # --- CATEGORY RULES (COVERS ALL OTHER CODES) ---
    rules = {
        'P01': {'desc_fr': f'Problème injection/admission ({dtc})', 'desc_ar': f'مشكل حقن/دخول ({dtc})', 'cause_fr': 'Système fuel/air', 'cause_ar': 'نظام الوقود/الهواء', 'solution_fr': 'Vérifier injection/fuites', 'solution_ar': 'تفقد الحقن/التسريب', 'prix': '80-400€', 'categorie': 'Injection'},
        'P02': {'desc_fr': f'Problème injecteur ({dtc})', 'desc_ar': f'مشكل بخاخ ({dtc})', 'cause_fr': 'Injecteur/câblage', 'cause_ar': 'بخاخ/أسلاك', 'solution_fr': 'Tester injecteur', 'solution_ar': 'افحص البخاخ', 'prix': '100-500€', 'categorie': 'Injecteurs'},
        'P03': {'desc_fr': f'Problème allumage ({dtc})', 'desc_ar': f'مشكل إشعال ({dtc})', 'cause_fr': 'Bougies/bobines', 'cause_ar': 'بوجيات/كويلات', 'solution_fr': 'Tester allumage', 'solution_ar': 'افحص الإشعال', 'prix': '80-450€', 'categorie': 'Allumage'},
        'P04': {'desc_fr': f'Problème antipollution ({dtc})', 'desc_ar': f'مشكل تلوث ({dtc})', 'cause_fr': 'Catalyseur/EGR/EVAP', 'cause_ar': 'كاتاليزر/EGR/EVAP', 'solution_fr': 'Vérifier échappement', 'solution_ar': 'تفقد العادم', 'prix': '100-1500€', 'categorie': 'Antipollution'},
        'P05': {'desc_fr': f'Problème vitesse/régime ({dtc})', 'desc_ar': f'مشكل سرعة/دوران ({dtc})', 'cause_fr': 'Capteurs vitesse', 'cause_ar': 'حساسات السرعة', 'solution_fr': 'Tester capteurs', 'solution_ar': 'افحص الحساسات', 'prix': '50-300€', 'categorie': 'Vitesse'},
        'P06': {'desc_fr': f'Problème calculateur ({dtc})', 'desc_ar': f'مشكل كمبيوتر ({dtc})', 'cause_fr': 'ECU/câblage', 'cause_ar': 'كمبيوتر/أسلاك', 'solution_fr': 'Vérifier ECU', 'solution_ar': 'تفقد الكمبيوتر', 'prix': '100-1000€', 'categorie': 'Électronique'},
        'P07': {'desc_fr': f'Problème transmission ({dtc})', 'desc_ar': f'مشكل علبة سرعة ({dtc})', 'cause_fr': 'Boîte auto', 'cause_ar': 'علبة السرعة', 'solution_fr': 'Diagnostiquer boîte', 'solution_ar': 'شخص العلبة', 'prix': '200-2000€', 'categorie': 'Transmission'},
        'P08': {'desc_fr': f'Problème transmission ({dtc})', 'desc_ar': f'مشكل علبة سرعة ({dtc})', 'cause_fr': 'Boîte/embrayage', 'cause_ar': 'علبة/قابض', 'solution_fr': 'Vérifier transmission', 'solution_ar': 'تفقد العلبة', 'prix': '150-1500€', 'categorie': 'Transmission'},
        'B':   {'desc_fr': f'Problème carrosserie ({dtc})', 'desc_ar': f'مشكل هيكل ({dtc})', 'cause_fr': 'Airbag/habitacle', 'cause_ar': 'وسائد/مقصورة', 'solution_fr': 'Vérifier airbag', 'solution_ar': 'تفقد الوسائد', 'prix': '100-800€', 'categorie': 'Carrosserie'},
        'C':   {'desc_fr': f'Problème châssis ({dtc})', 'desc_ar': f'مشكل شاسيه ({dtc})', 'cause_fr': 'ABS/ESP/freins', 'cause_ar': 'ABS/ESP/فرامل', 'solution_fr': 'Diagnostiquer freins', 'solution_ar': 'شخص الفرامل', 'prix': '100-1000€', 'categorie': 'Châssis'},
        'U':   {'desc_fr': f'Problème réseau ({dtc})', 'desc_ar': f'مشكل شبكة ({dtc})', 'cause_fr': 'CAN Bus', 'cause_ar': 'شبكة CAN', 'solution_fr': 'Vérifier CAN', 'solution_ar': 'تفقد شبكة CAN', 'prix': '150-600€', 'categorie': 'Réseau'},
    }
    
    for prefix, info in rules.items():
        if dtc.startswith(prefix): return info, False
        
    return {'desc_fr': f'Code {dtc} inconnu', 'desc_ar': f'كود {dtc} غير معروف', 'cause_fr': 'Consulter manuel', 'cause_ar': 'راجع الدليل', 'solution_fr': 'Diagnostic nécessaire', 'solution_ar': 'تشخيص ضروري', 'prix': 'N/A', 'categorie': 'Inconnue'}, False

# ============================================================================
# AI MODEL
# ============================================================================
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

# ============================================================================
# UI DESIGN (CSS)
# ============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    * { font-family: 'Inter', sans-serif; }
    .main { background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); min-height: 100vh; padding: 2rem; }
    h1 { text-align: center; color: #2c3e50; font-weight: 700; margin-bottom: 2rem; }
    .metric-card { background: white; padding: 1.5rem; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); text-align: center; transition: transform 0.2s; }
    .metric-card:hover { transform: translateY(-5px); }
    .severity-high { background: #ffebee; border-left: 5px solid #ef5350; padding: 1rem; margin: 1rem 0; border-radius: 8px; }
    .severity-medium { background: #fff8e1; border-left: 5px solid #ffa726; padding: 1rem; margin: 1rem 0; border-radius: 8px; }
    .severity-low { background: #e8f5e9; border-left: 5px solid #66bb6a; padding: 1rem; margin: 1rem 0; border-radius: 8px; }
    .info-box { background: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.05); margin-bottom: 1rem; }
    [data-testid="stSidebar"] { background-color: #ffffff; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    lang_choice = st.selectbox("🌐 Language / اللغة", ["Français 🇫🇷", "العربية 🇲🇦"])
    st.session_state.language = 'ar' if 'العربية' in lang_choice else 'fr'
    
    st.header("⚙️ Options")
    st.subheader("📁 Importer Données")
    upload_method = st.radio("Méthode:", ["CSV", "Texte/OCR", "Saisie Manuelle"])
    
    uploaded_file = None
    if upload_method == "CSV":
        uploaded_file = st.file_uploader("Choisir CSV", type=["csv", "txt"])
    elif upload_method == "Saisie Manuelle":
        st.text_input("Code DTC", key="manual_dtc")
        st.number_input("RPM", value=800, key="manual_rpm")
        st.number_input("Load %", value=15, key="manual_load")
        st.number_input("Temp °C", value=90, key="manual_temp")
        
    st.subheader("🔍 Recherche Rapide")
    search = st.text_input("Code DTC", placeholder="Ex: P0171")
    if search:
        valid_search = extract_valid_dtc(search)
        if valid_search:
            info, _ = get_dtc_info(valid_search)
            lang_key = 'ar' if st.session_state.language == 'ar' else 'fr'
            st.info(f"**{valid_search}**: {info[f'desc_{lang_key}']}\n\n💰 {info['prix']}")

t = TRANSLATIONS[st.session_state.language]

# ============================================================================
# MAIN CONTENT
# ============================================================================
st.title(t['title'])

# HEADER BANNER
st.markdown("""
<div style="background: linear-gradient(90deg, #1e3a8a, #3b82f6); padding: 1.5rem; border-radius: 15px; color: white; display: flex; align-items: center; justify-content: space-between;">
    <div><h2 style="margin:0; color:white;">🚗 AutoDiag Pro</h2><p style="margin:0; opacity:0.9;">Diagnostic Intelligent & Rapide</p></div>
    <div style="font-size: 2rem;">🔧</div>
</div>
""", unsafe_allow_html=True)

# PROCESSING
if upload_method == "CSV" and uploaded_file is not None:
    try:
        try: df = pd.read_csv(uploaded_file, encoding='utf-8')
        except: 
            try: df = pd.read_csv(uploaded_file, encoding='latin-1')
            except: df = pd.read_csv(uploaded_file, encoding='cp1252', sep=';')
        
        st.success(t['file_loaded'])
        
        # Detect Columns
        dtc_col = rpm_col = load_col = temp_col = None
        for col in df.columns:
            cl = col.lower()
            if 'dtc' in cl or 'code' in cl: dtc_col = col
            if 'rpm' in cl: rpm_col = col
            if 'load' in cl: load_col = col
            if 'temp' in cl: temp_col = col
            
        if not dtc_col:
            st.error(t['dtc_col_not_found']); st.stop()
            
        # Clean Data
        df['RPM'] = df[rpm_col].apply(clean_numeric) if rpm_col else 0.0
        df['Load'] = df[load_col].apply(clean_numeric) if load_col else 0.0
        df['Temp'] = df[temp_col].apply(clean_numeric) if temp_col else 0.0
        
        st.subheader(t['data_preview'])
        st.dataframe(df.head())
        
        # Analyze
        model, le = train_model()
        results = []
        lang_key = 'ar' if st.session_state.language == 'ar' else 'fr'
        
        for idx, row in df.iterrows():
            raw_val = str(row[dtc_col]).strip()
            if not raw_val or raw_val.lower() in ['none', 'nan', '-', '']: continue
            
            dtc = extract_valid_dtc(raw_val)
            if not dtc: continue
            
            rpm, load, temp = float(df['RPM'].iloc[idx]), float(df['Load'].iloc[idx]), float(df['Temp'].iloc[idx])
            info, is_known = get_dtc_info(dtc)
            
            try: dtc_enc = le.transform([dtc])[0] if dtc in le.classes_ else 0
            except: dtc_enc = 0
            
            severity = model.predict([[dtc_enc, rpm, load, temp]])[0]
            
            results.append({
                t['dtc_code']: dtc, 'Description': info[f'desc_{lang_key}'], t['category']: info['categorie'],
                t['cause']: info[f'cause_{lang_key}'], t['solution']: info[f'solution_{lang_key}'],
                t['price']: info['prix'], t['severity']: severity, t['status']: t['known'] if is_known else t['category_detected']
            })
            
        if not results: st.warning(t['no_dtc_found']); st.stop()
        
        res_df = pd.DataFrame(results)
        
        # Dashboard
        st.subheader(t['analysis_results'])
        c1, c2, c3 = st.columns(3)
        high = len(res_df[res_df[t['severity']] == 'High'])
        med = len(res_df[res_df[t['severity']] == 'Medium'])
        low = len(res_df[res_df[t['severity']] == 'Low'])
        
        c1.markdown(f'<div class="metric-card"><div style="font-size:2rem">🔴</div><div style="font-size:1.5rem;font-weight:bold;color:#ef5350">{high}</div><div>{t["high"]}</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-card"><div style="font-size:2rem">🟡</div><div style="font-size:1.5rem;font-weight:bold;color:#ffa726">{med}</div><div>{t["medium"]}</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="metric-card"><div style="font-size:2rem">🟢</div><div style="font-size:1.5rem;font-weight:bold;color:#66bb6a">{low}</div><div>{t["low"]}</div></div>', unsafe_allow_html=True)
        
        # Details
        st.subheader(t['details'])
        for _, r in res_df.iterrows():
            sev = r[t['severity']]
            box = "severity-high" if sev == 'High' else "severity-medium" if sev == 'Medium' else "severity-low"
            st.markdown(f"""
            <div class="info-box {box}">
                <h4>🔧 {r[t['dtc_code']]} <span style="font-size:0.8em;opacity:0.7">({r[t['status']]})</span></h4>
                <p><strong>{t['category']}:</strong> {r[t['category']]} | <strong>{t['price']}:</strong> {r[t['price']]}</p>
                <p><strong>Description:</strong> {r['Description']}</p>
                <p><strong>{t['cause']}:</strong> {r[t['cause']]}</p>
                <p><strong>{t['solution']}:</strong> {r[t['solution']]}</p>
            </div>
            """, unsafe_allow_html=True)
            
        csv = res_df.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(t['download'], csv, f"diag_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")
        
    except Exception as e:
        st.error(f"{t['error']} {str(e)}")

elif upload_method == "Texte/OCR":
    st.subheader("📸 Scanner/Copier-Coller")
    text_input = st.text_area("", height=200, placeholder="Ex: P0171 - System Too Lean")
    if st.button(t['analyze']):
        codes = re.findall(r'\b(P|B|C|U)\d{4}\b', text_input)
        if codes:
            st.success(f"✅ Codes: {', '.join(set([c[0]+c[1] for c in codes]))}")
        else: st.warning("⚠️ Aucun code trouvé")

elif upload_method == "Saisie Manuelle":
    if st.session_state.get('manual_dtc'):
        dtc = extract_valid_dtc(st.session_state.manual_dtc)
        if dtc:
            info, _ = get_dtc_info(dtc)
            lang_key = 'ar' if st.session_state.language == 'ar' else 'fr'
            st.info(f"**{dtc}**: {info[f'desc_{lang_key}']}\n\n💰 {info['prix']}")

# Footer
st.markdown(f"<div style='text-align:center;padding:2rem;color:#666;'>{t['footer']}</div>", unsafe_allow_html=True)
