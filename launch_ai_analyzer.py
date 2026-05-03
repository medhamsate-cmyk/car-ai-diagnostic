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
# FONCTION 1: Validation des codes DTC (Filtre les codes valides uniquement)
# ============================================================================
def extract_valid_dtc(value):
    """
    Extrait un code DTC valide OBD-II
    Retourne le code ou None
    """
    if not value:
        return None
    
    value = str(value).strip().upper()
    
    # Pattern: P/B/C/U + 4 chiffres
    match = re.search(r'\b(P|B|C|U)(\d{4})\b', value)
    
    if match:
        return match.group(0)
    
    return None

# ============================================================================
# FONCTION 2: Nettoyage des valeurs numériques
# ============================================================================
def clean_numeric(val):
    """
    Nettoie les valeurs: "800 rpm" → 800.0
    """
    if val is None or str(val).lower() in ['none', 'nan', 'error', '--', '', 'n/a']:
        return 0.0
    
    # Garder uniquement les chiffres et points
    clean = re.sub(r'[^\d.]', '', str(val))
    
    try:
        return float(clean) if clean else 0.0
    except ValueError:
        return 0.0

# ============================================================================
# FONCTION 3: Smart DTC Database (Tous les codes + catégories)
# ============================================================================
def get_dtc_info(dtc_code):
    """
    Analyse intelligente des codes DTC
    Retourne: info_dict, is_known
    """
    dtc = dtc_code.strip().upper()
    
    # Base de données spécifique
    specific_db = {
        'P0171': {
            'desc_fr': 'Système trop pauvre (Banque 1)',
            'desc_ar': 'نظام الوقود فقير جداً (بنك 1)',
            'cause_fr': 'Fuite d\'air, MAF défectueux, capteur O2',
            'cause_ar': 'تسريب هواء، حساس MAF، حساس O2',
            'solution_fr': 'Vérifier fuites admission, nettoyer MAF, tester O2',
            'solution_ar': 'تفقد تسريب الهواء، نظف MAF، افحص O2',
            'prix': '50-300€',
            'categorie': 'Injection'
        },
        'P0172': {
            'desc_fr': 'Système trop riche (Banque 1)',
            'desc_ar': 'نظام الوقود غني جداً (بنك 1)',
            'cause_fr': 'MAF, injecteurs qui fuient, régulateur pression',
            'cause_ar': 'حساس MAF، البخاخات، منظم الضغط',
            'solution_fr': 'Tester MAF, vérifier injecteurs',
            'solution_ar': 'افحص MAF، تفقد البخاخات',
            'prix': '100-400€',
            'categorie': 'Injection'
        },
        'P0300': {
            'desc_fr': 'Ratés d\'allumage multiples',
            'desc_ar': 'احتراق عشوائي متعدد',
            'cause_fr': 'Bougies usées, bobines défectueuses',
            'cause_ar': 'البوجيات تالفة، الكويلات معطوبة',
            'solution_fr': 'Remplacer bougies, tester bobines',
            'solution_ar': 'بدل البوجيات، افحص الكويلات',
            'prix': '100-500€',
            'categorie': 'Allumage'
        },
        'P0301': {
            'desc_fr': 'Ratés cylindre 1',
            'desc_ar': 'احتراق الأسطوانة 1',
            'cause_fr': 'Bougie/bobine cylindre 1',
            'cause_ar': 'بوجية/كويل الأسطوانة 1',
            'solution_fr': 'Tester bougie et bobine cylindre 1',
            'solution_ar': 'افحص بوجية وكويل الأسطوانة 1',
            'prix': '80-350€',
            'categorie': 'Allumage'
        },
        'P0420': {
            'desc_fr': 'Efficacité catalyseur faible',
            'desc_ar': 'كفاءة الكاتاليزر منخفضة',
            'cause_fr': 'Catalyseur usé, capteurs O2',
            'cause_ar': 'الكاتاليزر تالف، حساسات الأكسجين',
            'solution_fr': 'Vérifier capteurs O2 avant/après',
            'solution_ar': 'تفقد حساسات O2',
            'prix': '200-1500€',
            'categorie': 'Échappement'
        },
        'P0128': {
            'desc_fr': 'Thermostat température basse',
            'desc_ar': 'الترموستات حرارة منخفضة',
            'cause_fr': 'Thermostat bloqué ouvert',
            'cause_ar': 'الترموستات عالق في وضع المفتوح',
            'solution_fr': 'Remplacer thermostat',
            'solution_ar': 'استبدل الترموستات',
            'prix': '50-150€',
            'categorie': 'Refroidissement'
        },
        'P0101': {
            'desc_fr': 'Capteur MAF - Plage/performance',
            'desc_ar': 'حساس MAF - خارج النطاق',
            'cause_fr': 'MAF sale, fuite admission',
            'cause_ar': 'حساس MAF وسخ، تسريب هواء',
            'solution_fr': 'Nettoyer MAF, vérifier fuites',
            'solution_ar': 'نظف MAF، تفقد التسريبات',
            'prix': '80-300€',
            'categorie': 'Admission'
        },
        'P0442': {
            'desc_fr': 'Petite fuite EVAP',
            'desc_ar': 'تسريب صغير في نظام EVAP',
            'cause_fr': 'Bouchon carburant mal serré',
            'cause_ar': 'غطاء البنزين غير محكم',
            'solution_fr': 'Serrer/remplacer bouchon',
            'solution_ar': 'أحكم غطاء البنزين',
            'prix': '30-200€',
            'categorie': 'EVAP'
        },
    }
    
    # Si code connu
    if dtc in specific_db:
        return specific_db[dtc], True
    
    # Si code INCONNU → Détection par catégorie
    category_rules = {
        'P01': {
            'desc_fr': f'Problème injection/admission ({dtc})',
            'desc_ar': f'مشكل في الحقن أو الدخول ({dtc})',
            'cause_fr': 'Système fuel/air, capteurs MAF/O2, injecteurs',
            'cause_ar': 'نظام الوقود/الهواء، الحساسات، البخاخات',
            'solution_fr': 'Vérifier circuit fuel, capteurs, fuites',
            'solution_ar': 'تفقد دائرة الوقود، الحساسات، التسريبات',
            'prix': '80-400€',
            'categorie': 'Injection/Admission'
        },
        'P02': {
            'desc_fr': f'Problème circuit injecteur ({dtc})',
            'desc_ar': f'مشكل في دائرة البخاخات ({dtc})',
            'cause_fr': 'Injecteur, câblage, connecteurs',
            'cause_ar': 'بخاخ، أسلاك، موصلات',
            'solution_fr': 'Tester injecteur, vérifier câblage',
            'solution_ar': 'افحص البخاخ، تفقد الأسلاك',
            'prix': '100-500€',
            'categorie': 'Injecteurs'
        },
        'P03': {
            'desc_fr': f'Problème allumage/ratés ({dtc})',
            'desc_ar': f'مشكل في الإشعال/الاحتراق ({dtc})',
            'cause_fr': 'Bougies, bobines, fils bougies',
            'cause_ar': 'البوجيات، الكويلات، أسلاك البوجي',
            'solution_fr': 'Tester système allumage, compression',
            'solution_ar': 'افحص نظام الإشعال، الكومبراس',
            'prix': '80-450€',
            'categorie': 'Allumage'
        },
        'P04': {
            'desc_fr': f'Problème antipollution ({dtc})',
            'desc_ar': f'مشكل في نظام مكافحة التلوث ({dtc})',
            'cause_fr': 'Catalyseur, EGR, EVAP, O2',
            'cause_ar': 'الكاتاليزر، صمام EGR، نظام EVAP',
            'solution_fr': 'Vérifier système échappement/EGR',
            'solution_ar': 'تفقد نظام العادم/الصمامات',
            'prix': '100-1500€',
            'categorie': 'Antipollution'
        },
        'P05': {
            'desc_fr': f'Problème vitesse/régime ({dtc})',
            'desc_ar': f'مشكل في السرعة/الدوران ({dtc})',
            'cause_fr': 'Capteur vitesse, régulateur idle',
            'cause_ar': 'حساس السرعة، منظم الخمول',
            'solution_fr': 'Tester capteurs vitesse/TPS',
            'solution_ar': 'افحص حساسات السرعة',
            'prix': '50-300€',
            'categorie': 'Vitesse/Régime'
        },
        'P06': {
            'desc_fr': f'Problème calculateur/électronique ({dtc})',
            'desc_ar': f'مشكل في الكمبيوتر/الإلكترونيات ({dtc})',
            'cause_fr': 'ECU, câblage, relais, fusibles',
            'cause_ar': 'الكمبيوتر، الأسلاك، المرحلات، الفيوزات',
            'solution_fr': 'Vérifier ECU, connexions électriques',
            'solution_ar': 'تفقد الكمبيوتر، التوصيلات الكهربائية',
            'prix': '100-1000€',
            'categorie': 'Électronique'
        },
        'P07': {
            'desc_fr': f'Problème transmission ({dtc})',
            'desc_ar': f'مشكل في علبة السرعة ({dtc})',
            'cause_fr': 'Boîte auto, capteurs transmission',
            'cause_ar': 'علبة السرعة الأوتوماتيكية، الحساسات',
            'solution_fr': 'Diagnostiquer boîte, niveau huile',
            'solution_ar': 'شخص علبة السرعة، مستوى الزيت',
            'prix': '200-2000€',
            'categorie': 'Transmission'
        },
        'P08': {
            'desc_fr': f'Problème transmission ({dtc})',
            'desc_ar': f'مشكل في علبة السرعة ({dtc})',
            'cause_fr': 'Boîte, embrayage, différentiel',
            'cause_ar': 'علبة السرعة، القابض، التفاضل',
            'solution_fr': 'Vérifier transmission mécanique',
            'solution_ar': 'تفقد علبة السرعة الميكانيكية',
            'prix': '150-1500€',
            'categorie': 'Transmission'
        },
        'B': {
            'desc_fr': f'Problème carrosserie ({dtc})',
            'desc_ar': f'مشكل في الهيكل ({dtc})',
            'cause_fr': 'Airbag, ceintures, habitacle',
            'cause_ar': 'الوسادة الهوائية، الأحزمة، المقصورة',
            'solution_fr': 'Vérifier système airbag/carrosserie',
            'solution_ar': 'تفقد نظام الوسائد/الهيكل',
            'prix': '100-800€',
            'categorie': 'Carrosserie'
        },
        'C': {
            'desc_fr': f'Problème châssis ({dtc})',
            'desc_ar': f'مشكل في الشاسيه ({dtc})',
            'cause_fr': 'ABS, ESP, suspension, freins',
            'cause_ar': 'نظام ABS، الثبات، التعليق، الفرامل',
            'solution_fr': 'Diagnostiquer freins/suspension',
            'solution_ar': 'شخص الفرامل/التعليق',
            'prix': '100-1000€',
            'categorie': 'Châssis'
        },
        'U': {
            'desc_fr': f'Problème réseau communication ({dtc})',
            'desc_ar': f'مشكل في شبكة الاتصال ({dtc})',
            'cause_fr': 'CAN Bus, communication modules',
            'cause_ar': 'شبكة CAN، الاتصال بين الوحدات',
            'solution_fr': 'Vérifier réseau CAN, connexions',
            'solution_ar': 'تفقد شبكة CAN، التوصيلات',
            'prix': '150-600€',
            'categorie': 'Réseau'
        },
    }
    
    # Chercher catégorie
    for prefix, info in category_rules.items():
        if dtc.startswith(prefix):
            return info, False
    
    # Default
    return {
        'desc_fr': f'Code {dtc} non répertorié',
        'desc_ar': f'الكود {dtc} غير مسجل',
        'cause_fr': 'Consulter manuel constructeur',
        'cause_ar': 'راجع دليل الصانع',
        'solution_fr': 'Diagnostic approfondi nécessaire',
        'solution_ar': 'التشخيص العميق ضروري',
        'prix': 'N/A',
        'categorie': 'Inconnue'
    }, False

# ============================================================================
# FONCTION 4: Entraînement modèle AI
# ============================================================================
@st.cache_resource
def train_model():
    data = {
        'DTC': ['P0171', 'P0300', 'P0420', 'P0128', 'P0101', 'P0442', 'P0172', 'P0301'],
        'RPM': [800, 2200, 1500, 750, 2800, 900, 2000, 2400],
        'Load': [15, 45, 35, 10, 60, 20, 40, 50],
        'Temp': [90, 95, 105, 70, 110, 88, 92, 98],
        'Severity': ['Medium', 'High', 'Medium', 'Low', 'High', 'Low', 'Medium', 'High']
    }
    df = pd.DataFrame(data)
    le = LabelEncoder()
    df['DTC_Enc'] = le.fit_transform(df['DTC'])
    X = df[['DTC_Enc', 'RPM', 'Load', 'Temp']]
    y = df['Severity']
    model = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42)
    model.fit(X, y)
    return model, le

# ============================================================================
# INTERFACE STREAMLIT
# ============================================================================

# Sélecteur langue
with st.sidebar:
    lang_choice = st.selectbox("🌐 Language / اللغة", ["Français 🇫🇷", "العربية 🇲🇦"])
    st.session_state.language = 'ar' if 'العربية' in lang_choice else 'fr'

t = TRANSLATIONS[st.session_state.language]

# CSS
st.markdown("""
<style>
    h1 {
        text-align: center;
        background: linear-gradient(90deg, #1e3a8a, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 20px;
    }
    .success-box {
        background-color: #d1fae5;
        border-left: 5px solid #10b981;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
    .warning-box {
        background-color: #fef3c7;
        border-left: 5px solid #f59e0b;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
    .error-box {
        background-color: #fee2e2;
        border-left: 5px solid #ef4444;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
    .info-box {
        background-color: #dbeafe;
        border-left: 5px solid #3b82f6;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

st.title(t['title'])

# Sidebar
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
    if search:
        valid_search = extract_valid_dtc(search)
        if valid_search:
            info, known = get_dtc_info(valid_search)
            lang_key = 'ar' if st.session_state.language == 'ar' else 'fr'
            desc_key = f'desc_{lang_key}'
            st.info(f"**{info[desc_key]}**\n\n💰 {info['prix']}\n\n📂 {info['categorie']}")

# ============================================================================
# TRAITEMENT CSV
# ============================================================================
if upload_method == t['csv'] and uploaded_file is not None:
    try:
        # Lecture fichier
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
        st.write(f"**{t['columns_detected']}**: {', '.join(df.columns)}")
        
        # Détection colonnes
        dtc_col = rpm_col = load_col = temp_col = None
        
        for col in df.columns:
            col_lower = col.lower().strip()
            if 'dtc' in col_lower or 'code' in col_lower or 'defaut' in col_lower:
                dtc_col = col
            if 'rpm' in col_lower:
                rpm_col = col
            if 'load' in col_lower or 'charge' in col_lower:
                load_col = col
            if 'temp' in col_lower or 'coolant' in col_lower:
                temp_col = col
        
        if dtc_col is None:
            st.error(t['dtc_col_not_found'])
            dtc_col = st.selectbox(t['select_dtc_col'], df.columns.tolist())
        
        # Nettoyage données
        if rpm_col:
            df['RPM'] = df[rpm_col].apply(clean_numeric)
        else:
            df['RPM'] = 0.0
        
        if load_col:
            df['Load'] = df[load_col].apply(clean_numeric)
        else:
            df['Load'] = 0.0
        
        if temp_col:
            df['Temp'] = df[temp_col].apply(clean_numeric)
        else:
            df['Temp'] = 0.0
        
        st.subheader(t['data_preview'])
        st.dataframe(df.head())
        
        # Analyse
        model, le = train_model()
        results = []
        lang_key = 'ar' if st.session_state.language == 'ar' else 'fr'
        
        for idx, row in df.iterrows():
            # 1. Récupérer valeur brute
            raw_value = str(row[dtc_col]).strip()
            
            # 2. Ignorer valeurs vides
            if not raw_value or len(raw_value) < 4:
                continue
            
            if raw_value.lower() in ['none', 'nan', 'null', '-', '', 'n/a']:
                continue
            
            # 3. Extraire code DTC valide
            dtc = extract_valid_dtc(raw_value)
            
            # 4. Si pas code valide → ignorer
            if dtc is None:
                continue
            
            # 5. Récupérer données
            rpm = float(df['RPM'].iloc[idx])
            load = float(df['Load'].iloc[idx])
            temp = float(df['Temp'].iloc[idx])
            
            # 6. Obtenir informations DTC
            info, is_known = get_dtc_info(dtc)
            
            # 7. Prédire sévérité
            try:
                dtc_enc = le.transform([dtc])[0] if dtc in le.classes_ else 0
            except:
                dtc_enc = 0
            
            severity = model.predict([[dtc_enc, rpm, load, temp]])[0]
            
            # 8. Préparer résultat
            desc_key = f'desc_{lang_key}'
            cause_key = f'cause_{lang_key}'
            solution_key = f'solution_{lang_key}'
            
            results.append({
                t['dtc_code']: dtc,
                'Description': info[desc_key],
                t['category']: info['categorie'],
                t['cause']: info[cause_key],
                t['solution']: info[solution_key],
                t['price']: info['prix'],
                t['severity']: severity,
                t['status']: t['known'] if is_known else t['category_detected']
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
        fig = px.pie(results_df, names=t['severity'], 
                     color=t['severity'],
                     color_discrete_map={'High': '#ef4444', 'Medium': '#f59e0b', 'Low': '#10b981'},
                     title=t['distribution'])
        st.plotly_chart(fig, use_container_width=True)
        
        # Détails
        st.subheader(t['details'])
        for idx, row in results_df.iterrows():
            sev = row[t['severity']]
            box_class = "error-box" if sev == 'High' else "warning-box" if sev == 'Medium' else "success-box"
            
            st.markdown(f"""
            <div class="{box_class}">
                <h4>🔧 {row[t['dtc_code']]} {row[t['status']]}</h4>
                <p><strong>{t['category']}:</strong> {row[t['category']]}</p>
                <p><strong>Description:</strong> {row['Description']}</p>
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
        st.exception(e)

# ============================================================================
# TRAITEMENT TEXTE/OCR
# ============================================================================
elif upload_method == t['text_ocr']:
    st.subheader("📸 Scanner/Copier-Coller")
    st.info(t['paste_text'])
    
    text_input = st.text_area("", height=200, 
                              placeholder="Ex: P0171 - System Too Lean\nRPM: 800\nLoad: 15%")
    
    if st.button(t['analyze']):
        # Extraire codes valides
        dtc_matches = re.findall(r'\b(P|B|C|U)\d{4}\b', text_input)
        dtc_codes = list(set([m[0] + text_input[text_input.find(m[0]+m[1]):text_input.find(m[0]+m[1])+5] 
                              for m in dtc_matches]))
        
        if dtc_codes:
            st.success(f"✅ {len(dtc_codes)} code(s) trouvé(s): {', '.join(dtc_codes)}")
            
            results = []
            lang_key = 'ar' if st.session_state.language == 'ar' else 'fr'
            
            for dtc in dtc_codes:
                info, is_known = get_dtc_info(dtc)
                desc_key = f'desc_{lang_key}'
                
                results.append({
                    t['dtc_code']: dtc,
                    'Description': info[desc_key],
                    t['status']: t['known'] if is_known else t['category_detected']
                })
            
            st.dataframe(pd.DataFrame(results))
        else:
            st.warning("⚠️ Aucun code DTC trouvé")

# ============================================================================
# SAISIE MANUELLE
# ============================================================================
elif upload_method == t['manual']:
    if st.session_state.get('manual_dtc'):
        dtc = st.session_state.manual_dtc.upper()
        valid_dtc = extract_valid_dtc(dtc)
        
        if valid_dtc:
            info, is_known = get_dtc_info(valid_dtc)
            lang_key = 'ar' if st.session_state.language == 'ar' else 'fr'
            desc_key = f'desc_{lang_key}'
            
            st.info(f"**{valid_dtc}**\n\n{info[desc_key]}\n\n💰 {info['prix']}\n\n📂 {info['categorie']}")
        else:
            st.error("❌ Code DTC invalide")

# Footer
st.markdown("---")
st.markdown(t['footer'])
