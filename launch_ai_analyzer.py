import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os

st.set_page_config(page_title="🔧 Analyseur AI Diagnostic Launch X431", layout="wide")

# 🧠 Charger/Entraîner le modèle une seule fois
@st.cache_resource
def train_model():
    # Données d'entraînement réalistes (DTC + Freeze Frame)
    data = {
        'DTC_Type': ['P0171', 'P0300', 'P0420', 'P0128', 'P0101', 'P0442', 'P0172', 'P0301', 'P0455', 'P0440'],
        'RPM': [800, 2200, 1500, 750, 2800, 900, 2000, 2400, 1100, 1000],
        'Load_%': [15, 45, 35, 10, 60, 20, 40, 50, 25, 18],
        'Temp_C': [90, 95, 105, 70, 110, 88, 92, 98, 85, 87],
        'Severity': ['Medium', 'High', 'Medium', 'Low', 'High', 'Low', 'Medium', 'High', 'Low', 'Medium']
    }
    df = pd.DataFrame(data)
    le = LabelEncoder()
    df['DTC_Type_Enc'] = le.fit_transform(df['DTC_Type'])
    X = df[['DTC_Type_Enc', 'RPM', 'Load_%', 'Temp_C']]
    y = df['Severity']
    model = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42)
    model.fit(X, y)
    return model, le

def predict_severity(model, le, dtc, rpm, load, temp):
    try:
        dtc_enc = le.transform([dtc])[0]
    except ValueError:
        dtc_enc = le.transform([le.classes_[0]])[0]
    return model.predict([[dtc_enc, rpm, load, temp]])[0]

# 📚 Base de données des codes DTC (Français)
DTC_DB = {
    'P0171': {
        'name': '🔧 Système trop pauvre (Banque 1)',
        'priority': 'Medium',
        'causes': 'Fuite de vide, capteur MAF, capteur O2, pression de carburant',
        'action': '✅ Vérifier les tuyaux d\'admission, nettoyer MAF, tester pression carburant'
    },
    'P0300': {
        'name': '🔥 Ratemés d\'allumage multiples/aléatoires',
        'priority': 'High',
        'causes': 'Bougies, bobines, injecteurs, perte de compression',
        'action': '✅ Inspecter système d\'allumage, test de compression, vérifier fuel trim'
    },
    'P0420': {
        'name': '⚠️ Efficacité du catalyseur inférieure au seuil',
        'priority': 'Medium',
        'causes': 'Catalyseur défectueux, capteurs O2, fuites échappement',
        'action': '✅ Vérifier capteurs O2 avant/après catalyseur, inspecter fuites'
    },
    'P0128': {
        'name': '🌡️ Thermostat de liquide de refroidissement',
        'priority': 'Low',
        'causes': 'Thermostat ouvert, niveau bas, capteur défectueux',
        'action': '✅ Remplacer thermostat, vérifier niveau liquide, tester capteur'
    },
    'P0101': {
        'name': '💨 Capteur MAF - Plage/Performance',
        'priority': 'High',
        'causes': 'MAF sale, problème câblage, fuite de vide, filtre à air bouché',
        'action': '✅ Nettoyer/remplacer MAF, vérifier câblage, inspecter admission'
    },
    'P0442': {
        'name': '💨 Petite fuite EVAP détectée',
        'priority': 'Low',
        'causes': 'Bouchon carburant mal serré, fuite dans les tuyaux',
        'action': '✅ Serrer bouchon carburant, vérifier tuyaux EVAP'
    },
    'P0172': {
        'name': '🔧 Système trop riche (Banque 1)',
        'priority': 'Medium',
        'causes': 'Capteur MAF, injecteurs qui fuient, pression carburant élevée',
        'action': '✅ Nettoyer MAF, vérifier injecteurs, tester régulateur pression'
    },
    'P0301': {
        'name': '🔥 Ratemés d\'allumage cylindre 1',
        'priority': 'High',
        'causes': 'Bougie cylindre 1, bobine cylindre 1, injecteur cylindre 1',
        'action': '✅ Remplacer bougie cylindre 1, tester bobine, nettoyer injecteur'
    },
    'P0455': {
        'name': '💨 Grande fuite EVAP détectée',
        'priority': 'Low',
        'causes': 'Bouchon carburant manquant/desserré, grosse fuite EVAP',
        'action': '✅ Vérifier/remplacer bouchon carburant, inspecter système EVAP'
    },
    'P0440': {
        'name': '💨 Système EVAP - Dysfonctionnement',
        'priority': 'Medium',
        'causes': 'Problème système EVAP, vanne purge, capteur pression',
        'action': '✅ Tester vanne purge EVAP, vérifier capteur, inspecter tuyaux'
    },
}

def get_recommendation(dtc):
    return DTC_DB.get(dtc.upper(), {
        'name': '⚠️ Code DTC inconnu',
        'priority': 'Medium',
        'causes': 'Nécessite un diagnostic manuel',
        'action': '✅ Consulter manuel de service, vérifier câblage, effacer code et retester'
    })

# 🖥️ Interface Streamlit (Français)
st.title("🔧 Analyseur AI Diagnostic Launch X431")
st.markdown("### 📊 Analyse intelligente des défauts automobiles")
st.markdown("Téléchargez le fichier CSV exporté depuis votre scanner Launch. L'IA analysera les codes DTC, prédira la sévérité et donnera des recommandations étape par étape.")

uploaded_file = st.file_uploader("📤 Télécharger le rapport CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    st.subheader("📋 Aperçu des données brutes")
    st.dataframe(df.head())

    # Vérifier les colonnes requises
    required_cols = ['DTC_Code', 'RPM', 'Load_%', 'Temp_C']
    if not all(col in df.columns for col in required_cols):
        st.error(f"❌ Colonnes manquantes. Attendu: {required_cols}")
        st.info("💡 Colonnes disponibles: " + ", ".join(df.columns.tolist()))
        st.stop()

    # Entraîner le modèle
    if st.session_state.get('model') is None:
        with st.spinner("🤖 Entraînement du modèle AI en cours..."):
            model, le = train_model()
            st.session_state['model'] = model
            st.session_state['le'] = le
        st.success("✅ Modèle AI prêt!")

    # Analyse AI
    st.subheader("🔍 Résultats de l'analyse AI")
    results = []
    for _, row in df.iterrows():
        dtc = str(row['DTC_Code']).strip()
        severity = predict_severity(st.session_state['model'], st.session_state['le'], dtc, row['RPM'], row['Load_%'], row['Temp_C'])
        rec = get_recommendation(dtc)
        results.append({
            'DTC': dtc,
            'Description': rec['name'],
            'Sévérité_AI': severity,
            'Causes_probables': rec['causes'],
            'Actions_recommandées': rec['action'],
            'RPM_figés': row['RPM'],
            'Charge_%': row['Load_%'],
            'Température_°C': row['Temp_C']
        })

    res_df = pd.DataFrame(results)
    st.dataframe(res_df)

    # Statistiques
    st.subheader("📈 Répartition par priorité")
    c1, c2, c3 = st.columns(3)
    
    high_count = len(res_df[res_df['Sévérité_AI'] == 'High'])
    medium_count = len(res_df[res_df['Sévérité_AI'] == 'Medium'])
    low_count = len(res_df[res_df['Sévérité_AI'] == 'Low'])
    
    c1.metric("🔴 Élevée", high_count)
    c2.metric("🟡 Moyenne", medium_count)
    c3.metric("🟢 Faible", low_count)

    # Download
    csv_download = res_df.to_csv(index=False, encoding='utf-8-sig')
    st.download_button(
        label="📥 Télécharger le rapport AI (CSV)",
        data=csv_download,
        file_name="rapport_diagnostic_ai.csv",
        mime="text/csv"
    )
    
    # Résumé
    st.subheader("📝 Résumé de l'analyse")
    st.info(f"""
    **Total des défauts analysés:** {len(res_df)}  
    **Défauts critiques:** {high_count}  
    **Recommandation:** {'Intervention immédiate requise!' if high_count > 0 else 'Planifier la maintenance'}
    """)

else:
    st.info("📂 Téléchargez un fichier CSV pour commencer. Format attendu: `DTC_Code`, `RPM`, `Load_%`, `Temp_C`")
    
    st.markdown("""
    ### 📝 Exemple de format CSV:
    ```csv
    DTC_Code,RPM,Load_%,Temp_C
    P0171,800,15,90
    P0300,2200,45,95
    P0420,1500,35,105
    ```
    
    ### 🔧 Comment exporter depuis Launch X431:
    1. Effectuez un diagnostic complet
    2. Allez dans "Rapport" ou "Export"
    3. Choisissez "Exporter en CSV"
    4. Téléchargez le fichier ici
    """)
    
    st.markdown("""
    ### 📱 Autres scanners compatibles:
    - ✅ Autel MaxiSys
    - ✅ ELM327 + Torque Pro
    - ✅ Delphi DS150E
    - ✅ Foxwell
    - ✅ Tout scanner OBD-II standard
    """)