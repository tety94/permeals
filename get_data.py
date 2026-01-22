import pandas as pd
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import StandardScaler

EXCEL_FILE_NAME = 'database.xlsx'

CSF_ANALYSIS_NAME = 'CSF_analysis_On_pALS.csv'

def get_sheet(sheet_name=None):
    df = pd.read_excel(EXCEL_FILE_NAME,  sheet_name=sheet_name)
    # df = pd.read_csv(EXCEL_FILE_NAME,  sep=',' , decimal=',')
    df = df[df['pt_code'].notna()]

    df['site_of_onset_BS'] = 0
    df.loc[df['site_of_onset'] == 1, 'site_of_onset_BS'] = 1 #spinale

    df['genetic_WT_mut'] = df['genetic_WT_mut'].map({'WT': 0, 'mut': 1}).astype(float)

    return df

def get_csf_sheet():
    df = pd.read_csv(CSF_ANALYSIS_NAME, sep=';', decimal=',')
    df['pt_code'] = df['pALS code']

    return df


def get_patients_at_T(df, T=0):
    df = df[df['time_point'] == f'T{T}']
    return df


def rgb_to_hex(r, g, b):
    return f'#{r:02x}{g:02x}{b:02x}'

def highlight_cells(val):
    try:
        val = float(val)
    except:
        return ''

    if val < -0.5:
        # rosso da scuro (-1) a chiaro (-0.5)
        intensity = int(255 * (1 + val) / 0.5)
        hex_color = rgb_to_hex(255, intensity, intensity)
        return f'background-color: {hex_color}'
    elif val > 0.5:
        # verde da chiaro (0.5) a scuro (1)
        intensity = int(255 * (1 - val) / 0.5)
        hex_color = rgb_to_hex(intensity, 255, intensity)
        return f'background-color: {hex_color}'
    else:
        return ''


# converte i valori per la MRC
def convert_mrc_values(val):
    if val in ['NV', 'NA']:
        return None
    val = str(val).strip()
    base = val.rstrip('+-')  # rimuove + o -
    base = float(base)
    if val.endswith('+'):
        return base + 0.25
    elif val.endswith('-'):
        return base - 0.25
    else:
        return base

def diff_months(row):
    if pd.isna(row['alsfrs_0_date']) or pd.isna(row['onset_date']):
        return pd.NA
    rd = relativedelta(row['alsfrs_0_date'], row['onset_date'])
    return rd.years * 12 + rd.months + rd.days / 30  # giorni come frazione di mese (facoltativo)

def add_delta(df_merged):
    df_merged['onset_date'] = pd.to_datetime(df_merged['onset_date_gg/mm/aaaa'])
    df_merged['alsfrs_0_date'] = pd.to_datetime(df_merged['ALSFRS_T0_ALSFRS_T0_date'])
    df_merged['birth_date'] = pd.to_datetime(df_merged['birth_date'])

    df_merged['onset_age'] = (df_merged['onset_date'] - df_merged['birth_date']).dt.days / 365.25
    df_merged['onset_age'] = round(df_merged['onset_age'], 1)

    for i in range(5):
        df_merged[f'alsfrs_T{i * 3}_tot'] = 0
        for j in range(1, 13):
            c = df_merged[f'ALSFRSr_{j}_T{i * 3}']
            c = c.replace([float('inf'), -float('inf')], pd.NA)
            c = c.fillna(0)  # oppure un altro valore di default
            c = c.astype(int)
            df_merged[f'alsfrs_T{i * 3}_tot'] += c


    df_merged['delay'] = df_merged.apply(diff_months, axis=1)
    df_merged['delta'] = (48 - df_merged['alsfrs_T0_tot']) / df_merged['delay']
    return df_merged

def create_dates(df):
    df['birth_date'] = pd.to_datetime(df['birth_date'])
    df['onset_date'] = pd.to_datetime(df['onset_date_gg/mm/aaaa'])
    df['onset_age'] = round((df['onset_date'] - df['birth_date']).dt.days / 365.25,1)

    return df

def create_weight(df):
    df['pre-morbid_bmi'] = df['pre-morbid_weight_kg'] / df['height_m'] ** 2
    df['diagnosis_bmi'] = df['weight_at_diagnosis_kg'] / df['height_m'] ** 2
    df['delta_bmi'] = df['pre-morbid_bmi'] - df['diagnosis_bmi']

    return df


def get_plasma_columns():
    return [
        'plasma_GFAP', 'plasma_NFL', 'plasma_tau', 'plasma_UCHL1', 'plasma_MMP-9', 'plasma_MCP-1',

        #vescicole extracellulari
        # 'EV_plasma_n_particles/ml', 'EV_plasma_Mean_size', 'EV_plasma_Median_size',
        # 'Activated platelets(%)', 'plasma_platelets/ml'

        #non usare
        #'EV_plasma_date_collection', 'EV_plasma_n_aliquote', 'EV_plasma_Albumin', 'EV_plasma_ApoA1',
    ]

def get_liquor_columns():
    return [
        'CSF_GFAP',  'CSF_MCP-1', 'CSF_NFL', 'CSF_tau', 'CSF_UCHL1', 'CSF_MMP-9',
    ]

def get_respiratory_columns():
    return [
        'ABG_pH', 'ABG_pCO2', 'ABG_pO2', 'ABG_HCO3', 'ABG_lact', 'ABG_Cl', 'ABG_Ca', 'FVC'
    ]

def get_ematic_columns():
    return [
        'C_WBC_10^9/L', 'C_RBC_10^12/L', 'C_Hb_g/dL', 'C_Plts_10^9/L', 'C_MCV_fL', 'C_neu_10^9/L',
        'C_lymph_10^9/L', 'C_mono_10^9/L', 'C_eos_10^9/L', 'C_bas_10^9/L', 'C_cholesterol_mg/dL',
        'C_HDL_mg/dL', 'C_triglycerides_mg/dL', 'C_CK_UI/L', 'C_creatinine', 'C_albumine', 'C_phospate',
        'C_chloride', 'C_urea', 'C_uric_acid', 'NLR', 'LDL'
    ]

def get_umn_score_columns():
    return [
        'Penn_TOT', 'MGH_TOT'
    ]

def get_staging_columns():
    return [
        'C_MiToS_TOT', 'King\'s',
    ]

def get_alsfrs_columns():
    return [
        'ALSFRSr_1', 'ALSFRSr_2', 'ALSFRSr_3', 'ALSFRSr_4', 'ALSFRSr_5', 'ALSFRSr_6', 'ALSFRSr_7', 'ALSFRSr_8',
        'ALSFRSr_9', 'ALSFRSr_10', 'ALSFRSr_11', 'ALSFRSr_12', 'ALSFRSr_TOT',
        'delta_ALSFRS_onset'
    ]

def get_mrc_columns():
    return [
        'C_MRC_neck_flex', 'C_MRC_neck_est', 'C_MRC_delt_R', 'C_MRC_delt_L', 'C_MRC_BB_R', 'C_MRC_BB_L', 'C_MRC_TB_R',
        'C_MRC_TB_L', 'C_MRC_wr_flex_R', 'C_MRC_wr_flex_L', 'C_MRC_wr_est_R', 'C_MRC_wr_est_L', 'C_MRC_fing_flex_R',
        'C_MRC_fing_flex_L', 'C_MRC_fing_est_R', 'C_MRC_fing_est_L', 'C_MRC_thumb_R', 'C_MRC_thumb_L', 'C_MRC_hip_flex_R',
        'C_MRC_hip_flex_L', 'C_MRC_leg_ext_R', 'C_MRC_leg_ext_L', 'C_MRC_leg_flex_R', 'C_MRC_leg_flex_L',
        'C_MRC_ankle_ext_R', 'C_MRC_ankle_ext_L', 'C_MRC_ankle_flex_R', 'C_MRC_ankle_flex_L', 'MRC_composite_score'
    ]

def get_clinical_data():
    return [
        'sex', 'onset_age', 'dgn_age', 'dgn_delay', 'pre-morbid_weight', 'pre-morbid_BMI', 'genetic_WT_mut',
        # 'phenotype', 'el_escorial_criteria', 'Strong_CAT',
        'delta_weight_pre_dgn', 'delta_BMI_pre_dgn',
        'C9orf72',
        'HC_ALS_AD'
    ]

def get_macsplex_columns():
    return [
        # 'CD13', 'CD24', 'CD29', 'CD31', 'CD36', 'CD38', 'CD45', 'CD49a', 'CD49e', 'CD49f', 'CD54',
        #  'CD81', 'CD106', 'CD133/1',  'CD140a', 'CD222 IGF2R', 'CX3CR1', 'EGFR',
        # 'Ganglioside GD2', 'Podoplanin', 'CD9', 'CD44', 'CD63', 'CD107a',
        # 'GLAST ACSA-1', 'O4',  'CD11b', 'CSPG4'

                                        "CD13", "CD24", "CD29", "CD31", "CD36", "CD38", "CD45",
        "CD49a", "CD49e", "CD49f", "CD54", "CD81", "CD106",
        "CD133/1", "CD140a", "CD222 IGF2R", "CX3CR1", "EGFR",
        "Ganglioside GD2", "Podoplanin", "CD9", "CD44", "CD63",
        "CD107a", "GLAST ACSA-1", "O4", "CD11b", "CSPG4"
    ];


def preprocess_data(df, features, scale=True):
    X_raw = df[features].copy()

    # Rimuovi colonne completamente nulle
    all_null_cols = X_raw.columns[X_raw.isnull().all()].tolist()
    if all_null_cols:
        print(f"⚠️ Colonne completamente vuote rimosse: {all_null_cols}")
        X_raw.drop(columns=all_null_cols, inplace=True)

    # Separa tipi di dato
    num_cols = X_raw.select_dtypes(include='number').columns
    cat_cols = X_raw.select_dtypes(exclude='number').columns

    print(f"Numeriche: {len(num_cols)}, Categoriche: {len(cat_cols)}")

    # Imputazione
    X_filled = X_raw.copy()

    # Numeriche → media
    if len(num_cols) > 0:
        X_filled[num_cols] = X_filled[num_cols].fillna(
            X_filled[num_cols].mean()
        )

    # Categoriche → moda
    for col in cat_cols:
        if X_filled[col].isna().any():
            mode = X_filled[col].mode(dropna=True)
            if not mode.empty:
                X_filled[col] = X_filled[col].fillna(mode.iloc[0])
            else:
                X_filled[col] = X_filled[col].fillna("missing")

    # Scaling SOLO sulle numeriche
    if scale and len(num_cols) > 0:
        scaler = StandardScaler()
        X_filled[num_cols] = scaler.fit_transform(X_filled[num_cols])

    return X_filled.values, df.loc[X_filled.index], X_filled.columns.tolist()


def add_median_split_column(df, column_name):
    median_value = df[column_name].median()
    new_col = f"{column_name}_median"
    print(f'La mediana di {new_col} è: {median_value}')
    df[new_col] = (df[column_name] > median_value).astype(int)
    return df


def get_merged_csf_data(df, csf_df):
    merged_df = df.merge(
        csf_df,
        on='pt_code',
        how='left'
    )
    return merged_df