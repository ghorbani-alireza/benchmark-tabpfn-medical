import pandas as pd
import numpy as np


def load_echo_data(path):
	data1_path = path + "data1_echo_notes.pkl"
	df1 = pd.read_pickle(data1_path)

	# keep only rows with LV_systolic for target
	df_lv = df1.dropna(subset=["LV_systolic"]).copy()
	NORMAL_VALUES = [0, -1]
	TARGET = "y_LV_normal"
	df_lv[TARGET] = df_lv["LV_systolic"].apply(lambda x: 0 if x in NORMAL_VALUES else 1)

	# shuffle patient groups
	rng = np.random.default_rng(42)
	order = {sid: i for i, sid in enumerate(rng.permutation(df_lv["subject_id"].drop_duplicates().to_numpy()))}
	df_lv = (df_lv.assign(_ord=df_lv["subject_id"].map(order))
               		.sort_values(["_ord", "subject_id"], kind="stable")
               		.drop(columns="_ord").reset_index(drop=True))

	# groups for GroupKFold
	groups = df_lv["subject_id"].reset_index(drop=True)

	# columns to exclude
	drop_cols = [
    		"LV_systolic", "text", "row_id", "subject_id", "hadm_id",
    		"category", "chartdate", "date_time", "test",
    		"contrast", "technical_quality", "doppler", TARGET
	]
	df_clean = df_lv.drop(columns=drop_cols, errors="ignore").copy()

	# handle categorical
	num = df_clean.select_dtypes(include=["number", "bool"])
	cat = df_clean.select_dtypes(include=["object", "category"])
	high_card = [c for c in cat.columns if cat[c].nunique() > 30]
	cat_small = cat.drop(columns=high_card, errors="ignore")

	X_cat = pd.get_dummies(cat_small, drop_first=True, dummy_na=True)

	# combine
	X_df1 = pd.concat([num.reset_index(drop=True), X_cat.reset_index(drop=True)], axis=1)
	const_cols = [c for c in X_df1.columns if X_df1[c].nunique(dropna=True) <= 1]
	X_df1 = X_df1.drop(columns=const_cols, errors="ignore").fillna(0.0).astype("float32")

	y_df1 = df_lv[TARGET].astype(int).reset_index(drop=True)

	return X_df1, y_df1, groups
	pass



def load_glucose_data(path):

	data2_path = path + "data2_blood_glucose_management.pkl"
	df2 = pd.read_pickle(data2_path)

	df = df2.copy()

	# parse timestamps
	for c in ["TIMER","STARTTIME","ENDTIME","GLCTIMER","GLCTIMER_AL"]:
		if c in df.columns:
			df[c] = pd.to_datetime(df[c], errors="coerce")

	# insulin events / glucose readings sort per patient
	df_ins = df.loc[df["INPUT"].notna()].dropna(subset=["SUBJECT_ID","STARTTIME"]).sort_values(["SUBJECT_ID","STARTTIME"]).reset_index(drop=True)
	df_glc = df.loc[df["GLC"].notna()].dropna(subset=["SUBJECT_ID","GLCTIMER"]).sort_values(["SUBJECT_ID","GLCTIMER"]).reset_index(drop=True)

	# pair insulin event with the first glucose reading
	def next_glc(ins, glc):
		pos = np.searchsorted(glc["GLCTIMER"].to_numpy(), ins["STARTTIME"].to_numpy(), side="right")
		val = np.full(len(ins), np.nan)
		ok = pos < len(glc)
		val[ok] = glc["GLC"].to_numpy(dtype=float)[pos[ok]]
		return ins.assign(GLC_next=val)

	glc_by_sub = dict(tuple(df_glc.groupby("SUBJECT_ID", sort=False)))
	pairs = [next_glc(ins, glc_by_sub[sid]) for sid, ins in df_ins.groupby("SUBJECT_ID", sort=False) if sid in glc_by_sub]
	df_merged2 = pd.concat(pairs, ignore_index=True).dropna(subset=["GLC_next"]).reset_index(drop=True)

	# binary target
	TARGET = "y_glucose_normal"
	df_merged2[TARGET] = df_merged2["GLC_next"].between(70, 180).astype("int8")

	# drop
	df_merged2 = df_merged2.loc[~(df_merged2["GLC_AL"].notna() & df_merged2["GLC_AL"].le(0))].reset_index(drop=True)

	# shuffle patient groups
	rng = np.random.default_rng(42)
	order = {sid: i for i, sid in enumerate(rng.permutation(df_merged2["SUBJECT_ID"].drop_duplicates().to_numpy()))}
	df_merged2 = (df_merged2.assign(_ord=df_merged2["SUBJECT_ID"].map(order))
                         .sort_values(["_ord","STARTTIME"], kind="stable")
                         .drop(columns="_ord").reset_index(drop=True))

	# predictors
	predictors = ["LOS_ICU_days","first_ICU_stay","INPUT","INPUT_HRS","INSULINTYPE","EVENT",
               "INFXSTOP","GLCSOURCE","GLC_AL","GLCSOURCE_AL","RULE"]

	df_merged2["RULE"] = df_merged2["RULE"].apply(lambda v: f"rule_{int(v)}" if pd.notna(v) else np.nan).astype("object")
	for c in ["INSULINTYPE","EVENT","GLCSOURCE","GLCSOURCE_AL","RULE"]:
		df_merged2[c] = df_merged2[c].astype("category")

	# build raw X
	X_raw = df_merged2[predictors].reset_index(drop=True)

	# clean X
	num = X_raw.select_dtypes(include=["number", "bool"])
	cat = X_raw.select_dtypes(include=["object", "category"])
	X_cat = pd.get_dummies(cat, drop_first=True, dummy_na=True)
	X_df2 = pd.concat([num.reset_index(drop=True), X_cat.reset_index(drop=True)], axis=1)
	const_cols = [c for c in X_df2.columns if X_df2[c].nunique(dropna=True) <= 1]
	X_df2 = X_df2.drop(columns=const_cols).fillna(0.0).astype("float32")

	#  y & groups
	y_df2 = df_merged2[TARGET].astype(int).reset_index(drop=True)
	groups = df_merged2["SUBJECT_ID"].reset_index(drop=True)

	return X_df2, y_df2, groups
	pass





def load_blood_gas_data(path):
	data3_path = path + "data3_blood_gas_oximetry.pkl" 
	df3 = pd.read_pickle(data3_path)

	# target
	THRESH_SAO2 = 94.0
	assert 'SaO2' in df3.columns, "SaO2 column missing"
	df_temp = df3.copy()
	df_temp['y'] = (df_temp['SaO2'] < THRESH_SAO2).astype(int)

	rng = np.random.default_rng(42)
	order = {sid: i for i, sid in enumerate(rng.permutation(df_temp['unique_subject_id'].drop_duplicates().to_numpy()))}
	df_temp = (df_temp.assign(_ord=df_temp['unique_subject_id'].map(order))
                 .sort_values(['_ord', 'unique_subject_id'], kind='stable')
                 .drop(columns='_ord')
                 .reset_index(drop=True))

	groups = df_temp['unique_subject_id']

	# drop
	abg_cols = ['SaO2', 'pO2', 'pCO2', 'pH', 'Carboxyhemoglobin', 'Methemoglobin']
	outcome_cols = ['in_hospital_mortality', 'los_hospital', 'los_ICU']
	time_cols = [c for c in df_temp.columns if 'timestamp' in c.lower() or c.startswith('datetime_')]
	future_cols = [c for c in df_temp.columns if c.startswith('sofa_future_')]
	id_cols = ['unique_subject_id', 'unique_hospital_admission_id', 'unique_icustay_id',
           'subject_id', 'hospital_admission_id', 'icustay_id', 'hospitalid']

	drop_cols = set(abg_cols + outcome_cols + time_cols + future_cols + id_cols + ['y'])  # 'y' is the target

	# X
	keep_df = df_temp.drop(columns=list(drop_cols), errors='ignore')
	num = keep_df.select_dtypes(include=['number'])
	cat = keep_df.select_dtypes(include=['object', 'category'])

	X_df3 = pd.concat([num, pd.get_dummies(cat, drop_first=True, dummy_na=True)], axis=1).astype('float32')

	const_cols = [c for c in X_df3.columns if X_df3[c].nunique(dropna=True) <= 1]
	X_df3 = X_df3.drop(columns=const_cols, errors='ignore').fillna(0.0)

	# y
	y_df3 = df_temp['y'].astype(int)

	return X_df3, y_df3, groups
	pass