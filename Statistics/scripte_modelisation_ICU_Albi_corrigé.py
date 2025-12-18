# =========================================================
# 📊 RÉGRESSION LINÉAIRE MULTIPLE - MODÉLISATION ICU ALBI
# Conforme à la méthodologie Bottyán & Unger (2003)
# =========================================================

import pandas as pd
import numpy as np
from scipy import stats
import sys
import subprocess

print("="*70)
print("📊 RÉGRESSION LINÉAIRE MULTIPLE - MODÉLISATION ICU ALBI")
print("="*70)

# =========================================================
# VÉRIFICATION ET INSTALLATION DE SCIKIT-LEARN
# =========================================================
print("\n🔍 Vérification des dépendances...")

try:
    import sklearn
    print("✅ scikit-learn déjà installé")
    has_sklearn = True
except ImportError:
    print("⚠️ scikit-learn non trouvé")
    print("📥 Installation de scikit-learn...")
    try:
        python_exe = sys.executable
        print(f"   Python : {python_exe}")
        subprocess.check_call([python_exe, "-m", "pip", "install", "scikit-learn"])
        print("✅ scikit-learn installé avec succès !")
        import sklearn
        has_sklearn = True
    except Exception as e:
        print(f"❌ Erreur installation : {e}")
        print("🔌 Solution alternative : régression avec numpy/scipy")
        has_sklearn = False

# =========================================================
# FONCTIONS DE RÉGRESSION PERSONNALISÉES
# =========================================================

class SimpleLinearRegression:
    """Régression linéaire simple avec numpy"""
    
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
    
    def fit(self, X, y):
        """Ajustement par moindres carrés ordinaires"""
        X_with_intercept = np.column_stack([np.ones(len(X)), X])
        
        try:
            beta = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
            self.intercept_ = beta[0]
            self.coef_ = beta[1:]
        except np.linalg.LinAlgError:
            print("⚠️ Matrice singulière, utilisation de pseudo-inverse")
            beta = np.linalg.pinv(X_with_intercept) @ y
            self.intercept_ = beta[0]
            self.coef_ = beta[1:]
        
        return self
    
    def predict(self, X):
        """Prédiction"""
        return self.intercept_ + X @ self.coef_

def standardize(X):
    """Standardisation z-score"""
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1
    return (X - mean) / std, mean, std

def r2_score_custom(y_true, y_pred):
    """Coefficient de détermination R²"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

def rmse_custom(y_true, y_pred):
    """RMSE"""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def mae_custom(y_true, y_pred):
    """MAE"""
    return np.mean(np.abs(y_true - y_pred))

def cross_validation_score(X, y, k=5, seed=42):
    """Validation croisée k-fold manuelle"""
    np.random.seed(seed)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    
    fold_size = len(X) // k
    scores_r2 = []
    scores_rmse = []
    scores_mae = []
    
    for i in range(k):
        test_start = i * fold_size
        test_end = (i + 1) * fold_size if i < k - 1 else len(X)
        test_idx = indices[test_start:test_end]
        train_idx = np.concatenate([indices[:test_start], indices[test_end:]])
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        X_train_scaled, mean_train, std_train = standardize(X_train)
        X_test_scaled = (X_test - mean_train) / std_train
        
        model = SimpleLinearRegression()
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        scores_r2.append(r2_score_custom(y_test, y_pred))
        scores_rmse.append(rmse_custom(y_test, y_pred))
        scores_mae.append(mae_custom(y_test, y_pred))
    
    return np.mean(scores_r2), np.mean(scores_rmse), np.mean(scores_mae)

# =========================================================
# ÉTAPE 1 : CHARGER LES DONNÉES
# =========================================================
print("\n" + "="*70)
print("1️⃣ CHARGEMENT DES DONNÉES")
print("="*70)

csv_path = 'C:/master SIGMA/projet_atelier/albi/indicateurs_albi_icu.csv'

try:
    df = pd.read_csv(csv_path)
    print(f"✅ {len(df)} mailles chargées")
    print(f"📊 Colonnes disponibles : {len(df.columns)}")
    print(f"\n   Colonnes : {', '.join(df.columns[:10])}...")
except FileNotFoundError:
    print(f"❌ Fichier non trouvé : {csv_path}")
    print("\n⚠️ SOLUTION :")
    print("   1. Vérifiez que le fichier 'indicateurs_albi_icu.csv' existe")
    print("   2. Ce fichier doit contenir la colonne 'ICU_i' (agrégation 1km)")
    print("   3. Exécutez d'abord le script de calcul des indicateurs ICU")
    sys.exit(1)

# =========================================================
# ÉTAPE 2 : VÉRIFICATION DE LA VARIABLE DÉPENDANTE
# =========================================================
print("\n" + "="*70)
print("2️⃣ PRÉPARATION DE LA VARIABLE DÉPENDANTE")
print("="*70)

# Variable cible : ICU_i (intensité ICU agrégée)
target = 'ICU_i'

# Vérifier si ICU_i existe
if target not in df.columns:
    print(f"❌ La colonne '{target}' n'existe pas !")
    print(f"\n📋 Colonnes disponibles dans le CSV :")
    for col in df.columns:
        print(f"   • {col}")
    print("\n⚠️ ACTIONS REQUISES :")
    print("   1. Vérifiez que ICU_i a bien été calculé")
    print("   2. ICU_i = moyenne des pixels ICU 30m par carreau 1km")
    print("   3. Relancez le script de calcul des indicateurs si nécessaire")
    sys.exit(1)

print(f"✅ Variable dépendante : {target}")
print(f"   Description : Intensité de l'Îlot de Chaleur Urbain (°C)")
print(f"   Formule : ICU_i = LST_i - T_ref")

# Nettoyer les valeurs manquantes
df_clean = df.dropna(subset=[target])
print(f"\n✅ {len(df_clean)}/{len(df)} mailles avec ICU_i valide")

if len(df_clean) < 30:
    print(f"\n⚠️ ATTENTION : Seulement {len(df_clean)} observations valides")
    print("   Un minimum de 30-50 observations est recommandé pour la régression")

# Statistiques descriptives de ICU_i
print(f"\n📊 STATISTIQUES DE ICU_i :")
print(f"   Minimum    : {df_clean[target].min():.2f}°C")
print(f"   Maximum    : {df_clean[target].max():.2f}°C")
print(f"   Moyenne    : {df_clean[target].mean():.2f}°C")
print(f"   Médiane    : {df_clean[target].median():.2f}°C")
print(f"   Écart-type : {df_clean[target].std():.2f}°C")

# Test de normalité
skewness_icu = stats.skew(df_clean[target].dropna())
print(f"\n   Skewness   : {skewness_icu:.3f}", end="")
if abs(skewness_icu) > 1:
    print(" → Distribution asymétrique")
    print("   ⚠️ Considérer une transformation si |skew| > 1")
else:
    print(" → Distribution symétrique ✓")

# =========================================================
# ÉTAPE 3 : SÉLECTION DES PRÉDICTEURS
# =========================================================
print("\n" + "="*70)
print("3️⃣ SÉLECTION DES PRÉDICTEURS")
print("="*70)

# Variables morphologiques de base
variables_base = ['Bi', 'Hi', 'Wi', 'Vi', 'Ai']
variables_demo = ['Di', 'T65i', 'T14i']

# Rayons pour variables pondérées
rayons = [250, 500, 1000, 2000]

# Dictionnaire des modèles à tester
modeles = {}

# MODÈLE 1 : Variables locales seules
print("\n📌 Configuration des modèles :")
predicteurs_local = [v for v in variables_base if v in df_clean.columns]
if predicteurs_local:
    modeles['Local'] = predicteurs_local
    print(f"   ✓ Modèle Local : {len(predicteurs_local)} variables")

# MODÈLE 2 : Variables locales + démographiques
predicteurs_local_demo = [v for v in variables_base + variables_demo if v in df_clean.columns]
if len(predicteurs_local_demo) > len(predicteurs_local):
    modeles['Local+Demo'] = predicteurs_local_demo
    print(f"   ✓ Modèle Local+Demo : {len(predicteurs_local_demo)} variables")

# MODÈLE 3 : Variables pondérées par rayon
for L in rayons:
    predicteurs_L = [f'{var}_L{L}' for var in variables_base]
    predicteurs_L_disponibles = [p for p in predicteurs_L if p in df_clean.columns]
    if predicteurs_L_disponibles:
        modeles[f'L{L}m'] = predicteurs_L_disponibles
        print(f"   ✓ Modèle L{L}m : {len(predicteurs_L_disponibles)} variables")

# MODÈLE 4 : Mix local + pondéré L500
predicteurs_mix = variables_base + [f'{var}_L500' for var in variables_base]
predicteurs_mix_disponibles = [p for p in predicteurs_mix if p in df_clean.columns]
if len(predicteurs_mix_disponibles) > len(predicteurs_local):
    modeles['Mix_L500'] = predicteurs_mix_disponibles
    print(f"   ✓ Modèle Mix_L500 : {len(predicteurs_mix_disponibles)} variables")

# MODÈLE 5 : Mix complet (local + L500 + demo)
if 'Di' in df_clean.columns:
    predicteurs_complet = predicteurs_mix + variables_demo
    predicteurs_complet_disponibles = [p for p in predicteurs_complet if p in df_clean.columns]
    if len(predicteurs_complet_disponibles) > len(predicteurs_mix_disponibles):
        modeles['Complet'] = predicteurs_complet_disponibles
        print(f"   ✓ Modèle Complet : {len(predicteurs_complet_disponibles)} variables")

print(f"\n✅ Total : {len(modeles)} modèles configurés")

if len(modeles) == 0:
    print("\n❌ ERREUR : Aucun prédicteur disponible !")
    print("   Vérifiez que les colonnes Bi, Hi, Wi, Vi, Ai existent")
    sys.exit(1)

# =========================================================
# ÉTAPE 4 : RÉGRESSION LINÉAIRE MULTIPLE
# =========================================================
print("\n" + "="*70)
print("4️⃣ RÉGRESSION LINÉAIRE MULTIPLE")
print("="*70)

resultats = []

if has_sklearn:
    from sklearn.linear_model import LinearRegression, LassoCV, ElasticNetCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import KFold, cross_val_score
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    
    print("✅ Utilisation de scikit-learn\n")
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for nom_modele, predicteurs in modeles.items():
        print(f"\n{'='*60}")
        print(f"🔬 MODÈLE : {nom_modele}")
        print(f"{'='*60}")
        print(f"Variables : {', '.join(predicteurs)}")
        
        # Préparer les données
        X = df_clean[predicteurs].copy()
        y = df_clean[target].copy()
        
        # Supprimer les lignes avec NaN
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]
        
        if len(X) < 10:
            print(f"⚠️ Pas assez de données ({len(X)} observations)")
            continue
        
        print(f"\n📊 Nombre d'observations : {len(X)}")
        
        # Standardisation
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 1. RÉGRESSION LINÉAIRE CLASSIQUE (OLS)
        print(f"\n1️⃣ Régression Linéaire (OLS)")
        lr = LinearRegression()
        lr.fit(X_scaled, y)
        y_pred = lr.predict(X_scaled)
        
        r2_train = r2_score(y, y_pred)
        rmse_train = np.sqrt(mean_squared_error(y, y_pred))
        mae_train = mean_absolute_error(y, y_pred)
        
        # Validation croisée
        cv_r2 = cross_val_score(lr, X_scaled, y, cv=kfold, scoring='r2').mean()
        cv_rmse = -cross_val_score(lr, X_scaled, y, cv=kfold, 
                                    scoring='neg_root_mean_squared_error').mean()
        cv_mae = -cross_val_score(lr, X_scaled, y, cv=kfold, 
                                   scoring='neg_mean_absolute_error').mean()
        
        print(f"   Train - R²: {r2_train:.4f}, RMSE: {rmse_train:.4f}°C, MAE: {mae_train:.4f}°C")
        print(f"   CV    - R²: {cv_r2:.4f}, RMSE: {cv_rmse:.4f}°C, MAE: {cv_mae:.4f}°C")
        
        # Top 3 variables
        coeffs = pd.DataFrame({
            'Variable': predicteurs,
            'Coefficient': lr.coef_
        }).sort_values('Coefficient', key=abs, ascending=False)
        
        print(f"\n   🔍 Top 3 variables influentes :")
        for idx, (i, row) in enumerate(coeffs.head(3).iterrows(), 1):
            effet = "↑ réchauffe" if row['Coefficient'] > 0 else "↓ refroidit"
            print(f"      {idx}. {row['Variable']:<15} : {row['Coefficient']:>7.4f} {effet}")
        
        resultats.append({
            'Modèle': nom_modele,
            'Méthode': 'OLS',
            'N_vars': len(predicteurs),
            'R2': r2_train,
            'R2_CV': cv_r2,
            'RMSE': rmse_train,
            'RMSE_CV': cv_rmse,
            'MAE': mae_train,
            'MAE_CV': cv_mae,
            'Overfit': r2_train - cv_r2
        })
        
        # 2. RÉGRESSION LASSO (si plus de 5 variables)
        if len(predicteurs) > 5:
            print(f"\n2️⃣ Régression Lasso (L1)")
            try:
                lasso = LassoCV(cv=kfold, random_state=42, max_iter=5000)
                lasso.fit(X_scaled, y)
                y_pred_lasso = lasso.predict(X_scaled)
                
                r2_lasso = r2_score(y, y_pred_lasso)
                rmse_lasso = np.sqrt(mean_squared_error(y, y_pred_lasso))
                mae_lasso = mean_absolute_error(y, y_pred_lasso)
                
                # Variables sélectionnées (coef non nuls)
                n_selected = np.sum(lasso.coef_ != 0)
                
                print(f"   Alpha optimal : {lasso.alpha_:.6f}")
                print(f"   Variables sélectionnées : {n_selected}/{len(predicteurs)}")
                print(f"   Train - R²: {r2_lasso:.4f}, RMSE: {rmse_lasso:.4f}°C")
                
                resultats.append({
                    'Modèle': nom_modele,
                    'Méthode': 'Lasso',
                    'N_vars': n_selected,
                    'R2': r2_lasso,
                    'R2_CV': np.nan,  # Déjà en CV
                    'RMSE': rmse_lasso,
                    'RMSE_CV': np.nan,
                    'MAE': mae_lasso,
                    'MAE_CV': np.nan,
                    'Overfit': 0
                })
            except Exception as e:
                print(f"   ⚠️ Erreur Lasso : {e}")

else:
    # Version numpy/scipy
    print("✅ Utilisation de numpy/scipy\n")
    
    for nom_modele, predicteurs in modeles.items():
        print(f"\n{'='*60}")
        print(f"🔬 MODÈLE : {nom_modele}")
        print(f"{'='*60}")
        print(f"Variables : {', '.join(predicteurs)}")
        
        X = df_clean[predicteurs].values
        y = df_clean[target].values
        
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X = X[mask]
        y = y[mask]
        
        if len(X) < 10:
            print(f"⚠️ Pas assez de données ({len(X)} observations)")
            continue
        
        print(f"\n📊 Nombre d'observations : {len(X)}")
        
        # Standardisation
        X_scaled, mean_X, std_X = standardize(X)
        
        # Régression
        model = SimpleLinearRegression()
        model.fit(X_scaled, y)
        y_pred = model.predict(X_scaled)
        
        # Métriques
        r2 = r2_score_custom(y, y_pred)
        rmse = rmse_custom(y, y_pred)
        mae = mae_custom(y, y_pred)
        
        # Validation croisée
        cv_r2, cv_rmse, cv_mae = cross_validation_score(X, y, k=5)
        
        print(f"\n   Train - R²: {r2:.4f}, RMSE: {rmse:.4f}°C, MAE: {mae:.4f}°C")
        print(f"   CV    - R²: {cv_r2:.4f}, RMSE: {cv_rmse:.4f}°C, MAE: {cv_mae:.4f}°C")
        
        # Top 3 variables
        coeffs_sorted = sorted(zip(predicteurs, model.coef_), 
                              key=lambda x: abs(x[1]), reverse=True)
        
        print(f"\n   🔍 Top 3 variables influentes :")
        for idx, (var, coef) in enumerate(coeffs_sorted[:3], 1):
            effet = "↑ réchauffe" if coef > 0 else "↓ refroidit"
            print(f"      {idx}. {var:<15} : {coef:>7.4f} {effet}")
        
        resultats.append({
            'Modèle': nom_modele,
            'Méthode': 'OLS',
            'N_vars': len(predicteurs),
            'R2': r2,
            'R2_CV': cv_r2,
            'RMSE': rmse,
            'RMSE_CV': cv_rmse,
            'MAE': mae,
            'MAE_CV': cv_mae,
            'Overfit': r2 - cv_r2
        })

# =========================================================
# ÉTAPE 5 : COMPARAISON DES MODÈLES
# =========================================================
print("\n" + "="*70)
print("5️⃣ COMPARAISON DES MODÈLES")
print("="*70)

df_resultats = pd.DataFrame(resultats)
df_resultats_sorted = df_resultats.sort_values('R2_CV', ascending=False)

print("\n📊 CLASSEMENT DES MODÈLES (par R² CV) :\n")
print("="*95)
print(f"{'Modèle':<15} {'Méthode':<10} {'N':<5} {'R²':<8} {'R²_CV':<8} {'RMSE_CV':<10} {'MAE_CV':<10} {'Overfit':<8}")
print("="*95)

for _, row in df_resultats_sorted.iterrows():
    overfit_str = f"{row['Overfit']:.4f}" if not np.isnan(row['Overfit']) else "N/A"
    r2_cv_str = f"{row['R2_CV']:.4f}" if not np.isnan(row['R2_CV']) else "N/A"
    rmse_cv_str = f"{row['RMSE_CV']:.4f}" if not np.isnan(row['RMSE_CV']) else "N/A"
    mae_cv_str = f"{row['MAE_CV']:.4f}" if not np.isnan(row['MAE_CV']) else "N/A"
    
    print(f"{row['Modèle']:<15} {row['Méthode']:<10} {row['N_vars']:<5.0f} "
          f"{row['R2']:<8.4f} {r2_cv_str:<8} {rmse_cv_str:<10} {mae_cv_str:<10} {overfit_str:<8}")

print("="*95)

# Sélection du meilleur modèle
best_model = df_resultats_sorted.iloc[0]
print(f"\n🏆 MEILLEUR MODÈLE : {best_model['Modèle']} ({best_model['Méthode']})")
print(f"\n   📈 Performances :")
print(f"      • R² (CV)     = {best_model['R2_CV']:.4f} ({best_model['R2_CV']*100:.2f}% variance expliquée)")
print(f"      • RMSE (CV)   = {best_model['RMSE_CV']:.4f}°C")
print(f"      • MAE (CV)    = {best_model['MAE_CV']:.4f}°C")
if not np.isnan(best_model['Overfit']):
    print(f"      • Surapprentissage = {best_model['Overfit']:.4f}")
    if best_model['Overfit'] > 0.1:
        print(f"        ⚠️ Surapprentissage détecté (> 0.10)")

# =========================================================
# ÉTAPE 6 : ANALYSE DÉTAILLÉE DU MEILLEUR MODÈLE
# =========================================================
print("\n" + "="*70)
print("6️⃣ ANALYSE DÉTAILLÉE DU MEILLEUR MODÈLE")
print("="*70)

best_name = best_model['Modèle']
predicteurs_best = modeles[best_name]

print(f"\n📋 Modèle : {best_name}")
print(f"   Variables : {', '.join(predicteurs_best)}")

X_best = df_clean[predicteurs_best].values
y_best = df_clean[target].values

mask = ~(np.isnan(X_best).any(axis=1) | np.isnan(y_best))
X_best = X_best[mask]
y_best = y_best[mask]

X_best_scaled, mean_best, std_best = standardize(X_best)

if has_sklearn:
    model_final = LinearRegression()
else:
    model_final = SimpleLinearRegression()

model_final.fit(X_best_scaled, y_best)
y_pred_final = model_final.predict(X_best_scaled)

# Équation du modèle
print(f"\n📐 ÉQUATION DU MODÈLE (variables standardisées) :")
print(f"\nICU_i = {model_final.intercept_:.4f}", end="")
for i, var in enumerate(predicteurs_best):
    coef = model_final.coef_[i]
    signe = " +" if coef >= 0 else " "
    print(f"{signe}{coef:.4f}·{var}", end="")
print(" + ε\n")

# Coefficients par ordre d'importance
print("📊 COEFFICIENTS (par ordre d'importance absolue) :\n")
coeffs_final = list(zip(predicteurs_best, model_final.coef_))
coeffs_final.sort(key=lambda x: abs(x[1]), reverse=True)

print("-"*70)
print(f"{'Variable':<20} {'Coefficient':<15} {'Impact'}")
print("-"*70)

for var, coef in coeffs_final:
    if coef > 0:
        impact = "↑ Augmente ICU (effet réchauffant)"
    else:
        impact = "↓ Diminue ICU (effet refroidissant)"
    print(f"{var:<20} {coef:>8.4f}       {impact}")

print("-"*70)

# Interprétation physique
print("\n💡 INTERPRÉTATION PHYSIQUE :")

print("\n🔥 Facteurs augmentant l'ICU (coefficients positifs) :")
has_positive = False
for var, coef in coeffs_final:
    if coef > 0:
        has_positive = True
        if 'Bi' in var or 'B_' in var:
            print(f"   • {var}: Surfaces imperméables → Absorption chaleur, réduction évapotranspiration")
        elif 'Hi' in var or 'H_' in var:
            print(f"   • {var}: Bâti en hauteur → Piégeage radiatif, réduction ventilation")
        elif 'Di' in var:
            print(f"   • {var}: Densité population → Chaleur anthropique (trafic, climatisation)")
        elif 'T65i' in var or 'T14i' in var:
            print(f"   • {var}: Structure démographique → Effet indirect sur usage énergétique")
        else:
            print(f"   • {var}: Effet réchauffant sur l'ICU")

if not has_positive:
    print("   (Aucun facteur réchauffant significatif)")

print("\n❄️ Facteurs diminuant l'ICU (coefficients négatifs) :")
has_negative = False
for var, coef in coeffs_final:
    if coef < 0:
        has_negative = True
        if 'Vi' in var or 'V_' in var:
            print(f"   • {var}: Végétation → Évapotranspiration, ombrage, albédo")
        elif 'Wi' in var or 'W_' in var:
            print(f"   • {var}: Surfaces en eau → Évaporation, inertie thermique")
        elif 'Ai' in var or 'A_' in var:
            print(f"   • {var}: Altitude → Gradient adiabatique, exposition vents")
        else:
            print(f"   • {var}: Effet refroidissant sur l'ICU")

if not has_negative:
    print("   (Aucun facteur refroidissant significatif)")

# Diagnostics du modèle
if has_sklearn:
    r2_final = r2_score(y_best, y_pred_final)
    rmse_final = np.sqrt(mean_squared_error(y_best, y_pred_final))
    mae_final = mean_absolute_error(y_best, y_pred_final)
else:
    r2_final = r2_score_custom(y_best, y_pred_final)
    rmse_final = rmse_custom(y_best, y_pred_final)
    mae_final = mae_custom(y_best, y_pred_final)

print("\n\n📈 DIAGNOSTICS DU MODÈLE :")
print(f"\n1. Coefficient de détermination (R²) : {r2_final:.4f}")
print(f"   → Le modèle explique {r2_final*100:.2f}% de la variance de l'ICU")
print(f"   → {(1-r2_final)*100:.2f}% de variance non expliquée (facteurs omis, bruit)")

print(f"\n2. Erreur quadratique moyenne (RMSE) : {rmse_final:.4f}°C")
print(f"   → Erreur moyenne de prédiction (sensible aux valeurs extrêmes)")

print(f"\n3. Erreur absolue moyenne (MAE) : {mae_final:.4f}°C")
print(f"   → Erreur typique en valeur absolue")

# Analyse des résidus
residus = y_best - y_pred_final

print(f"\n4. Analyse des résidus :")
print(f"   Moyenne    : {residus.mean():.6f}°C (devrait être ≈ 0)")
print(f"   Écart-type : {residus.std():.4f}°C")
print(f"   Min        : {residus.min():.4f}°C")
print(f"   Max        : {residus.max():.4f}°C")

# Test de normalité des résidus
skew_residus = stats.skew(residus)
print(f"   Skewness   : {skew_residus:.3f}", end="")
if abs(skew_residus) < 0.5:
    print(" ✓ (résidus symétriques)")
else:
    print(" ⚠️ (résidus asymétriques)")

# =========================================================
# ÉTAPE 7 : SAUVEGARDE DES RÉSULTATS
# =========================================================
print("\n" + "="*70)
print("7️⃣ SAUVEGARDE DES RÉSULTATS")
print("="*70)

output_folder = 'C:/master SIGMA/projet_atelier/albi/'

# 1. Comparaison des modèles
fichier_comparaison = output_folder + 'regression_icu_comparaison.csv'
df_resultats.to_csv(fichier_comparaison, index=False, encoding='utf-8-sig')
print(f"\n✅ {fichier_comparaison}")
print(f"   Contenu : Comparaison de {len(df_resultats)} configurations")

# 2. Coefficients du meilleur modèle
coeffs_df = pd.DataFrame(coeffs_final, columns=['Variable', 'Coefficient'])
fichier_coeffs = output_folder + 'regression_icu_coefficients.csv'
coeffs_df.to_csv(fichier_coeffs, index=False, encoding='utf-8-sig')
print(f"\n✅ {fichier_coeffs}")
print(f"   Contenu : Coefficients standardisés du modèle {best_name}")

# 3. Prédictions et résidus
df_predictions = pd.DataFrame({
    'ICU_i_observee': y_best,
    'ICU_i_predite': y_pred_final,
    'Residus': residus,
    'Residus_pct': (residus / y_best * 100)
})
fichier_pred = output_folder + 'regression_icu_predictions.csv'
df_predictions.to_csv(fichier_pred, index=False, encoding='utf-8-sig')
print(f"\n✅ {fichier_pred}")
print(f"   Contenu : {len(y_best)} observations avec prédictions et résidus")

# 4. Rapport détaillé
fichier_rapport = output_folder + 'regression_icu_rapport.txt'
with open(fichier_rapport, 'w', encoding='utf-8') as f:
    f.write("="*70 + "\n")
    f.write("RAPPORT D'ANALYSE - MODÉLISATION ICU ALBI\n")
    f.write("Méthodologie : Bottyán & Unger (2003)\n")
    f.write("="*70 + "\n\n")
    
    f.write("1. VARIABLE DÉPENDANTE\n")
    f.write("-"*70 + "\n")
    f.write(f"   Variable : {target}\n")
    f.write(f"   Définition : Intensité de l'Îlot de Chaleur Urbain\n")
    f.write(f"   Formule : ICU_i = LST_i - T_ref\n")
    f.write(f"   Unité : °C\n")
    f.write(f"   N observations : {len(df_clean)}\n")
    f.write(f"   Moyenne : {df_clean[target].mean():.2f}°C\n")
    f.write(f"   Écart-type : {df_clean[target].std():.2f}°C\n")
    f.write(f"   Min - Max : [{df_clean[target].min():.2f}, {df_clean[target].max():.2f}]°C\n\n")
    
    f.write("2. MEILLEUR MODÈLE\n")
    f.write("-"*70 + "\n")
    f.write(f"   Nom : {best_name}\n")
    f.write(f"   Méthode : {best_model['Méthode']}\n")
    f.write(f"   Nombre de variables : {best_model['N_vars']:.0f}\n")
    f.write(f"   R² (validation croisée) : {best_model['R2_CV']:.4f}\n")
    f.write(f"   RMSE (validation croisée) : {best_model['RMSE_CV']:.4f}°C\n")
    f.write(f"   MAE (validation croisée) : {best_model['MAE_CV']:.4f}°C\n\n")
    
    f.write("3. ÉQUATION DU MODÈLE\n")
    f.write("-"*70 + "\n")
    f.write(f"   ICU_i = {model_final.intercept_:.4f}")
    for i, var in enumerate(predicteurs_best):
        coef = model_final.coef_[i]
        signe = " +" if coef >= 0 else " "
        f.write(f"{signe}{coef:.4f}·{var}")
    f.write(" + ε\n\n")
    
    f.write("4. COEFFICIENTS (par ordre d'importance)\n")
    f.write("-"*70 + "\n")
    for var, coef in coeffs_final:
        impact = "↑ réchauffe" if coef > 0 else "↓ refroidit"
        f.write(f"   {var:<20} : {coef:>8.4f}  {impact}\n")
    f.write("\n")
    
    f.write("5. DIAGNOSTICS\n")
    f.write("-"*70 + "\n")
    f.write(f"   R² : {r2_final:.4f}\n")
    f.write(f"   RMSE : {rmse_final:.4f}°C\n")
    f.write(f"   MAE : {mae_final:.4f}°C\n")
    f.write(f"   Résidus - Moyenne : {residus.mean():.6f}°C\n")
    f.write(f"   Résidus - Écart-type : {residus.std():.4f}°C\n")
    f.write(f"   Résidus - Skewness : {skew_residus:.3f}\n")

print(f"\n✅ {fichier_rapport}")
print(f"   Contenu : Rapport complet de l'analyse")

# =========================================================
# CONCLUSION
# =========================================================
print("\n" + "="*70)
print("🎉 ANALYSE TERMINÉE AVEC SUCCÈS !")
print("="*70)

print("\n📌 RÉCAPITULATIF :")
print(f"   • Variable modélisée : ICU_i (Intensité ICU)")
print(f"   • Unité : °C (écart par rapport à T_ref)")
print(f"   • Résolution spatiale : Carreaux INSEE 1km × 1km")
print(f"   • Méthode : Régression linéaire multiple")
print(f"   • Validation : Validation croisée 5-fold")
print(f"   • Meilleur modèle : {best_name}")
print(f"   • Performance : R² = {best_model['R2_CV']:.4f}, RMSE = {best_model['RMSE_CV']:.4f}°C")

print("\n📁 FICHIERS GÉNÉRÉS :")
print(f"   1. regression_icu_comparaison.csv  - Comparaison modèles")
print(f"   2. regression_icu_coefficients.csv - Coefficients")
print(f"   3. regression_icu_predictions.csv  - Prédictions et résidus")
print(f"   4. regression_icu_rapport.txt      - Rapport complet")

print("\n📊 PROCHAINES ÉTAPES SUGGÉRÉES :")
print("   1. Visualiser les résidus (carte + histogramme)")
print("   2. Tester autocorrélation spatiale (Moran's I)")
print("   3. Cartographier ICU observé vs prédit")
print("   4. Analyser les zones à forte erreur de prédiction")

print("\n" + "="*70)