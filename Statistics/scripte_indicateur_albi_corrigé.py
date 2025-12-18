# =========================================================
# 🎯 SCRIPT COMPLET - INDICATEURS ALBI CORRIGÉ
# VERSION ICU : Calcul de l'Intensité de l'Îlot de Chaleur Urbain
# Méthodologie : ICU 30m → Agrégation vers carreaux INSEE 1km
# =========================================================

import processing
from qgis.core import (QgsProject, QgsVectorLayer, QgsRasterLayer, QgsFeature, 
                       QgsField, QgsGeometry, QgsDistanceArea, QgsPointXY,
                       QgsVectorFileWriter, QgsCoordinateTransformContext,
                       QgsRasterBandStats)
from qgis.PyQt.QtCore import QVariant
import math
import os
import time
import datetime
import numpy as np

print("="*70)
print("🚀 DÉMARRAGE DU CALCUL DES INDICATEURS ALBI - VERSION ICU")
print("="*70)

# --- Fonction pour récupérer une couche ---
def get_layer(name):
    layers = QgsProject.instance().mapLayersByName(name)
    if not layers:
        raise ValueError(f"❌ Couche '{name}' non trouvée !")
    print(f"✅ {name}")
    return layers[0]

# --- Configuration ---
output_folder = 'C:/master SIGMA/projet_atelier/albi/'
output_gpkg = output_folder + 'indicateurs_albi_icu.gpkg'

# =========================================================
# ÉTAPE 1 : CHARGER LA GRILLE INSEE ET LES COUCHES
# =========================================================
print("\n" + "="*70)
print("1️⃣ CHARGEMENT DES COUCHES")
print("="*70)

# Grille INSEE
grille = get_layer('pop_insee_stat')
print(f"   📊 Nombre de mailles : {grille.featureCount()}")
print(f"   🗺 Système de coordonnées : {grille.crs().authid()}")

# Couche bâti
bati = get_layer('bati')
print(f"   🏠 Nombre de bâtiments : {bati.featureCount()}")

# Surface en eau
eau = get_layer('surface_en_eau')
print(f"   💧 Nombre d'éléments eau : {eau.featureCount()}")

# Surface agricole/végétalisée
veg = get_layer('surface_agricole')
print(f"   🌿 Nombre d'éléments végétaux : {veg.featureCount()}")

# MNT (optionnel)
try:
    mnt = get_layer('MNT')
    has_mnt = True
except:
    print("⚠️ MNT non trouvé (optionnel)")
    has_mnt = False

# Température du sol (LST Landsat 30m)
lst = get_layer('temperature_sol_albi')
print(f"   🌡️ Résolution LST : {lst.rasterUnitsPerPixelX()}m × {lst.rasterUnitsPerPixelY()}m")

# =========================================================
# ÉTAPE 1.5 : CALCUL DE L'ICU À 30M (NOUVELLE ÉTAPE)
# =========================================================
print("\n" + "="*70)
print("1.5️⃣ CALCUL DE L'ICU À RÉSOLUTION NATIVE (30M)")
print("="*70)

# --- Définir la température de référence (T_ref) ---
print("\n📊 Calcul de la température de référence (T_ref)...")

# OPTION 1 : Utiliser le 10e percentile des températures (zones les plus froides)
# OPTION 2 : Utiliser la moyenne des zones rurales/végétalisées
# OPTION 3 : Définir manuellement une zone de référence

# On va utiliser l'OPTION 1 : 10e percentile
stats = lst.dataProvider().bandStatistics(1, QgsRasterBandStats.All)
print(f"   📈 Statistiques LST :")
print(f"      Min : {stats.minimumValue:.2f}°C")
print(f"      Max : {stats.maximumValue:.2f}°C")
print(f"      Moyenne : {stats.mean:.2f}°C")

# Calculer le percentile 10 avec l'outil QGIS
# Alternative : définir T_ref manuellement
# Pour l'instant, on va utiliser le minimum + 2°C comme approximation du 10e percentile
T_ref = stats.minimumValue + 2.0

print(f"\n   🎯 Température de référence retenue : T_ref = {T_ref:.2f}°C")
print(f"      (Approximation : min + 2°C)")

# --- Créer le raster ICU_30m = LST - T_ref ---
print("\n🔥 Création du raster ICU à 30m...")

icu_30m_path = output_folder + 'ICU_30m.tif'

# Utiliser la calculatrice raster
processing.run("gdal:rastercalculator", {
    'INPUT_A': lst,
    'BAND_A': 1,
    'FORMULA': f'A - {T_ref}',
    'OUTPUT': icu_30m_path,
    'NO_DATA': None,
    'RTYPE': 5  # Float32
})

print(f"✅ Raster ICU 30m créé : {icu_30m_path}")

# Charger le raster ICU
icu_30m = QgsRasterLayer(icu_30m_path, "ICU_30m")
if not icu_30m.isValid():
    raise ValueError("❌ Impossible de charger le raster ICU_30m")

QgsProject.instance().addMapLayer(icu_30m)
print("✅ Raster ICU_30m ajouté au projet")

# Statistiques du raster ICU
stats_icu = icu_30m.dataProvider().bandStatistics(1, QgsRasterBandStats.All)
print(f"\n   📊 Statistiques ICU_30m :")
print(f"      Min : {stats_icu.minimumValue:.2f}°C")
print(f"      Max : {stats_icu.maximumValue:.2f}°C")
print(f"      Moyenne : {stats_icu.mean:.2f}°C")

# =========================================================
# ÉTAPE 2 : PRÉPARER LA GRILLE
# =========================================================
print("\n" + "="*70)
print("2️⃣ PRÉPARATION DE LA GRILLE")
print("="*70)

# Vérifier l'existence de id_carreau_1km
if grille.fields().indexOf('id_carreau_1km') == -1:
    raise ValueError("❌ Le champ 'id_carreau_1km' n'existe pas dans pop_insee_stat")
print("✅ Utilisation de 'id_carreau_1km' comme identifiant")

# Calculer la surface de chaque maille
grille = processing.run("qgis:fieldcalculator", {
    'INPUT': grille,
    'FIELD_NAME': 'superficie',
    'FIELD_TYPE': 0,
    'FIELD_LENGTH': 10,
    'FIELD_PRECISION': 2,
    'NEW_FIELD': True,
    'FORMULA': '$area',
    'OUTPUT': 'TEMPORARY_OUTPUT'
})['OUTPUT']
print("✅ Surface calculée pour chaque maille")

result = grille

# =========================================================
# ÉTAPE 3 : CALCULER LES INDICATEURS LOCAUX
# =========================================================
print("\n" + "="*70)
print("3️⃣ CALCUL DES INDICATEURS LOCAUX")
print("="*70)

# --- 3.1 Surface bâtie (Bi) et hauteur moyenne (Hi) ---
print("\n🏗 Calcul surface bâtie et hauteur moyenne...")

bati_decoupe = processing.run("native:intersection", {
    'INPUT': bati,
    'OVERLAY': result,
    'INPUT_FIELDS': ['hauteur'],
    'OVERLAY_FIELDS': ['id_carreau_1km', 'superficie'],
    'OUTPUT': 'TEMPORARY_OUTPUT'
})['OUTPUT']

print("✅ Bâti découpé par la grille")

bati_decoupe = processing.run("qgis:fieldcalculator", {
    'INPUT': bati_decoupe,
    'FIELD_NAME': 'surface_bati',
    'FIELD_TYPE': 0,
    'FIELD_LENGTH': 10,
    'FIELD_PRECISION': 2,
    'NEW_FIELD': True,
    'FORMULA': '$area',
    'OUTPUT': 'TEMPORARY_OUTPUT'
})['OUTPUT']

result = processing.run("qgis:joinbylocationsummary", {
    'INPUT': result,
    'JOIN': bati_decoupe,
    'PREDICATE': [0],
    'JOIN_FIELDS': ['surface_bati', 'hauteur'],
    'SUMMARIES': [5, 2, 0],
    'DISCARD_NONMATCHING': False,
    'OUTPUT': 'TEMPORARY_OUTPUT'
})['OUTPUT']

result = processing.run("qgis:fieldcalculator", {
    'INPUT': result,
    'FIELD_NAME': 'Bi',
    'FIELD_TYPE': 0,
    'FIELD_LENGTH': 10,
    'FIELD_PRECISION': 2,
    'NEW_FIELD': True,
    'FORMULA': 'CASE WHEN "superficie" > 0 THEN (coalesce("surface_bati_sum", 0) / "superficie") * 100 ELSE 0 END',
    'OUTPUT': 'TEMPORARY_OUTPUT'
})['OUTPUT']

result = processing.run("qgis:fieldcalculator", {
    'INPUT': result,
    'FIELD_NAME': 'Hi',
    'FIELD_TYPE': 0,
    'FIELD_LENGTH': 10,
    'FIELD_PRECISION': 2,
    'NEW_FIELD': True,
    'FORMULA': 'CASE WHEN "hauteur_count" > 0 THEN "hauteur_sum" / "hauteur_count" ELSE 0 END',
    'OUTPUT': 'TEMPORARY_OUTPUT'
})['OUTPUT']

print("✅ Bi et Hi calculés")

# --- 3.2 Surface en eau (Wi) ---
print("\n💧 Calcul surface en eau...")

eau_decoupe = processing.run("native:intersection", {
    'INPUT': eau,
    'OVERLAY': result,
    'INPUT_FIELDS': [],
    'OVERLAY_FIELDS': ['id_carreau_1km', 'superficie'],
    'OUTPUT': 'TEMPORARY_OUTPUT'
})['OUTPUT']

eau_decoupe = processing.run("qgis:fieldcalculator", {
    'INPUT': eau_decoupe,
    'FIELD_NAME': 'surf_eau_reel',
    'FIELD_TYPE': 0,
    'FIELD_LENGTH': 10,
    'FIELD_PRECISION': 2,
    'NEW_FIELD': True,
    'FORMULA': '$area',
    'OUTPUT': 'TEMPORARY_OUTPUT'
})['OUTPUT']

result = processing.run("qgis:joinbylocationsummary", {
    'INPUT': result,
    'JOIN': eau_decoupe,
    'PREDICATE': [0],
    'JOIN_FIELDS': ['surf_eau_reel'],
    'SUMMARIES': [5],
    'DISCARD_NONMATCHING': False,
    'OUTPUT': 'TEMPORARY_OUTPUT'
})['OUTPUT']

result = processing.run("qgis:fieldcalculator", {
    'INPUT': result,
    'FIELD_NAME': 'Wi',
    'FIELD_TYPE': 0,
    'FIELD_LENGTH': 10,
    'FIELD_PRECISION': 2,
    'NEW_FIELD': True,
    'FORMULA': 'CASE WHEN "superficie" > 0 THEN (coalesce("surf_eau_reel_sum", 0) / "superficie") * 100 ELSE 0 END',
    'OUTPUT': 'TEMPORARY_OUTPUT'
})['OUTPUT']

print("✅ Wi calculé")

# --- 3.3 Surface végétalisée (Vi) ---
print("\n🌿 Calcul surface végétalisée...")

veg_decoupe = processing.run("native:intersection", {
    'INPUT': veg,
    'OVERLAY': result,
    'INPUT_FIELDS': [],
    'OVERLAY_FIELDS': ['id_carreau_1km', 'superficie'],
    'OUTPUT': 'TEMPORARY_OUTPUT'
})['OUTPUT']

veg_decoupe = processing.run("qgis:fieldcalculator", {
    'INPUT': veg_decoupe,
    'FIELD_NAME': 'surf_veg_reel',
    'FIELD_TYPE': 0,
    'FIELD_LENGTH': 10,
    'FIELD_PRECISION': 2,
    'NEW_FIELD': True,
    'FORMULA': '$area',
    'OUTPUT': 'TEMPORARY_OUTPUT'
})['OUTPUT']

result = processing.run("qgis:joinbylocationsummary", {
    'INPUT': result,
    'JOIN': veg_decoupe,
    'PREDICATE': [0],
    'JOIN_FIELDS': ['surf_veg_reel'],
    'SUMMARIES': [5],
    'DISCARD_NONMATCHING': False,
    'OUTPUT': 'TEMPORARY_OUTPUT'
})['OUTPUT']

result = processing.run("qgis:fieldcalculator", {
    'INPUT': result,
    'FIELD_NAME': 'Vi',
    'FIELD_TYPE': 0,
    'FIELD_LENGTH': 10,
    'FIELD_PRECISION': 2,
    'NEW_FIELD': True,
    'FORMULA': 'CASE WHEN "superficie" > 0 THEN (coalesce("surf_veg_reel_sum", 0) / "superficie") * 100 ELSE 0 END',
    'OUTPUT': 'TEMPORARY_OUTPUT'
})['OUTPUT']

print("✅ Vi calculé")

# --- 3.4 Altitude moyenne (Ai) ---
if has_mnt:
    print("\n⛰️ Calcul altitude moyenne...")
    result = processing.run("native:zonalstatisticsfb", {
        'INPUT': result,
        'INPUT_RASTER': mnt,
        'RASTER_BAND': 1,
        'COLUMN_PREFIX': 'Alt_',
        'STATISTICS': [2],
        'OUTPUT': 'TEMPORARY_OUTPUT'
    })['OUTPUT']
    
    result = processing.run("qgis:fieldcalculator", {
        'INPUT': result,
        'FIELD_NAME': 'Ai',
        'FIELD_TYPE': 0,
        'FIELD_LENGTH': 10,
        'FIELD_PRECISION': 2,
        'NEW_FIELD': True,
        'FORMULA': 'coalesce("Alt_mean", 0)',
        'OUTPUT': 'TEMPORARY_OUTPUT'
    })['OUTPUT']
    print("✅ Ai calculé")
else:
    result = processing.run("qgis:fieldcalculator", {
        'INPUT': result,
        'FIELD_NAME': 'Ai',
        'FIELD_TYPE': 0,
        'NEW_FIELD': True,
        'FORMULA': '0',
        'OUTPUT': 'TEMPORARY_OUTPUT'
    })['OUTPUT']
    print("⚠️ Ai = 0 (pas de MNT)")

# --- 3.5 ICU MOYEN PAR CARREAU (ICU_i) - MODIFIÉ ---
print("\n🔥 Calcul de l'ICU moyen par carreau INSEE (agrégation 30m → 1km)...")

# Utiliser le raster ICU_30m créé précédemment
result = processing.run("native:zonalstatisticsfb", {
    'INPUT': result,
    'INPUT_RASTER': icu_30m,
    'RASTER_BAND': 1,
    'COLUMN_PREFIX': 'ICU_',
    'STATISTICS': [2],  # Mean
    'OUTPUT': 'TEMPORARY_OUTPUT'
})['OUTPUT']

# Créer le champ ICU_i (variable dépendante du modèle)
result = processing.run("qgis:fieldcalculator", {
    'INPUT': result,
    'FIELD_NAME': 'ICU_i',
    'FIELD_TYPE': 0,
    'FIELD_LENGTH': 10,
    'FIELD_PRECISION': 2,
    'NEW_FIELD': True,
    'FORMULA': 'coalesce("ICU_mean", 0)',
    'OUTPUT': 'TEMPORARY_OUTPUT'
})['OUTPUT']

print("✅ ICU_i calculé (intensité moyenne de l'îlot de chaleur par carreau)")
print("   ℹ️ ICU_i = moyenne des pixels ICU_30m contenus dans chaque carreau")

# --- 3.6 Indicateurs démographiques ---
print("\n👥 Calcul indicateurs démographiques...")

has_pop = result.fields().indexOf('pop') != -1
has_pop0014 = result.fields().indexOf('pop0014') != -1
has_pop65p = result.fields().indexOf('pop65p') != -1

if has_pop:
    print(f"   📊 Champ population : 'pop'")
    
    result = processing.run("qgis:fieldcalculator", {
        'INPUT': result,
        'FIELD_NAME': 'Di',
        'FIELD_TYPE': 0,
        'FIELD_LENGTH': 10,
        'FIELD_PRECISION': 2,
        'NEW_FIELD': True,
        'FORMULA': 'CASE WHEN "superficie" > 0 THEN ("pop" / ("superficie" / 1000000)) ELSE 0 END',
        'OUTPUT': 'TEMPORARY_OUTPUT'
    })['OUTPUT']
    
    if has_pop65p:
        result = processing.run("qgis:fieldcalculator", {
            'INPUT': result,
            'FIELD_NAME': 'T65_plus',
            'FIELD_TYPE': 0,
            'FIELD_LENGTH': 10,
            'FIELD_PRECISION': 2,
            'NEW_FIELD': True,
            'FORMULA': 'CASE WHEN "pop" > 0 THEN ("pop65p" / "pop") * 100 ELSE 0 END',
            'OUTPUT': 'TEMPORARY_OUTPUT'
        })['OUTPUT']
    
    if has_pop0014:
        result = processing.run("qgis:fieldcalculator", {
            'INPUT': result,
            'FIELD_NAME': 'T0_14',
            'FIELD_TYPE': 0,
            'FIELD_LENGTH': 10,
            'FIELD_PRECISION': 2,
            'NEW_FIELD': True,
            'FORMULA': 'CASE WHEN "pop" > 0 THEN ("pop0014" / "pop") * 100 ELSE 0 END',
            'OUTPUT': 'TEMPORARY_OUTPUT'
        })['OUTPUT']
    
    print("   ✅ Indicateurs démographiques calculés")
else:
    print("   ⚠️ Données démographiques non disponibles")

print("\n✅ Tous les indicateurs locaux calculés")

# =========================================================
# ÉTAPE 4 : GÉOMÉTRIES ET CENTROÏDES
# =========================================================
print("\n" + "="*70)
print("4️⃣ CONVERSION EN GÉOMÉTRIES SIMPLES")
print("="*70)

result = processing.run("native:multiparttosingleparts", {
    'INPUT': result,
    'OUTPUT': 'TEMPORARY_OUTPUT'
})['OUTPUT']
print(f"✅ {result.featureCount()} entités")

# Réattribuer des ID uniques
result.startEditing()
fid_idx = result.fields().indexOf('fid')
if fid_idx != -1:
    result.dataProvider().deleteAttributes([fid_idx])
    result.updateFields()

result.dataProvider().addAttributes([QgsField('uid', QVariant.Int)])
result.updateFields()

uid_idx = result.fields().indexOf('uid')
for i, feat in enumerate(result.getFeatures(), 1):
    result.changeAttributeValue(feat.id(), uid_idx, i)
result.commitChanges()

# Centroïdes
result = processing.run("native:addxyfields", {
    'INPUT': result,
    'CRS_PROJECTION': result.crs(),
    'PREFIX': 'cent_',
    'OUTPUT': 'TEMPORARY_OUTPUT'
})['OUTPUT']
print("✅ Centroïdes calculés")

# =========================================================
# ÉTAPE 5 : MOYENNES PONDÉRÉES SPATIALES
# =========================================================
print("\n" + "="*70)
print("5️⃣ CALCUL DES MOYENNES PONDÉRÉES SPATIALES")
print("="*70)

L_values = [250, 500, 1000, 2000]
variables = ['Bi', 'Hi', 'Wi', 'Vi', 'Ai']

result.startEditing()
for L in L_values:
    for var in variables:
        field_name = f"{var}_L{L}"
        if result.fields().indexOf(field_name) == -1:
            result.dataProvider().addAttributes([QgsField(field_name, QVariant.Double)])
result.updateFields()
result.commitChanges()

distance_calc = QgsDistanceArea()
distance_calc.setSourceCrs(result.crs(), QgsProject.instance().transformContext())

# Collecter données
mailles_data = []
for feat in result.getFeatures():
    data = {
        'fid': feat.id(),
        'centroid': feat.geometry().centroid().asPoint()
    }
    for var in variables:
        val = feat[var]
        data[var] = val if val is not None else 0
    mailles_data.append(data)

print(f"✅ {len(mailles_data)} mailles collectées")

# Calcul des moyennes pondérées
for L in L_values:
    print(f"\n🔄 Calcul pour L = {L}m...")
    result.startEditing()
    
    for i, maille_i in enumerate(mailles_data):
        if (i + 1) % 10 == 0:
            print(f"   {i+1}/{len(mailles_data)}", end='\r')
        
        weights = []
        sum_w = 0
        for j, maille_j in enumerate(mailles_data):
            d = distance_calc.measureLine(maille_i['centroid'], maille_j['centroid'])
            w = math.exp(-d / L)
            sum_w += w
            weights.append((j, w))
        
        for var in variables:
            weighted_sum = sum(w/sum_w * mailles_data[j][var] for j, w in weights)
            field_idx = result.fields().indexOf(f"{var}_L{L}")
            result.changeAttributeValue(maille_i['fid'], field_idx, weighted_sum)
    
    result.commitChanges()
    print(f"\n   ✅ L={L}m terminé")

# =========================================================
# ÉTAPE 6 : SAUVEGARDE
# =========================================================
print("\n" + "="*70)
print("6️⃣ SAUVEGARDE")
print("="*70)

# Copie propre
uri = f"polygon?crs={result.crs().authid()}"
mem = QgsVectorLayer(uri, "temp", "memory")
mem.dataProvider().addAttributes(result.fields().toList())
mem.updateFields()

features = []
for f in result.getFeatures():
    nf = QgsFeature(mem.fields())
    nf.setGeometry(f.geometry())
    nf.setAttributes(f.attributes())
    features.append(nf)
mem.dataProvider().addFeatures(features)
print(f"✅ {len(features)} entités copiées")

# Supprimer anciennes couches
for layer in QgsProject.instance().mapLayers().values():
    if layer.source().startswith(output_gpkg):
        QgsProject.instance().removeMapLayer(layer.id())
time.sleep(1)

# Supprimer fichier
if os.path.exists(output_gpkg):
    try:
        os.remove(output_gpkg)
    except:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_gpkg = output_folder + f'indicateurs_albi_icu_{timestamp}.gpkg'

# Export GPKG
opts = QgsVectorFileWriter.SaveVectorOptions()
opts.driverName = "GPKG"
opts.layerName = "indicateurs_icu"

err = QgsVectorFileWriter.writeAsVectorFormatV3(
    mem, output_gpkg, QgsCoordinateTransformContext(), opts
)

if err[0] == QgsVectorFileWriter.NoError:
    print(f"✅ {output_gpkg}")
    layer = QgsVectorLayer(output_gpkg, "Indicateurs ICU Albi", "ogr")
    QgsProject.instance().addMapLayer(layer)
else:
    raise Exception(f"Erreur: {err}")

# Export CSV
csv_path = output_folder + 'indicateurs_albi_icu.csv'
QgsVectorFileWriter.writeAsVectorFormatV3(
    mem, csv_path, QgsCoordinateTransformContext(),
    QgsVectorFileWriter.SaveVectorOptions()
)
print(f"✅ {csv_path}")

# =========================================================
# STATISTIQUES
# =========================================================
print("\n" + "="*70)
print("📊 STATISTIQUES")
print("="*70)

stats = ['Bi', 'Hi', 'Wi', 'Vi', 'Ai', 'ICU_i']  # ICU_i au lieu de Ti
if has_pop:
    stats.append('Di')
if has_pop65p:
    stats.append('T65_plus')
if has_pop0014:
    stats.append('T0_14')

print(f"\n{'Variable':<12} {'Min':>8} {'Max':>8} {'Moy':>8}")
print("-"*40)
for s in stats:
    vals = [f[s] for f in layer.getFeatures() if f[s] is not None]
    if vals:
        import statistics
        print(f"{s:<12} {min(vals):>8.2f} {max(vals):>8.2f} {statistics.mean(vals):>8.2f}")

print("\n" + "="*70)
print("🎉 TERMINÉ !")
print("="*70)
print(f"\n📁 Fichiers créés :")
print(f"   • {icu_30m_path} (raster ICU 30m)")
print(f"   • {output_gpkg} (couche vectorielle)")
print(f"   • {csv_path} (export CSV)")
print(f"\n🔬 {layer.featureCount()} entités avec {len(layer.fields())} attributs")
print(f"\n📌 IMPORTANT :")
print(f"   • Variable dépendante : ICU_i (intensité de l'îlot de chaleur)")
print(f"   • T_ref utilisée : {T_ref:.2f}°C")
print(f"   • Résolution native : 30m × 30m")
print(f"   • Agrégation : carreaux INSEE 1km × 1km")