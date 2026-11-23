"""
Registro de experimentos.

Cada experimento define el modelo, las variables de metadatos y la receta de
entrenamiento. Todos comparten EXACTAMENTE la misma receta salvo el factor que
se ablaciona.

Campos:
  model        'image' | 'concat' | 'gated' | 'logreg' (tabular, 04_tabular_baselines.py)
  meta_vars    variables de metadatos (age, sex, site, diam, ratio, source)
  age_enc      'cont' | 'bin' | 'both'
  diam_enc     'cont' | 'bin' | 'both'
  missing      'indicator' | 'impute' | 'legacy'
  gate, xattn, aux   componentes de la arquitectura 'gated'
  mixup        Mixup (imagen + vector de metadatos + etiqueta con la misma lambda)
  loss         'focal' | 'ce'
  permute_train  metadatos permutados aleatoriamente en entrenamiento (control negativo)
  sources      None (ambas) o 'VRUH' (entrenar y evaluar solo en VRUH)
  split_kind   'cv' | 'loso'
  tier         1 = principal (3 repeticiones), 2 = ablación (repetición 0 por defecto)
  question     qué pregunta responde el experimento
"""

META4 = ["age", "sex", "site", "diam"]

BASE = dict(model="concat", meta_vars=META4, age_enc="cont", diam_enc="cont", missing="indicator",
            gate=True, xattn=True, aux=False, mixup=True, loss="focal", permute_train=False,
            sources=None, split_kind="cv", tier=1, meta_noise=0.1)


def E(**kw):
    d = dict(BASE)
    d.update(kw)
    return d


EXPERIMENTS = {
    # ---------------- Comparación principal ----------------
    "image": E(model="image", meta_vars=[], question="Baseline solo imagen (misma receta)"),
    "concat_meta": E(question="Imagen + metadatos, fusión simple por concatenación (comparación primaria)"),
    "gated_meta": E(model="gated", aux=True,
                    question="Arquitectura propuesta corregida: gated fusion + cross-attention espacial + cabezas auxiliares"),
    # ---------------- Controles de confusión y negativos ----------------
    "image_source": E(meta_vars=["source"], question="imagen + indicador de fuente"),
    "concat_meta_source": E(meta_vars=META4 + ["source"], question="¿aportan los metadatos más allá de la fuente?"),
    "concat_meta_permtrain": E(permute_train=True, question="control con metadatos permutados"),
    "concat_meta_legacy": E(missing="legacy", meta_vars=META4 + ["ratio"],
                            question="codificación original (ratio=0.5 imputado, ceros) para cuantificar el sesgo"),
    # ---------------- Fuente única (metadatos completos) ----------------
    "image_vruh": E(model="image", meta_vars=[], sources="VRUH", question="solo VRUH"),
    "concat_meta_vruh": E(sources="VRUH", question="solo VRUH, metadatos casi completos"),
    "gated_meta_vruh": E(model="gated", aux=True, sources="VRUH", question="solo VRUH"),
    # ---------------- Leave-one-source-out ----------------
    "image_loso": E(model="image", meta_vars=[], split_kind="loso", question="LOSO"),
    "concat_meta_loso": E(split_kind="loso", question="LOSO"),
    "gated_meta_loso": E(model="gated", aux=True, split_kind="loso", question="LOSO"),
    # ---------------- Ablación de variables ----------------
    **{f"concat_only_{v}": E(meta_vars=[v], tier=2, question=f"solo {v}") for v in META4},
    **{f"concat_drop_{v}": E(meta_vars=[m for m in META4 if m != v], tier=1 if v == "diam" else 2,
                             question=f"sin {v}" + (" (el diámetro faltante identifica ISIC; nivel 1)" if v == "diam" else ""))
       for v in META4},
    # ---------------- Codificación y datos faltantes ----------------
    "concat_meta_binned": E(age_enc="bin", diam_enc="bin", tier=2, question="edad y diámetro discretizados"),
    "concat_meta_impute": E(missing="impute", tier=2, question="imputación sin indicador de faltante"),
    "concat_meta_ratio": E(meta_vars=META4 + ["ratio"], tier=2, question="añadir ratio con indicador"),
    # ---------------- Ablación de arquitectura ----------------
    "gated_noaux": E(model="gated", aux=False, tier=2, question="sin cabezas auxiliares"),
    "gated_noxattn": E(model="gated", aux=True, xattn=False, tier=2, question="sin cross-attention"),
    "gated_nogate": E(model="gated", aux=True, gate=False, tier=2, question="sin gated fusion (suma)"),
    "gated_nomixup": E(model="gated", aux=True, mixup=False, tier=2, question="sin Mixup"),
    "gated_ce": E(model="gated", aux=True, loss="ce", tier=2, question="cross-entropy ponderada en vez de focal"),
    "image_nomixup": E(model="image", meta_vars=[], mixup=False, tier=2, question="receta sin Mixup"),
    "concat_nomixup": E(mixup=False, tier=2, question="receta sin Mixup"),
    "image_ce": E(model="image", meta_vars=[], loss="ce", tier=2, question="receta con CE"),
    "concat_ce": E(loss="ce", tier=2, question="receta con CE"),
}

# ---------------- Paso exploratorio 4: receta sobre concat_meta ----------------
# tier 3 = exploratorio; se entrenan con --out pipeline/runs_exploratory (06_analyze.py no los ve).
EXPLORATORY = {
    "x4_moddrop15": E(mod_drop_img=0.15, tier=3, question="Paso 4 R1: modality dropout de la imagen p=0.15"),
    "x4_moddrop30": E(mod_drop_img=0.30, tier=3, question="Paso 4 R1: modality dropout de la imagen p=0.30"),
    "x4_moddrop50": E(mod_drop_img=0.50, tier=3, question="Paso 4 R1: modality dropout de la imagen p=0.50"),
    "x4_moddrop30_sym": E(mod_drop_img=0.30, mod_drop_meta=0.30, tier=3,
                          question="Paso 4 R1: modality dropout simétrico 0.30/0.30"),
    "x4_aux10": E(aux=True, aux_weight=0.1, tier=3, question="Paso 4 R2: concat + cabezas auxiliares (peso 0.1)"),
    "x4_aux30": E(aux=True, aux_weight=0.3, tier=3, question="Paso 4 R2: concat + cabezas auxiliares (peso 0.3)"),
    "x4_ce_nomix": E(loss="ce", mixup=False, tier=3, question="Paso 4 R3: concat con CE y sin Mixup"),
}
EXPERIMENTS.update(EXPLORATORY)

# ---------------- Batería x6 (tras la corrección de metadatos) ----------------
# Arquitecturas que empiezan exactamente como el modelo de imagen y receta de la rama de metadatos.
# x6_image_s* / x6_concat_s*: semillas adicionales para comparar ensembles (menos varianza entre semillas).
X6 = {
    "x6_film": E(model="film", tier=3, question="M2: FiLM (metadatos modulan la proyección de imagen)"),
    "x6_film_mlr10": E(model="film", meta_lr_mult=10, tier=3, question="M2: FiLM con lr x10 en la rama de metadatos"),
    "x6_add_lin": E(model="add_lin", meta_lr_mult=30, tier=3,
                    question="logit = imagen + lineal(metadatos), end-to-end (lr x30 en el término lineal)"),
    "x6_add_mlp": E(model="add_mlp", meta_lr_mult=10, tier=3, question="logit = imagen + MLP(metadatos), end-to-end"),
    "x6_concat_mlr10": E(meta_lr_mult=10, tier=3, question="concat con lr x10 en la rama de metadatos"),
    **{f"x6_add_mlp_s{k}": E(model="add_mlp", meta_lr_mult=10, tier=3, question="semilla adicional de x6_add_mlp")
       for k in range(1, 5)},
    "x6_concat_light": E(meta_dropout=0.1, meta_noise=0.0, tier=3,
                         question="concat con dropout 0,1 y sin ruido en la rama de metadatos"),
    **{f"x6_image_s{k}": E(model="image", meta_vars=[], tier=3, question="semilla adicional de image")
       for k in (1, 2)},
    **{f"x6_concat_s{k}": E(tier=3, question="semilla adicional de concat_meta") for k in (1, 2)},
    **{f"x6_image_s{k}": E(model="image", meta_vars=[], tier=3, question="semilla adicional de image") for k in (3, 4)},
    **{f"x6_concat_s{k}": E(tier=3, question="semilla adicional de concat_meta") for k in (3, 4)},
    **{f"x6_image_vruh_s{k}": E(model="image", meta_vars=[], sources="VRUH", tier=3,
                                question="semilla adicional de image_vruh") for k in (1, 2)},
    **{f"x6_concat_meta_vruh_s{k}": E(sources="VRUH", tier=3, question="semilla adicional de concat_meta_vruh")
       for k in (1, 2, 3, 4)},
    **{f"x6_image_vruh_s{k}": E(model="image", meta_vars=[], sources="VRUH", tier=3,
                                question="semilla adicional de image_vruh") for k in (3, 4)},
}
EXPERIMENTS.update(X6)
EXPLORATORY.update(X6)

# ---------------- x7: metadatos como información privilegiada (LUPI) ----------------
# Modelo de IMAGEN con cabezas auxiliares que predicen diámetro, edad, sexo y localización desde la imagen
# (solo en entrenamiento). La inferencia usa solo la imagen -> se compara con las semillas de `image`.
X7 = {
    "x7_lupi": E(model="image", meta_vars=[], aux=True, aux_diam=True, aux_weight=0.3, tier=3,
                 question="LUPI: imagen + tareas auxiliares (diámetro, edad, sexo, localización), peso 0,3"),
    "x7_lupi10": E(model="image", meta_vars=[], aux=True, aux_diam=True, aux_weight=0.1, tier=3,
                   question="LUPI con peso 0,1"),
    "x7_lupi_vruh": E(model="image", meta_vars=[], aux=True, aux_diam=True, aux_weight=0.3, sources="VRUH", tier=3,
                      question="LUPI solo VRUH (diámetro casi completo)"),
    # comprobación de robustez a la semilla de x7_lupi10 (pasó la puerta con una sola semilla, 2026-10-10)
    **{f"x7_lupi10_s{k}": E(model="image", meta_vars=[], aux=True, aux_diam=True, aux_weight=0.1, tier=3,
                            question="semilla adicional de x7_lupi10") for k in range(1, 5)},
    **{f"x7_lupi10_vruh{'' if k == 0 else f'_s{k}'}": E(model="image", meta_vars=[], aux=True, aux_diam=True,
                                                        aux_weight=0.1, sources="VRUH", tier=3,
                                                        question="LUPI peso 0,1 solo VRUH")
       for k in range(3)},
}
EXPERIMENTS.update(X7)
EXPLORATORY.update(X7)

# ---------------- x8: ajuste de lr × wd en los DOS brazos (protocolo fijado el 2026-10-20) ----------------
# Misma rejilla en image y concat_meta; lr 1e-4 / wd 0.01 es la receta actual (image y concat_meta ya entrenados).
X8_GRID = [(lr, wd) for lr in ("3e-5", "1e-4", "3e-4") for wd in ("0.01", "0.1") if (lr, wd) != ("1e-4", "0.01")]
X8 = {
    f"x8_{arm}_lr{lr}_wd{wd}": E(**(dict(model="image", meta_vars=[]) if arm == "image" else {}),
                                 lr=float(lr), wd=float(wd), tier=3,
                                 question=f"Ajuste de receta en los dos brazos: {arm}, lr {lr}, wd {wd}")
    for arm in ("image", "concat") for lr, wd in X8_GRID
}
EXPERIMENTS.update(X8)
EXPLORATORY.update(X8)

# ---------------- x9: red base × resolución en los DOS brazos (protocolo fijado el 2026-10-20) ----------------
# Receta actual (lr 1e-4, wd 0.01); DenseNet121 a 224 px es la referencia (image y concat_meta, ya entrenados).
X9_GRID = [("densenet121", 320), ("resnet50", 224), ("resnet50", 320)]
X9 = {
    f"x9_{arm}_{bb}_{size}": E(**(dict(model="image", meta_vars=[]) if arm == "image" else {}),
                               backbone=bb, img_size=size, tier=3,
                               question=f"Red base y resolución en los dos brazos: {arm}, {bb}, {size} px")
    for arm in ("image", "concat") for bb, size in X9_GRID
}
EXPERIMENTS.update(X9)
EXPLORATORY.update(X9)

# ---------------- x10: antecedentes personales y familiares en la red, solo VRUH (2026-10-21) ----------------
X10 = {f"x10_hx_vruh{'' if k == 0 else f'_s{k}'}": E(meta_vars=META4 + ["personal_hx", "family_hx"], sources="VRUH", tier=3,
                                                    question="Antecedentes personales y familiares en concat, solo VRUH")
       for k in range(5)}
EXPERIMENTS.update(X10)
EXPLORATORY.update(X10)

# ---------------- x11: métodos de entrenamiento en los DOS brazos, 3 semillas (2026-10-21) ----------------
# base | thick (cabeza de clase de grosor, solo train) | adv (fuente con inversión del gradiente); todos guardan prob_tta
X11_METHODS = {"base": {}, "thick": dict(thick_weight=0.3), "adv": dict(adv_weight=0.3)}
X11 = {f"x11_{mt}_{arm}{'' if k == 0 else f'_s{k}'}": E(**(dict(model="image", meta_vars=[]) if arm == "image" else {}),
                                                       **kw, save_tta=True, tier=3,
                                                       question=f"Método {mt} en los dos brazos: {arm}, semilla {k}")
       for mt, kw in X11_METHODS.items() for arm in ("image", "concat") for k in range(3)}
EXPERIMENTS.update(X11)
EXPLORATORY.update(X11)

# ---------------- x13: métodos que dan más peso a los metadatos, 3 semillas (2026-10-21) ----------------
# Brazo de imagen de ogm/gblend/staged/contrast: x11_base_image (misma receta). breslow: en los dos brazos.
X13_METHODS = {"ogm": dict(ogm_alpha=0.5), "gblend": dict(gblend_weight=0.5), "staged": dict(staged=True),
               "contrast": dict(contrast_weight=0.1), "breslow": dict(breslow_weight=0.3)}
X13 = {f"x13_{mt}_{arm}{'' if k == 0 else f'_s{k}'}": E(**(dict(model="image", meta_vars=[]) if arm == "image" else {}),
                                                       **kw, save_tta=True, tier=3,
                                                       question=f"x13 {mt}: {arm}, semilla {k}")
       for mt, kw in X13_METHODS.items() for arm in (("image", "concat") if mt == "breslow" else ("concat",))
       for k in range(3)}
EXPERIMENTS.update(X13)
EXPLORATORY.update(X13)

# ---------------- x14: replicación de gblend y breslow en las repeticiones 1 y 2 (2026-10-21) ----------------
X14 = {f"x14_gbbres_concat{'' if k == 0 else f'_s{k}'}": E(gblend_weight=0.5, breslow_weight=0.3, save_tta=True, tier=3,
                                                          question=f"x14 gblend + breslow: concat, semilla {k}")
       for k in range(3)}
EXPERIMENTS.update(X14)
EXPLORATORY.update(X14)

# ---------------- x15: SWA + label smoothing y destilación desde la fusión tardía (2026-10-21) ----------------
X15_METHODS = {"swa": dict(swa_start=20, epochs=40, label_smoothing=0.1), "kd": dict(kd_beta=0.5)}
X15 = {f"x15_{mt}_{arm}{'' if k == 0 else f'_s{k}'}": E(**(dict(model="image", meta_vars=[]) if arm == "image" else {}),
                                                       **kw, save_tta=True, tier=3,
                                                       question=f"x15 {mt}: {arm}, semilla {k}")
       for mt, kw in X15_METHODS.items() for arm in ("image", "concat") for k in range(3)}
EXPERIMENTS.update(X15)
EXPLORATORY.update(X15)

# ---------------- x16: LP-FT, muestreo por lesión y FiLM sobre el mapa de características (2026-10-21) ----------------
X16_METHODS = {"lpft": dict(lpft_epochs=5), "lesw": dict(lesion_weighted=True), "fmfilm": dict(fmap_film=True)}
X16 = {f"x16_{mt}_{arm}{'' if k == 0 else f'_s{k}'}": E(**(dict(model="image", meta_vars=[]) if arm == "image" else {}),
                                                       **kw, save_tta=True, tier=3,
                                                       question=f"x16 {mt}: {arm}, semilla {k}")
       for mt, kw in X16_METHODS.items() for arm in (("concat",) if mt == "fmfilm" else ("image", "concat"))
       for k in range(3)}
EXPERIMENTS.update(X16)
EXPLORATORY.update(X16)

# Modelos tabulares (sin red; 04_tabular_baselines.py), mismas particiones
TABULAR = {
    "logreg_meta": dict(meta_vars=META4, missing="indicator", question="solo metadatos (regresión logística)"),
    "logreg_meta_source": dict(meta_vars=META4 + ["source"], missing="indicator", question="Confusión por fuente"),
    "logreg_source": dict(meta_vars=["source"], missing="indicator", question="solo fuente"),
    "logreg_meta_legacy": dict(meta_vars=META4 + ["ratio"], missing="legacy", question="Confusión por fuente"),
    "logreg_meta_nodiam": dict(meta_vars=["age", "sex", "site"], missing="indicator", question="sin diámetro"),
    "logreg_meta_vruh": dict(meta_vars=META4, missing="indicator", sources="VRUH", question="Confusión por fuente"),
}


def splits_for(exp: str, all_split_ids, repeats=None):
    cfg = EXPERIMENTS[exp]
    kind = cfg["split_kind"]
    if repeats is None:
        repeats = [0, 1, 2] if cfg["tier"] == 1 else [0]
    out = []
    for s in all_split_ids:
        if not s.startswith(kind + "_"):
            continue
        rep = int(s.split("_r")[1].split("_")[0]) if kind == "cv" else int(s.split("_s")[-1])
        if rep in repeats:
            out.append(s)
    return sorted(out)
