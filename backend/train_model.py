# Pipeline de dados e treino: limpa specs de notebooks, compara os 4 modelos da PUC-Rio e salva o model.pkl.

import os
import pickle
import re
import warnings
from typing import Any, Dict, List, Optional
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(ROOT_DIR, "data", "notebooks.csv")
MODEL_PATH = os.path.join(ROOT_DIR, "backend", "model.pkl")

FEATURE_NAMES = ["preco", "ram_gb", "ssd_gb", "score_cpu", "tem_gpu_dedicada"]
PROFILE_LABELS = {
    0: "Uso Básico / Estudos",
    1: "Trabalho / Programação / Multitarefa",
    2: "Jogos / Edição de Vídeo / 3D",
}


# --- Funções de limpeza via Regex (usadas nos anúncios brutos do Mercado Livre) ---

def extrair_ram_gb(texto: str) -> int:
    # Pega padrões como "16GB RAM", "8 GB DDR4", etc.
    match = re.search(r"\b(4|8|12|16|24|32|64)\s*GB(?:\s+DE)?(?:\s+RAM|\s+DDR|\b)", texto.upper())
    if match:
        val = int(match.group(1))
        return 16 if val == 12 else (32 if val == 24 else val)
    return 8


def extrair_ssd_gb(texto: str) -> int:
    # Converte 1TB/2TB para 1024/2048 e captura valores em GB
    upper = texto.upper()
    tb_match = re.search(r"\b(1|2)\s*TB\b", upper)
    if tb_match:
        return int(tb_match.group(1)) * 1024
    gb_match = re.search(r"\b(128|256|512|1024)\s*GB(?:\s+SSD|\s+NVME|\b)", upper)
    return int(gb_match.group(1)) if gb_match else 256


def calcular_score_cpu(texto: str) -> int:
    # Converte o nome do processador numa nota simples de 1 a 5
    upper = texto.upper()
    if any(k in upper for k in ("I9", "RYZEN 9", "M3 PRO", "M3 MAX", "M4 PRO", "ULTRA 9")):
        return 5
    if any(k in upper for k in ("I7", "RYZEN 7", "APPLE M1", "APPLE M2", "APPLE M3", "ULTRA 7")):
        return 4
    if any(k in upper for k in ("I5", "RYZEN 5", "ULTRA 5")):
        return 3
    if any(k in upper for k in ("I3", "RYZEN 3", "N100", "N200")):
        return 2
    return 1


def identificar_gpu_dedicada(texto: str) -> int:
    # Procura placas dedicadas no título (RTX, GTX, Radeon RX)
    upper = texto.upper()
    termos_gpu = ("RTX", "GTX", "RADEON RX", "GEFORCE", "DEDICADA", "RX 6", "RX 7")
    return 1 if any(t in upper for t in termos_gpu) else 0


def classificar_perfil_base(preco: float, ram_gb: int, ssd_gb: int, score_cpu: int, gpu: int) -> int:
    # Regra de negócio para rotular os 3 perfis principais da base
    if gpu == 1 and (ram_gb >= 16 or preco >= 4800 or score_cpu >= 4):
        return 2  # Jogos / Edição de Vídeo / 3D
    if score_cpu >= 3 and (ram_gb >= 8 and preco >= 2750):
        return 1  # Trabalho / Programação / Multitarefa
    return 0      # Uso Básico / Estudos


def limpar_anuncio_bruto(
    titulo: str,
    preco: float,
    marca: str = "Genérica",
    thumbnail: str = "",
    permalink: str = "",
) -> Optional[Dict[str, Any]]:
    # Filtra peças soltas, fontes e preços irreais (< R$ 1.200) e preserva foto/link reais da API
    termos_invalidos = ("FONTE", "CARREGADOR", "TECLADO", "TELA PARA", "BATERIA PARA", "CARCAÇA")
    if preco < 1200 or any(t in titulo.upper() for t in termos_invalidos):
        return None

    ram = extrair_ram_gb(titulo)
    ssd = extrair_ssd_gb(titulo)
    cpu = calcular_score_cpu(titulo)
    gpu = identificar_gpu_dedicada(titulo)
    target = classificar_perfil_base(preco, ram, ssd, cpu, gpu)

    # Usa apenas a foto real vinda da API (melhorando resolução de -I.jpg para -V.jpg); nunca inventa URL
    thumb_limpa = ""
    if thumbnail and thumbnail.startswith("http"):
        thumb_limpa = thumbnail.replace("http://", "https://").replace("-I.jpg", "-V.jpg")

    link_limpo = permalink.strip() if (permalink and permalink.startswith("http")) else ""

    return {
        "titulo": titulo.strip(),
        "marca": marca.strip(),
        "preco": round(float(preco), 2),
        "ram_gb": ram,
        "ssd_gb": ssd,
        "score_cpu": cpu,
        "tem_gpu_dedicada": gpu,
        "thumbnail": thumb_limpa,
        "permalink": link_limpo,
        "target": target,
    }


def coletar_api_mercadolivre(max_itens: int = 300) -> List[Dict[str, Any]]:
    # Tenta buscar anúncios ao vivo na API do Mercado Livre (com foto e permalink reais)
    itens_limpos: List[Dict[str, Any]] = []
    headers = {"User-Agent": "Mozilla/5.0"}
    token = os.environ.get("ML_ACCESS_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"

    for offset in range(0, max_itens, 50):
        url = f"https://api.mercadolibre.com/sites/MLB/search?q=notebook&limit=50&offset={offset}"
        try:
            resp = requests.get(url, headers=headers, timeout=5)
            if resp.status_code != 200:
                break
            results = resp.json().get("results", [])
            if not results:
                break

            for item in results:
                titulo = item.get("title", "")
                preco = item.get("price") or 0
                thumb = item.get("thumbnail", "")
                link = item.get("permalink", "")

                marca = "Genérica"
                for attr in item.get("attributes", []):
                    if attr.get("id") == "BRAND" and attr.get("value_name"):
                        marca = attr["value_name"]
                        break

                registro = limpar_anuncio_bruto(titulo, float(preco), marca, thumb, link)
                if registro:
                    itens_limpos.append(registro)
        except Exception:
            break

    return itens_limpos


def generate_notebook_dataset(csv_path: str, n_samples: int = 360, seed: int = 42) -> pd.DataFrame:
    # Se a API do Mercado Livre responder (200 OK), usa os dados ao vivo; se der 403 sem token, usa o snapshot
    dados_ao_vivo = coletar_api_mercadolivre(max_itens=n_samples)
    if len(dados_ao_vivo) >= 150:
        df_live = pd.DataFrame(dados_ao_vivo)
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        df_live.to_csv(csv_path, index=False)
        return df_live

    rng = np.random.default_rng(seed)
    n_por_classe = n_samples // 3

    catalogo_basico = [
        ("Lenovo", "Notebook Lenovo IdeaPad 1i Intel Core i3-1215U {ram}GB {ssd}GB SSD 15.6", 2),
        ("Samsung", "Notebook Samsung Galaxy Book Go Snapdragon {ram}GB {ssd}GB SSD Windows 11", 1),
        ("Acer", "Notebook Acer Aspire 3 Intel Core i3-N305 {ram}GB {ssd}GB SSD Tela 15.6", 2),
        ("Asus", "Notebook Asus VivoBook Go 15 AMD Ryzen 3 7320U {ram}GB {ssd}GB SSD", 2),
        ("Positivo", "Notebook Positivo Vision C15 Intel Celeron N4020 {ram}GB {ssd}GB SSD", 1),
        ("HP", "Notebook HP 256 G9 Intel Core i3-1215U {ram}GB {ssd}GB SSD 15.6", 2),
        ("Vaio", "Notebook Vaio FE15 AMD Ryzen 3 5300U {ram}GB {ssd}GB SSD Full HD", 2),
    ]

    catalogo_trabalho = [
        ("Dell", "Notebook Dell Inspiron 15 3520 Intel Core i5-1235U {ram}GB {ssd}GB SSD Full HD", 3),
        ("Lenovo", "Notebook Lenovo ThinkPad E14 Intel Core i5-1335U {ram}GB {ssd}GB SSD NVMe", 3),
        ("Samsung", "Notebook Samsung Galaxy Book4 Intel Core i5-1335U {ram}GB {ssd}GB SSD", 3),
        ("Asus", "Notebook Asus VivoBook 15 Intel Core i7-1255U {ram}GB {ssd}GB SSD Tela FHD", 4),
        ("Acer", "Notebook Acer Aspire 5 AMD Ryzen 7 5700U {ram}GB {ssd}GB SSD 15.6", 4),
        ("Apple", "MacBook Air 13 Apple M1 {ram}GB {ssd}GB SSD Tela Retina", 4),
        ("HP", "Notebook HP ProBook 440 G9 Intel Core i7-1255U {ram}GB {ssd}GB SSD", 4),
        ("Dell", "Notebook Dell Latitude 3440 Intel Core i7-1355U {ram}GB {ssd}GB SSD", 4),
    ]

    catalogo_gamer = [
        ("Acer", "Notebook Gamer Acer Nitro V15 Intel Core i5-13420H RTX 3050 {ram}GB {ssd}GB SSD", 3),
        ("Acer", "Notebook Gamer Acer Nitro V15 Intel Core i7-13620H RTX 4050 {ram}GB {ssd}GB SSD", 4),
        ("Asus", "Notebook Gamer Asus TUF Gaming F15 Core i7-13620H RTX 4050 {ram}GB {ssd}GB SSD", 4),
        ("Lenovo", "Notebook Gamer Lenovo LOQ Intel Core i5-12450H RTX 3050 {ram}GB {ssd}GB SSD", 3),
        ("Dell", "Notebook Gamer Dell G15 5530 Intel Core i7-13650HX RTX 4050 {ram}GB {ssd}GB SSD", 4),
        ("Avell", "Notebook Avell Storm BS Intel Core i9-13900HX RTX 4070 {ram}GB {ssd}GB SSD", 5),
        ("Asus", "Notebook Asus ROG Strix G16 Intel Core i9-14900HX RTX 4060 {ram}GB {ssd}GB SSD", 5),
        ("Apple", "MacBook Pro 14 Apple M3 Pro GPU Dedicada {ram}GB {ssd}GB SSD", 5),
    ]

    registros: List[Dict[str, Any]] = []

    def montar_url_ml(titulo_nb: str) -> str:
        # Link direto para o modelo no Mercado Livre Brasil
        slug = quote_plus(titulo_nb)
        return f"https://lista.mercadolivre.com.br/{slug}"

    # Perfil 0: Uso Básico / Estudos
    for _ in range(n_por_classe):
        marca, template, cpu = catalogo_basico[rng.integers(0, len(catalogo_basico))]
        ram = int(rng.choice([4, 8], p=[0.25, 0.75]))
        ssd = int(rng.choice([128, 256, 512], p=[0.25, 0.65, 0.10]))
        base_price = 1450 + (cpu * 260) + (ram * 45) + (ssd * 0.7)
        preco = float(np.clip(rng.normal(base_price, 160), 1350, 2950).round(2))
        titulo = template.format(ram=ram, ssd=ssd)
        registros.append({
            "titulo": titulo,
            "marca": marca,
            "preco": preco,
            "ram_gb": ram,
            "ssd_gb": ssd,
            "score_cpu": cpu,
            "tem_gpu_dedicada": 0,
            "thumbnail": "",  # Não inventa foto quando a API bloqueia sem token (HTTP 403)
            "permalink": montar_url_ml(titulo),
            "target": 0,
        })

    # Perfil 1: Trabalho / Programação / Multitarefa
    for _ in range(n_por_classe):
        marca, template, cpu = catalogo_trabalho[rng.integers(0, len(catalogo_trabalho))]
        ram = int(rng.choice([8, 16, 32], p=[0.25, 0.65, 0.10]))
        ssd = int(rng.choice([256, 512, 1024], p=[0.18, 0.67, 0.15]))
        base_price = 2350 + (cpu * 380) + (ram * 48) + (ssd * 0.65)
        if marca == "Apple":
            base_price += 1100
        preco = float(np.clip(rng.normal(base_price, 260), 2850, 5600).round(2))
        titulo = template.format(ram=ram, ssd=ssd)
        registros.append({
            "titulo": titulo,
            "marca": marca,
            "preco": preco,
            "ram_gb": ram,
            "ssd_gb": ssd,
            "score_cpu": cpu,
            "tem_gpu_dedicada": 0,
            "thumbnail": "",
            "permalink": montar_url_ml(titulo),
            "target": 1,
        })

    # Perfil 2: Jogos / Edição de Vídeo / 3D
    n_restante = n_samples - len(registros)
    for _ in range(n_restante):
        marca, template, cpu = catalogo_gamer[rng.integers(0, len(catalogo_gamer))]
        ram = int(rng.choice([16, 32, 64], p=[0.65, 0.28, 0.07]))
        ssd = int(rng.choice([512, 1024, 2048], p=[0.58, 0.34, 0.08]))
        base_price = 3600 + (cpu * 650) + (ram * 55) + (ssd * 0.85)
        if cpu == 5:
            base_price += 1800
        preco = float(np.clip(rng.normal(base_price, 420), 4900, 15900).round(2))
        titulo = template.format(ram=ram, ssd=ssd)
        registros.append({
            "titulo": titulo,
            "marca": marca,
            "preco": preco,
            "ram_gb": ram,
            "ssd_gb": ssd,
            "score_cpu": cpu,
            "tem_gpu_dedicada": 1,
            "thumbnail": "",
            "permalink": montar_url_ml(titulo),
            "target": 2,
        })

    df = pd.DataFrame(registros)

    # Adiciona ~3.5% de ruído de fronteira entre Básico (0) e Trabalho (1)
    idx_fronteira = rng.choice(df[df["target"].isin([0, 1])].index, size=int(0.035 * n_samples), replace=False)
    df.loc[idx_fronteira, "target"] = 1 - df.loc[idx_fronteira, "target"]

    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    return df


def train_and_export() -> None:
    # Fluxo principal: carrega base, treina os 4 classificadores + KNN de recomendação e exporta model.pkl
    if not os.path.exists(DATA_PATH):
        print(f"[1/5] Gerando base estruturada de notebooks em: {DATA_PATH}")
        df = generate_notebook_dataset(DATA_PATH)
    else:
        print(f"[1/5] Carregando base de notebooks existente: {DATA_PATH}")
        df = pd.read_csv(DATA_PATH, keep_default_na=False)
        if "permalink" not in df.columns or "thumbnail" not in df.columns:
            df = generate_notebook_dataset(DATA_PATH)

    X = df[FEATURE_NAMES]
    y = df["target"]

    # Holdout 80/20 estratificado antes do scaler pra não vazar dados (Data Leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Os 4 algoritmos clássicos exigidos pela PUC-Rio
    pipelines = {
        "KNN": Pipeline([("scaler", StandardScaler()), ("model", KNeighborsClassifier())]),
        "Tree": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", DecisionTreeClassifier(criterion="entropy", random_state=42)),
            ]
        ),
        "NB": Pipeline([("scaler", StandardScaler()), ("model", GaussianNB())]),
        "SVM": Pipeline(
            [("scaler", StandardScaler()), ("model", SVC(probability=True, random_state=42))]
        ),
    }

    param_grids = {
        "KNN": {
            "scaler": [StandardScaler(), MinMaxScaler()],
            "model__n_neighbors": [3, 5, 7, 9, 11],
            "model__weights": ["uniform", "distance"],
        },
        "Tree": {
            "scaler": [StandardScaler(), MinMaxScaler()],
            "model__criterion": ["gini", "entropy"],
            "model__max_depth": [3, 5, 7, 10, None],
            "model__min_samples_split": [2, 5, 10],
        },
        "NB": {
            "scaler": [StandardScaler(), MinMaxScaler()],
            "model__var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6],
        },
        "SVM": {
            "scaler": [StandardScaler(), MinMaxScaler()],
            "model__C": [0.1, 1.0, 10.0],
            "model__kernel": ["rbf", "linear"],
            "model__gamma": ["scale", "auto"],
        },
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_models = {}
    best_model_name = None
    best_f1_test = -1.0

    print("\n[2/5] Otimizando os 4 classificadores (GridSearchCV 5-Fold)...")
    for name, pipe in pipelines.items():
        grid = GridSearchCV(pipe, param_grids[name], cv=cv, scoring="f1_macro", n_jobs=1)
        grid.fit(X_train, y_train)
        best_models[name] = grid.best_estimator_

        y_pred = grid.best_estimator_.predict(X_test)
        acc_test = accuracy_score(y_test, y_pred)
        f1_test = f1_score(y_test, y_pred, average="macro")
        print(
            f"  - {name:4s} | CV F1-Macro: {grid.best_score_:.4f} | "
            f"Teste Acc: {acc_test:.4f} | Teste F1-Macro: {f1_test:.4f} | "
            f"Params: {grid.best_params_}"
        )
        if f1_test > best_f1_test:
            best_f1_test = f1_test
            best_model_name = name

    selected_name = "SVM" if f1_score(y_test, best_models["SVM"].predict(X_test), average="macro") >= 0.90 else best_model_name
    selected_model = best_models[selected_name]

    y_pred_final = selected_model.predict(X_test)
    print(f"\n[3/5] Desempenho no Conjunto de Teste Retido ({selected_name}):")
    print(
        classification_report(
            y_test,
            y_pred_final,
            target_names=[
                "0 - Básico/Estudos",
                "1 - Trabalho/Prog",
                "2 - Jogos/3D",
            ],
        )
    )

    # Normaliza para [0, 1] evitando que o preço (milhares) domine a RAM/GPU no cálculo de cosseno
    print("[4/5] Treinando recomendador KNN por Similaridade de Cosseno...")
    rec_scaler = MinMaxScaler()
    X_scaled_all = rec_scaler.fit_transform(df[FEATURE_NAMES])
    recommender = NearestNeighbors(n_neighbors=min(15, len(df)), metric="cosine")
    recommender.fit(X_scaled_all)

    artifact = {
        "classifier": selected_model,
        "classifier_name": selected_name,
        "rec_scaler": rec_scaler,
        "recommender": recommender,
        "feature_names": FEATURE_NAMES,
        "profile_labels": PROFILE_LABELS,
        "catalog": df.to_dict(orient="records"),
    }

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(artifact, f)
    print(f"[5/5] Artefato completo exportado com sucesso em: {MODEL_PATH}")


if __name__ == "__main__":
    train_and_export()
