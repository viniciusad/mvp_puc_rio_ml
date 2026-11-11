# Testes automatizados (PyTest): Quality Gate Duplo (métricas da PUC-Rio + coerência das recomendações KNN).

import os
import sys
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from backend.app import app
from backend.model_service import ModelService

THRESHOLD_ACURACIA_MINIMA = 0.80
THRESHOLD_F1_MINIMO = 0.80
THRESHOLD_COMPATIBILIDADE_MINIMA = 0.80


@pytest.fixture
def service() -> ModelService:
    # Carrega o artefato de produção model.pkl
    model_path = os.path.join(ROOT_DIR, "backend", "model.pkl")
    return ModelService(model_path)


@pytest.fixture
def dataset():
    # Separa o conjunto de teste (Holdout 80/20) exatamente igual ao treino
    csv_path = os.path.join(ROOT_DIR, "data", "notebooks.csv")
    df = pd.read_csv(csv_path)
    X = df[ModelService.FEATURE_NAMES]
    y = df["target"]
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    return X_test.values, y_test.values


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as test_client:
        yield test_client


def test_desempenho_modelo_quality_gate(service: ModelService, dataset):
    # Quality Gate 1: barra o modelo se Acurácia ou F1-Macro ficarem abaixo de 80%
    X_test, y_true = dataset
    y_pred = []

    for sample in X_test:
        res = service.predict(sample.tolist())
        y_pred.append(res["classe"])

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")

    print(
        f"\n[Quality Gate Classificação] Acurácia: {acc:.4f} (Mínima: {THRESHOLD_ACURACIA_MINIMA:.2f})"
    )
    print(
        f"[Quality Gate Classificação] F1-Macro: {f1:.4f} (Mínimo: {THRESHOLD_F1_MINIMO:.2f})"
    )

    assert acc >= THRESHOLD_ACURACIA_MINIMA, (
        f"Reprovado no Quality Gate: Acurácia {acc:.4f} abaixo de {THRESHOLD_ACURACIA_MINIMA}"
    )
    assert f1 >= THRESHOLD_F1_MINIMO, (
        f"Reprovado no Quality Gate: F1-Macro {f1:.4f} abaixo de {THRESHOLD_F1_MINIMO}"
    )


def test_coerencia_recomendacao_knn_regras_negocio(service: ModelService):
    # Quality Gate 2: garante que perfil Jogos/3D só receba notebooks com GPU dedicada e >= 16GB RAM
    perfil_estudos = [2500, 8, 256, 2, 0]
    perfil_trabalho = [4200, 16, 512, 3, 0]
    perfil_gamer = [6800, 16, 512, 4, 1]

    res_estudos = service.predict(perfil_estudos)
    res_trabalho = service.predict(perfil_trabalho)
    res_gamer = service.predict(perfil_gamer)

    assert res_estudos["classe"] == 0
    assert res_trabalho["classe"] == 1
    assert res_gamer["classe"] == 2

    recs_gamer = res_gamer["recomendacoes"]
    assert len(recs_gamer) == 5

    for nb in recs_gamer:
        assert nb["tem_gpu_dedicada"] == 1, "Recomendação Gamer/3D precisa ter GPU dedicada"
        assert nb["ram_gb"] >= 16, "Recomendação Gamer/3D precisa ter pelo menos 16GB de RAM"
        assert "permalink" in nb and nb["permalink"].startswith("https://")
        assert "thumbnail" in nb and isinstance(nb["thumbnail"], str)

    media_compat = sum(nb["compatibilidade"] for nb in recs_gamer) / len(recs_gamer)
    print(f"[Quality Gate KNN Cosseno] Compatibilidade média Top-5 Gamer: {media_compat:.4f}")
    assert media_compat >= THRESHOLD_COMPATIBILIDADE_MINIMA


def test_contrato_inferencia_model_service(service: ModelService):
    # Verifica estrutura dos 3 blocos de saída e rejeição de entradas inválidas
    res = service.predict([4200, 16, 512, 3, 0])

    assert "predicao" in res
    assert "classe" in res
    assert "confianca" in res
    assert "recomendacoes" in res
    assert "tendencias" in res
    assert res["classe"] in (0, 1, 2)
    assert 0.0 <= res["confianca"] <= 1.0
    assert len(res["recomendacoes"]) == 5
    assert "preco_medio_categoria" in res["tendencias"]

    with pytest.raises(ValueError):
        service.predict([2500, 8, 256])


def test_api_endpoints_health_e_predict(client):
    # Testa integração das rotas HTTP da aplicação Full Stack
    resp_health = client.get("/api/health")
    assert resp_health.status_code == 200
    assert resp_health.get_json()["status"] == "online"

    resp_predict = client.post(
        "/api/predict",
        json={"features": [6800, 16, 512, 4, 1]},
    )
    assert resp_predict.status_code == 200
    pred_data = resp_predict.get_json()
    assert pred_data["classe"] == 2
    assert len(pred_data["recomendacoes"]) == 5

    resp_dataset = client.get("/api/dataset?limit=5")
    assert resp_dataset.status_code == 200
    assert len(resp_dataset.get_json()["amostra"]) == 5

    resp_invalid = client.post("/api/predict", json={"features": [2500, 8]})
    assert resp_invalid.status_code == 400

    resp_train_page = client.get("/train")
    assert resp_train_page.status_code == 200
    assert b"train_model.py" in resp_train_page.data
