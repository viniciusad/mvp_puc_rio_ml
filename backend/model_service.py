# Camada de serviço (SOLID/DRY): carrega o model.pkl uma única vez e entrega classificação, Top-5 KNN e tendências.

import os
import pickle
from collections import Counter
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


class ModelService:
    # Singleton pra não recarregar o modelo do disco a cada requisição HTTP
    _instance: Optional["ModelService"] = None
    _artifact: Any = None
    _model_path: Optional[str] = None

    FEATURE_NAMES: List[str] = [
        "preco",
        "ram_gb",
        "ssd_gb",
        "score_cpu",
        "tem_gpu_dedicada",
    ]

    PROFILE_LABELS: Dict[int, str] = {
        0: "Uso Básico / Estudos",
        1: "Trabalho / Programação / Multitarefa",
        2: "Jogos / Edição de Vídeo / 3D",
    }

    def __new__(cls, model_path: Optional[str] = None) -> "ModelService":
        resolved_path = cls._resolve_path(model_path)
        if cls._instance is None or cls._artifact is None or cls._model_path != resolved_path:
            instance = super(ModelService, cls).__new__(cls)
            instance._load_model(resolved_path)
            cls._instance = instance
            cls._model_path = resolved_path
        return cls._instance

    @staticmethod
    def _resolve_path(model_path: Optional[str] = None) -> str:
        # Resolve o caminho do model.pkl independente de onde o script foi chamado
        if model_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            return os.path.join(base_dir, "model.pkl")
        if os.path.exists(model_path):
            return os.path.abspath(model_path)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        candidate = os.path.join(base_dir, os.path.basename(model_path))
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
        return os.path.abspath(model_path)

    def _load_model(self, model_path: str) -> None:
        # Lê o artefato binário (.pkl) com classificador, scaler, KNN cosseno e catálogo
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo não encontrado no caminho: {model_path}")
        with open(model_path, "rb") as file:
            self._artifact = pickle.load(file)

    def reload(self) -> None:
        # Atualiza o modelo em memória logo após rodar o /train
        if self._model_path:
            self._load_model(self._model_path)

    def _recomendar_top5(
        self, input_df: pd.DataFrame, numeric_features: List[float], predicted_class: int
    ) -> List[Dict[str, Any]]:
        # Aplica o mesmo MinMaxScaler do treino e busca os 5 notebooks mais próximos por cosseno
        rec_scaler = self._artifact["rec_scaler"]
        recommender = self._artifact["recommender"]
        catalog = self._artifact["catalog"]

        user_scaled = rec_scaler.transform(input_df)
        distances, indices = recommender.kneighbors(user_scaled)

        candidatos: List[Dict[str, Any]] = []
        exige_gamer = predicted_class == 2 or int(round(numeric_features[4])) == 1

        for dist, idx in zip(distances[0], indices[0]):
            item = catalog[int(idx)]
            # Garante regra de negócio: perfil Jogos/3D só recomenda máquina com GPU dedicada e >= 16GB RAM
            if exige_gamer and (int(item["tem_gpu_dedicada"]) != 1 or int(item["ram_gb"]) < 16):
                continue

            compat = float(np.clip(1.0 - float(dist), 0.0, 1.0))
            thumb_val = str(item.get("thumbnail") or "").strip()
            if thumb_val == "nan":
                thumb_val = ""
            link_val = str(item.get("permalink") or "").strip()
            if link_val == "nan" or not link_val:
                from urllib.parse import quote_plus
                link_val = f"https://lista.mercadolivre.com.br/{quote_plus(str(item['titulo']))}"

            candidatos.append({
                "ranking": len(candidatos) + 1,
                "titulo": item["titulo"],
                "marca": item["marca"],
                "preco": round(float(item["preco"]), 2),
                "ram_gb": int(item["ram_gb"]),
                "ssd_gb": int(item["ssd_gb"]),
                "score_cpu": int(item["score_cpu"]),
                "tem_gpu_dedicada": int(item["tem_gpu_dedicada"]),
                "thumbnail": thumb_val,
                "permalink": link_val,
                "perfil_id": int(item["target"]),
                "perfil_nome": self.PROFILE_LABELS.get(int(item["target"]), "Geral"),
                "distancia_cosseno": round(float(dist), 4),
                "compatibilidade": round(compat, 4),
                "compatibilidade_pct": round(compat * 100.0, 1),
            })
            if len(candidatos) == 5:
                break

        return candidatos

    def _calcular_tendencias(
        self, orcamento_usuario: float, predicted_class: int
    ) -> Dict[str, Any]:
        # Agrega estatísticas reais do catálogo para mostrar tendências do segmento escolhido
        catalog = self._artifact["catalog"]
        itens_perfil = [row for row in catalog if int(row["target"]) == predicted_class]
        if not itens_perfil:
            itens_perfil = catalog

        precos = [float(r["preco"]) for r in itens_perfil]
        rams = [int(r["ram_gb"]) for r in itens_perfil]
        ssds = [int(r["ssd_gb"]) for r in itens_perfil]
        cpus = [int(r["score_cpu"]) for r in itens_perfil]
        gpus = [int(r["tem_gpu_dedicada"]) for r in itens_perfil]

        preco_medio = round(float(np.mean(precos)), 2)
        preco_min = round(float(np.min(precos)), 2)
        preco_max = round(float(np.max(precos)), 2)

        contagem_marcas = Counter(r["marca"] for r in itens_perfil)
        total_cat = len(itens_perfil)
        marcas_dominantes = [
            {
                "marca": marca,
                "quantidade": qtd,
                "participacao_pct": round((qtd / total_cat) * 100.0, 1),
            }
            for marca, qtd in contagem_marcas.most_common(3)
        ]

        diff_pct = round(((orcamento_usuario - preco_medio) / preco_medio) * 100.0, 1)
        if diff_pct >= 10.0:
            status_orc = "Acima da média da categoria (folga para configurações topo de linha)"
        elif diff_pct <= -12.0:
            status_orc = "Abaixo da média da categoria (foco em modelos de entrada/promoções)"
        else:
            status_orc = "Alinhado à média praticada no mercado para este perfil"

        return {
            "perfil_analisado": self.PROFILE_LABELS.get(predicted_class, "Geral"),
            "total_anuncios_perfil": total_cat,
            "preco_medio_categoria": preco_medio,
            "preco_min_categoria": preco_min,
            "preco_max_categoria": preco_max,
            "marcas_dominantes": marcas_dominantes,
            "configuracao_dominante": {
                "ram_gb_tipica": Counter(rams).most_common(1)[0][0],
                "ssd_gb_tipico": Counter(ssds).most_common(1)[0][0],
                "score_cpu_medio": round(float(np.mean(cpus)), 1),
                "gpu_dedicada_pct": round((sum(gpus) / total_cat) * 100.0, 1),
            },
            "analise_orcamento": {
                "orcamento_informado": round(float(orcamento_usuario), 2),
                "diferenca_pct_media": diff_pct,
                "status": status_orc,
            },
        }

    def predict(self, features: List[float]) -> Dict[str, Any]:
        # Valida os 5 atributos de entrada e devolve os 3 blocos: perfil, Top-5 notebooks e tendências
        if not isinstance(features, (list, tuple, np.ndarray)) or len(features) != len(self.FEATURE_NAMES):
            raise ValueError(
                f"Esperados exatamente {len(self.FEATURE_NAMES)} atributos numéricos: "
                f"{', '.join(self.FEATURE_NAMES)}."
            )

        try:
            numeric_features = [float(val) for val in features]
        except (TypeError, ValueError) as exc:
            raise ValueError("Todos os atributos enviados devem ser valores numéricos válidos.") from exc

        if not np.all(np.isfinite(numeric_features)):
            raise ValueError("Os atributos não podem conter valores NaN ou infinitos.")

        input_df = pd.DataFrame([numeric_features], columns=self.FEATURE_NAMES)

        classifier = self._artifact["classifier"] if isinstance(self._artifact, dict) else self._artifact
        prediction = int(classifier.predict(input_df)[0])

        prob: Optional[float] = None
        if hasattr(classifier, "predict_proba"):
            probabilities = classifier.predict_proba(input_df)
            prob = round(float(np.max(probabilities)), 4)

        label = self.PROFILE_LABELS.get(prediction, "Uso Básico / Estudos")

        recomendacoes: List[Dict[str, Any]] = []
        tendencias: Dict[str, Any] = {}
        classifier_name = "SVM"

        if isinstance(self._artifact, dict) and "recommender" in self._artifact:
            classifier_name = self._artifact.get("classifier_name", "SVM")
            recomendacoes = self._recomendar_top5(input_df, numeric_features, prediction)
            tendencias = self._calcular_tendencias(numeric_features[0], prediction)

        return {
            "predicao": label,
            "classe": prediction,
            "confianca": prob,
            "algoritmo": classifier_name,
            "recomendacoes": recomendacoes,
            "tendencias": tendencias,
        }
