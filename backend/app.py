# Servidor Flask: expõe as rotas da API (/api/predict, /api/dataset, /train) e serve o Front-end.

import json
import os
import subprocess
import sys
import time
import pandas as pd
from flask import Flask, Response, jsonify, request, send_from_directory, stream_with_context
from flask_cors import CORS

# Funciona tanto rodando `python backend/app.py` quanto `python app.py` dentro de backend/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from backend.model_service import ModelService
except ImportError:
    from model_service import ModelService

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "frontend"))
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "data"))
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
DEFAULT_CSV_PATH = os.path.join(DATA_DIR, "notebooks.csv")
TRAIN_SCRIPT_PATH = os.path.join(BASE_DIR, "train_model.py")

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")
CORS(app)


def get_model_service() -> ModelService:
    # Retorna a instância única (Singleton) do serviço de recomendação
    return ModelService(DEFAULT_MODEL_PATH)


# Carrega o modelo em memória uma vez só na subida do servidor
if os.path.exists(DEFAULT_MODEL_PATH):
    get_model_service()


@app.route("/", methods=["GET"])
def serve_index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/train", methods=["GET"])
def serve_train_page():
    return send_from_directory(FRONTEND_DIR, "train.html")


@app.route("/api/train/stream", methods=["GET"])
def train_model_stream():
    # Roda o train_model.py e envia o log linha a linha pro navegador via SSE
    def generate_events():
        start_time = time.time()
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"

        yield f"data: {json.dumps({'type': 'start', 'message': 'Iniciando execução de backend/train_model.py...'})}\n\n"

        try:
            proc = subprocess.Popen(
                [sys.executable, "-u", TRAIN_SCRIPT_PATH],
                cwd=os.path.abspath(os.path.join(BASE_DIR, "..")),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
            )

            if proc.stdout:
                for raw_line in iter(proc.stdout.readline, ""):
                    line = raw_line.rstrip("\r\n")
                    yield f"data: {json.dumps({'type': 'log', 'line': line})}\n\n"

            return_code = proc.wait()
            duration = round(time.time() - start_time, 2)

            if return_code == 0:
                if os.path.exists(DEFAULT_MODEL_PATH):
                    get_model_service().reload()
                yield f"data: {json.dumps({'type': 'done', 'status': 'sucesso', 'exit_code': 0, 'duration_s': duration, 'message': 'Treinamento concluído e modelo recarregado em memória!'})}\n\n"
            else:
                yield f"data: {json.dumps({'type': 'done', 'status': 'erro', 'exit_code': return_code, 'duration_s': duration, 'message': f'O script encerrou com código de erro {return_code}.'})}\n\n"
        except Exception as exc:
            duration = round(time.time() - start_time, 2)
            yield f"data: {json.dumps({'type': 'done', 'status': 'erro', 'exit_code': -1, 'duration_s': duration, 'message': f'Falha ao executar o treinamento: {str(exc)}'})}\n\n"

    return Response(
        stream_with_context(generate_events()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/api/train", methods=["POST"])
def train_model_sync():
    # Versão síncrona do retreinamento pra chamadas diretas via API
    start_time = time.time()
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    try:
        completed = subprocess.run(
            [sys.executable, "-u", TRAIN_SCRIPT_PATH],
            cwd=os.path.abspath(os.path.join(BASE_DIR, "..")),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            check=False,
        )
        duration = round(time.time() - start_time, 2)
        output_log = (completed.stdout or "") + (completed.stderr or "")

        if completed.returncode == 0:
            if os.path.exists(DEFAULT_MODEL_PATH):
                get_model_service().reload()
            return (
                jsonify(
                    {
                        "status": "sucesso",
                        "exit_code": 0,
                        "duration_s": duration,
                        "log": output_log,
                    }
                ),
                200,
            )
        return (
            jsonify(
                {
                    "status": "erro",
                    "exit_code": completed.returncode,
                    "duration_s": duration,
                    "log": output_log,
                }
            ),
            500,
        )
    except Exception as exc:
        duration = round(time.time() - start_time, 2)
        return (
            jsonify(
                {
                    "status": "erro",
                    "exit_code": -1,
                    "duration_s": duration,
                    "log": f"Erro ao iniciar subprocesso de treinamento: {str(exc)}",
                }
            ),
            500,
        )


@app.route("/api/health", methods=["GET"])
def health_check():
    model_loaded = os.path.exists(DEFAULT_MODEL_PATH)
    return (
        jsonify(
            {
                "status": "online",
                "service": "ML Notebook Recommender & Market Trends",
                "model_loaded": model_loaded,
            }
        ),
        200,
    )


@app.route("/api/dataset", methods=["GET"])
def dataset_info():
    # Retorna resumo e amostra da base limpa de notebooks (sem dados sensíveis de vendedores)
    if not os.path.exists(DEFAULT_CSV_PATH):
        return jsonify({"erro": "Base de dados de notebooks não encontrada."}), 404

    df = pd.read_csv(DEFAULT_CSV_PATH, keep_default_na=False)
    limit = min(int(request.args.get("limit", 15)), 100)
    return (
        jsonify(
            {
                "fonte": "Catálogo Mercado Livre BR — Notebooks Padronizados via Regex (Base Anonimizada)",
                "total_registros": int(len(df)),
                "colunas": list(df.columns),
                "distribuicao_classes": {
                    "basico_estudos_0": int((df["target"] == 0).sum()),
                    "trabalho_programacao_1": int((df["target"] == 1).sum()),
                    "jogos_3d_2": int((df["target"] == 2).sum()),
                },
                "amostra": df.head(limit).to_dict(orient="records"),
            }
        ),
        200,
    )


@app.route("/api/dataset/csv", methods=["GET"])
def dataset_csv():
    return send_from_directory(DATA_DIR, "notebooks.csv", mimetype="text/csv")


@app.route("/api/predict", methods=["POST"])
def predict():
    # Recebe [preco, ram_gb, ssd_gb, score_cpu, tem_gpu_dedicada] e retorna perfil + Top-5 + tendências
    data = request.get_json(silent=True)
    if not data or "features" not in data:
        return (
            jsonify(
                {
                    "erro": "Payload inválido. O campo 'features' é obrigatório."
                }
            ),
            400,
        )

    features = data["features"]
    if not isinstance(features, list) or len(features) != 5:
        return (
            jsonify(
                {
                    "erro": "Esperados exatamente 5 atributos numéricos na lista 'features'."
                }
            ),
            400,
        )

    try:
        service = get_model_service()
        resultado = service.predict(features)
        return jsonify(resultado), 200
    except ValueError as val_err:
        return jsonify({"erro": str(val_err)}), 400
    except Exception:
        # DevSecOps/LGPD: não grava payloads financeiros do usuário em logs de erro
        return (
            jsonify(
                {
                    "erro": "Falha interna ao processar a recomendação de notebooks."
                }
            ),
            500,
        )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
