"""
Batería de experimentos para comparar en MLflow.

Cada configuración se ejecuta como un run independiente (subproceso fresco) con
sus parámetros (rerank_method, top_n, pool_size). Así quedan registrados,
comparables y graficables en MLflow.

Uso:
    python experiments.py                 # todos los experimentos, 3 muestras
    python experiments.py --samples 5     # 5 muestras por experimento
    python experiments.py --mode all      # incluye RAGAS + corpus (mas lento)
"""

import os
import sys
import time
import argparse
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON   = sys.executable

# Cada dict es un run. La clave 'run_name' es el nombre en MLflow; el resto son
# variables de entorno que cambian el comportamiento del pipeline.
EXPERIMENTS = [
    {"run_name": "rerank_none",         "RERANK_METHOD": "none",         "RETRIEVAL_POOL_SIZE": "10", "RETRIEVAL_TOP_N": "5"},
    {"run_name": "rerank_mmr",          "RERANK_METHOD": "mmr",          "RETRIEVAL_POOL_SIZE": "10", "RETRIEVAL_TOP_N": "5"},
    {"run_name": "rerank_crossencoder", "RERANK_METHOD": "crossencoder", "RETRIEVAL_POOL_SIZE": "10", "RETRIEVAL_TOP_N": "5"},
    {"run_name": "mmr_topn_3",          "RERANK_METHOD": "mmr",          "RETRIEVAL_POOL_SIZE": "10", "RETRIEVAL_TOP_N": "3"},
    {"run_name": "mmr_topn_8",          "RERANK_METHOD": "mmr",          "RETRIEVAL_POOL_SIZE": "12", "RETRIEVAL_TOP_N": "8"},
]


def run_one(exp: dict, samples: int, mode: str) -> bool:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"]  = "1"
    for key, val in exp.items():
        if key != "run_name":
            env[key] = val

    cmd = [
        PYTHON, "-m", "evaluation.runner",
        "--mode", mode,
        "--run-name", exp["run_name"],
    ]
    if samples:  # si es None/0 se usan todas las muestras del dataset
        cmd += ["--samples", str(samples)]
    result = subprocess.run(cmd, cwd=BASE_DIR, env=env)
    return result.returncode == 0


def main():
    ap = argparse.ArgumentParser(description="Batería de experimentos NotionMap")
    ap.add_argument("--samples", type=int, default=None,
                    help="Muestras por experimento (default: todas las del dataset)")
    ap.add_argument("--mode", default="judge", choices=["judge", "all"],
                    help="judge = LLM Judge (rapido); all = RAGAS + corpus tambien")
    args = ap.parse_args()

    print("=" * 60)
    print(f"  BATERIA DE EXPERIMENTOS — {len(EXPERIMENTS)} configuraciones")
    print(f"  samples={args.samples or 'todas'}  mode={args.mode}")
    print("=" * 60)

    t0 = time.time()
    ok = 0
    for i, exp in enumerate(EXPERIMENTS, 1):
        print(f"\n[{i}/{len(EXPERIMENTS)}] {exp['run_name']}  "
              f"(rerank={exp['RERANK_METHOD']}, top_n={exp['RETRIEVAL_TOP_N']}, "
              f"pool={exp['RETRIEVAL_POOL_SIZE']})")
        if run_one(exp, args.samples, args.mode):
            ok += 1
        else:
            print(f"  [!] El experimento '{exp['run_name']}' termino con error.")

    mins = round((time.time() - t0) / 60, 1)
    print("\n" + "=" * 60)
    print(f"  COMPLETADO: {ok}/{len(EXPERIMENTS)} experimentos en {mins} min")
    print("=" * 60)
    print("\n  Ver resultados en MLflow:")
    print("    python -m mlflow ui --backend-store-uri sqlite:///mlflow.db")
    print("    -> http://localhost:5000  (pestana 'Model training' -> Compare)")


if __name__ == "__main__":
    main()
