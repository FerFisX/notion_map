# Corre 3 evaluaciones variando el metodo de re-ranking para comparar en MLflow.
$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED  = "1"
Set-Location "C:\Users\afar1\OneDrive\Escritorio\notion_map"

$py  = "C:\Users\afar1\AppData\Local\Programs\Python\Python312\python.exe"
$log = "C:\Users\afar1\AppData\Local\Temp\experiments.log"
Set-Content $log "INICIANDO EXPERIMENTOS COMPARATIVOS"

foreach ($m in @("none", "mmr", "crossencoder")) {
    $env:RERANK_METHOD = $m
    Add-Content $log "`n===== rerank_$m =====`n"
    & $py -u -m evaluation.runner --mode judge --samples 3 --run-name "rerank_$m" 1>> $log 2>> $log
    Add-Content $log "`n----- fin rerank_$m -----`n"
}
Add-Content $log "`nCOMPLETADO"
