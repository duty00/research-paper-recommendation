param(
    [string]$DataDir = 'assignment-data-mining-261',
    [string]$OutputDir = 'outputs\my_run01'
)
$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
        throw 'Create .venv and install requirements.txt first; see README.md.'
    }
    & '.\.venv\Scripts\python.exe' main.py run --data-dir $DataDir --out $OutputDir
    if ($LASTEXITCODE -ne 0) { throw 'Training failed. Review the error above.' }
} finally {
    Pop-Location
}
