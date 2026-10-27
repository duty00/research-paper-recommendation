param([string]$Python = '')
$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    if (-not $Python) {
        if (Test-Path -LiteralPath '.venv\Scripts\python.exe') {
            $Python = (Resolve-Path -LiteralPath '.venv\Scripts\python.exe').Path
        } else { $Python = 'python' }
    }
    & $Python -c "import sys, sysconfig; assert sys.version_info[:2] in [(3,11),(3,12)], 'Use Python 3.11 or 3.12 (64 bit)'; assert sys.maxsize > 2**32, 'Use 64-bit Python'; assert sys.platform == 'win32' and sysconfig.get_platform() == 'win-amd64', 'Use native Windows x64 Python, not MSYS2/MinGW'; print(sys.version)"
    if ($LASTEXITCODE -ne 0) { throw 'Install Python 3.12 (64 bit), then run setup again.' }
    if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
        & $Python -m venv .venv
        if ($LASTEXITCODE -ne 0) { throw 'Could not create .venv.' }
    }
    & '.\.venv\Scripts\python.exe' -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
    & '.\.venv\Scripts\python.exe' -m unittest discover -s tests -v
    if ($LASTEXITCODE -ne 0) { throw 'Tests failed.' }
} finally {
    Pop-Location
}
