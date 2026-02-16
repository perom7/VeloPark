# Starts Celery worker (solo pool for Windows)
$ErrorActionPreference = 'Stop'
$python = Join-Path $PSScriptRoot '..\.venv\Scripts\python.exe'
if (!(Test-Path $python)) { throw "Python venv not found at $python" }
& $python '-m' 'celery' '-A' 'backend.tasks.celery' 'worker' '-l' 'info' '-P' 'solo'