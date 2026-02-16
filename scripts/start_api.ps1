# Starts the Flask API using the workspace venv
$ErrorActionPreference = 'Stop'
$python = Join-Path $PSScriptRoot '..\.venv\Scripts\python.exe'
if (!(Test-Path $python)) { throw "Python venv not found at $python" }
& $python '..\backend\app.py'