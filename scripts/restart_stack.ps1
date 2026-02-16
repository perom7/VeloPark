<#!
.SYNOPSIS
    Restart the complete Parking App stack (Redis/Memurai, Flask API, Celery worker, Celery beat).
.DESCRIPTION
    1. Ensures Memurai (Redis) service is running.
    2. Terminates any existing python.exe processes for the API or Celery tasks.
    3. Starts fresh terminals for: API, Celery worker, Celery beat.
    4. Performs a health check.
.PARAMETER NoHealth
    Skip the final health check.
.PARAMETER KillAllPython
    Kill ALL python.exe processes (use only if isolated dev box).
.NOTES
    Run from project root or directly: powershell -ExecutionPolicy Bypass -File scripts\restart_stack.ps1
#>
param(
    [switch]$NoHealth,
    [switch]$KillAllPython
)

$ErrorActionPreference = 'Stop'
Write-Host "[restart] Starting restart sequence..." -ForegroundColor Cyan

# Resolve project root
$projectRoot = Resolve-Path (Join-Path $PSScriptRoot '..') | Select-Object -ExpandProperty Path
Set-Location $projectRoot

# 1. Ensure Memurai / Redis service running
try {
    $svc = Get-Service -Name 'Memurai' -ErrorAction SilentlyContinue
    if ($svc) {
        if ($svc.Status -ne 'Running') {
            Write-Host "[restart] Starting Memurai service..." -ForegroundColor Yellow
            Start-Service Memurai
            $svc.WaitForStatus('Running','00:00:10')
        }
        Write-Host "[restart] Memurai status: $($svc.Status)" -ForegroundColor Green
    } else {
        Write-Warning '[restart] Memurai service not found. Assuming external/WSL Redis.'
    }
} catch { Write-Warning "[restart] Redis/Memurai start failed: $_" }

# 2. Terminate existing python processes for app/celery
Write-Host "[restart] Killing existing Parking App python processes..." -ForegroundColor Cyan
$targets = @()
if ($KillAllPython) {
    $targets = Get-Process -Name python -ErrorAction SilentlyContinue
} else {
    $procList = Get-CimInstance Win32_Process -Filter 'Name="python.exe"'
    foreach ($p in $procList) {
        if ($p.CommandLine -match 'backend\\app.py' -or $p.CommandLine -match 'celery -A backend.tasks.celery') {
            $targets += (Get-Process -Id $p.ProcessId -ErrorAction SilentlyContinue)
        }
    }
}
if ($targets.Count -gt 0) {
    foreach ($t in $targets) {
        try {
            Write-Host "[restart] Stopping PID $($t.Id) : $($t.ProcessName)" -ForegroundColor Yellow
            $t.Kill()
        } catch { Write-Warning "[restart] Failed to kill $($t.Id): $_" }
    }
} else {
    Write-Host '[restart] No existing target processes found.' -ForegroundColor DarkGray
}

# 3. Launch new terminals for API, worker, beat
function Start-StackProcess($title, $command) {
    Write-Host "[restart] Launching $title..." -ForegroundColor Cyan
    $psCmd = "cd `"$projectRoot`"; .\\.venv\\Scripts\\Activate.ps1; $command"
    Start-Process powershell -ArgumentList '-NoExit','-Command', $psCmd -WindowStyle Normal
}

Start-StackProcess 'Flask API' '.\\.venv\\Scripts\\python.exe backend\\app.py'
Start-StackProcess 'Celery Worker' '.\\.venv\\Scripts\\python.exe -m celery -A backend.tasks.celery worker -l info'
Start-StackProcess 'Celery Beat' '.\\.venv\\Scripts\\python.exe -m celery -A backend.tasks.celery beat -l info'

# 4. Health check (delay to allow API boot)
if (-not $NoHealth) {
    Write-Host '[restart] Waiting for API to boot...' -ForegroundColor Cyan
    Start-Sleep -Seconds 4
    try {
        $health = Invoke-RestMethod -Uri 'http://localhost:5000/api/health' -TimeoutSec 5
        Write-Host "[restart] Health: $($health.status)" -ForegroundColor Green
    } catch { Write-Warning "[restart] Health check failed: $_" }
}

Write-Host '[restart] Done.' -ForegroundColor Green
