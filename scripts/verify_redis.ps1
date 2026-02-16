<#
 Verifies that a local Redis server is reachable using the app's REDIS_URL
 Defaults to redis://localhost:6379/0
#>
$ErrorActionPreference = 'Stop'

$python = ".\.venv\Scripts\python.exe"
if (!(Test-Path $python)) { Write-Error ".venv Python not found. Create venv first."; exit 1 }

$code = @'
import os, sys
try:
    import redis
except Exception as e:
    print("redis-py not installed; run pip install -r backend/requirements.txt", file=sys.stderr)
    sys.exit(1)
url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
r = redis.Redis.from_url(url)
try:
    pong = r.ping()
    print(f"Redis OK at {url}: {pong}")
    sys.exit(0)
except Exception as e:
    print(f"Redis NOT reachable at {url}: {e}", file=sys.stderr)
    sys.exit(2)
'@
# Write code to a temp file to avoid quoting issues on Windows
$tmpPy = Join-Path $env:TEMP "verify_redis_$([System.Guid]::NewGuid().ToString('N')).py"
Set-Content -LiteralPath $tmpPy -Value $code -Encoding UTF8 -NoNewline
try {
    & $python $tmpPy
    exit $LASTEXITCODE
}
finally {
    Remove-Item -LiteralPath $tmpPy -ErrorAction SilentlyContinue
}
