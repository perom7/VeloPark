# Deprecated: Docker-based Redis start script
# This project no longer uses Docker for Redis. Please install and run a local Redis instead.
# Options:
#  - Memurai (Windows service)
#  - WSL2 + Ubuntu redis-server
# Then verify with the VS Code task: "Redis: Verify (Local)"

Write-Warning "This script is deprecated. Use a local Redis (Memurai/WSL2) and run 'Redis: Verify (Local)'."
exit 1