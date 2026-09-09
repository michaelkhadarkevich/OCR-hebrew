@echo off
powershell -NoProfile -ExecutionPolicy Bypass -Command "$p=Get-NetTCPConnection -LocalPort 8770 -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1; if($p){$proc=Get-CimInstance Win32_Process -Filter ('ProcessId=' + $p.OwningProcess) -ErrorAction SilentlyContinue; if($proc -and $proc.CommandLine -match 'server.py'){Stop-Process -Id $p.OwningProcess -Force; Write-Host 'Manual Segmenter v4 stopped.'} else {Write-Host 'Port 8770 is used by another program; nothing was stopped.'}} else {Write-Host 'Manual Segmenter v4 is not running.'}"
pause
