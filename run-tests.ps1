# Run all Playwright tests for the Aplikácia na podporu rozhodavnia pri tréningu predikčných modelov app
# Usage:
#   .\run-tests.ps1              - Run all tests (headed - vidíš browser)
#   .\run-tests.ps1 --headless   - Run headless (bez okna)
#   .\run-tests.ps1 -k heatmap   - Run only heatmap tests
#   .\run-tests.ps1 -k "init or pipeline" - Run init + pipeline tests

param(
    [switch]$Headless,
    [string]$K,
    [string[]]$ExtraArgs
)

Set-Location $PSScriptRoot

$extraArgsString = ""

# Parse arguments
if ($Headless) {
    $extraArgsString += " --browser chromium"
    $env:HEADED = ""
}

if ($K) {
    $extraArgsString += " -k `"$K`""
}

foreach ($arg in $ExtraArgs) {
    $extraArgsString += " $arg"
}

# Check if server is running
try {
    $response = Invoke-WebRequest -Uri "http://localhost:3333" -TimeoutSec 5 -ErrorAction Stop
    $serverRunning = $true
} catch {
    $serverRunning = $false
}

if (-not $serverRunning) {
    Write-Host " Server nie je spustený na porte 3333"
    Write-Host " Spúšťam server na pozadí..."
    $serverProcess = Start-Process -FilePath "bash" -ArgumentList "serve.sh" -NoNewWindow -PassThru
    $SERVER_PID = $serverProcess.Id
    Start-Sleep -Seconds 2
    Write-Host "   Server PID: $SERVER_PID"
    $global:ServerPid = $SERVER_PID
    # Register cleanup on exit
    $null = Register-ObjectEvent -InputObject ([System.AppDomain]::CurrentDomain) -EventName "ProcessExit" -Action {
        if ($global:ServerPid) {
            Stop-Process -Id $global:ServerPid -ErrorAction SilentlyContinue
            Write-Host "   Server zastavený"
        }
    }
}

Write-Host " Spúšťam Playwright testy..."
Write-Host ""

if ($env:HEADED -or -not $Headless) {
    & python3 -m pytest --headed --browser chromium $extraArgsString.Split()
} else {
    & python3 -m pytest --browser chromium --timeout 120 $extraArgsString.Split()
}