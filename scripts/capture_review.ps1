[CmdletBinding()]
param(
  [string]$Url = 'http://localhost:8766/',
  [string]$OutputPath = (Join-Path $env:USERPROFILE 'Downloads\headroom-review.png'),
  [ValidateRange(320, 1600)][int]$Size = 800,
  [ValidateRange(5, 120)][int]$TimeoutSeconds = 60,
  [ValidateSet('Chrome', 'Edge')][string]$Browser = 'Edge'
)

$ErrorActionPreference = 'Stop'
$browserPath = if ($Browser -eq 'Chrome') {
  'C:\Program Files\Google\Chrome\Application\chrome.exe'
} else {
  'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'
}
$processName = if ($Browser -eq 'Chrome') { 'chrome.exe' } else { 'msedge.exe' }
if (-not (Test-Path $browserPath)) { throw "$Browser not found at $browserPath" }
try {
  Invoke-WebRequest -UseBasicParsing -Uri 'http://localhost:8766/' -TimeoutSec 3 | Out-Null
} catch {
  throw 'Headroom is not responding at http://localhost:8766/. Start run.sh first.'
}

$userData = Join-Path $env:TEMP "headroom-review-$PID"
$process = $null
$socket = $null
$script:cdpId = 0

function Send-Cdp([string]$Method, [hashtable]$Params = @{}) {
  $script:cdpId++
  $id = $script:cdpId
  $json = @{ id = $id; method = $Method; params = $Params } | ConvertTo-Json -Compress -Depth 20
  $bytes = [System.Text.Encoding]::UTF8.GetBytes($json)
  $segment = [System.ArraySegment[byte]]::new($bytes)
  $commandDeadline = (Get-Date).AddSeconds(45)
  $sendTimeout = [System.Threading.CancellationTokenSource]::new([TimeSpan]::FromSeconds(10))
  try {
    $null = $socket.SendAsync($segment, [System.Net.WebSockets.WebSocketMessageType]::Text, $true, $sendTimeout.Token).GetAwaiter().GetResult()
  } finally {
    $sendTimeout.Dispose()
  }

  while ($true) {
    if ((Get-Date) -ge $commandDeadline) { throw "Timed out waiting for $Browser CDP response to $Method." }
    $memory = [System.IO.MemoryStream]::new()
    do {
      $buffer = New-Object byte[] 65536
      $receiveSegment = [System.ArraySegment[byte]]::new($buffer)
      $remainingSeconds = [Math]::Max(1, [Math]::Min(25, ($commandDeadline - (Get-Date)).TotalSeconds))
      $receiveTimeout = [System.Threading.CancellationTokenSource]::new([TimeSpan]::FromSeconds($remainingSeconds))
      try {
        $result = $socket.ReceiveAsync($receiveSegment, $receiveTimeout.Token).GetAwaiter().GetResult()
      } catch [System.OperationCanceledException] {
        throw "Timed out waiting for $Browser CDP response to $Method."
      } finally {
        $receiveTimeout.Dispose()
      }
      if ($result.MessageType -eq [System.Net.WebSockets.WebSocketMessageType]::Close) {
        throw "$Browser closed the DevTools connection."
      }
      $memory.Write($buffer, 0, $result.Count)
    } while (-not $result.EndOfMessage)

    $message = [System.Text.Encoding]::UTF8.GetString($memory.ToArray()) | ConvertFrom-Json
    $memory.Dispose()
    if ($message.id -eq $id) {
      if ($message.error) { throw "CDP $Method failed: $($message.error.message)" }
      return $message.result
    }
  }
}

try {
  $browserArgs = @(
    '--headless=new', '--no-first-run', '--disable-background-networking',
    '--enable-webgl', '--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader',
    '--remote-debugging-port=0', "--user-data-dir=$userData", "--window-size=$Size,$Size", $Url
  )
  $process = Start-Process -FilePath $browserPath -ArgumentList $browserArgs -PassThru
  $activePortFile = Join-Path $userData 'DevToolsActivePort'
  $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
  while (-not (Test-Path $activePortFile) -and (Get-Date) -lt $deadline) { Start-Sleep -Milliseconds 100 }
  if (-not (Test-Path $activePortFile)) { throw "$Browser did not publish its DevTools port in time." }
  $port = [int](Get-Content $activePortFile -TotalCount 1)

  $target = $null
  $requestedUrl = ([Uri]::new($Url)).AbsoluteUri
  while (-not $target -and (Get-Date) -lt $deadline) {
    try {
      $targets = Invoke-RestMethod -UseBasicParsing -Uri "http://127.0.0.1:$port/json/list" -TimeoutSec 3
      $target = $targets | Where-Object { $_.type -eq 'page' -and $_.url -eq $requestedUrl } | Select-Object -First 1
    } catch { }
    if (-not $target) { Start-Sleep -Milliseconds 150 }
  }
  if (-not $target) { throw "$Browser opened, but could not find the Headroom page." }

  $socket = [System.Net.WebSockets.ClientWebSocket]::new()
  $webSocketUrl = [string]$target.webSocketDebuggerUrl
  $connectTimeout = [System.Threading.CancellationTokenSource]::new([TimeSpan]::FromSeconds(10))
  try {
    $null = $socket.ConnectAsync([Uri]::new($webSocketUrl), $connectTimeout.Token).GetAwaiter().GetResult()
  } finally {
    $connectTimeout.Dispose()
  }
  Send-Cdp 'Page.enable' | Out-Null
  Send-Cdp 'Runtime.enable' | Out-Null
  Send-Cdp 'Emulation.setDeviceMetricsOverride' @{ width = $Size; height = $Size; deviceScaleFactor = 1; mobile = $false } | Out-Null

  $lastState = ''
  do {
    $stateResponse = Send-Cdp 'Runtime.evaluate' @{
      expression = "(() => { const d = document.documentElement?.dataset; return JSON.stringify({avatar:d?.headroomAvatarReady,hair:d?.headroomHairReady}); })()"
      returnByValue = $true
    }
    if ($stateResponse.result.exceptionDetails) {
      throw "Readiness evaluation failed: $($stateResponse.result.exceptionDetails | ConvertTo-Json -Compress -Depth 5)"
    }
    $lastState = $stateResponse.result.value
    if (-not $lastState) { throw "Readiness evaluation returned no value: $($stateResponse | ConvertTo-Json -Compress -Depth 5)" }
    $state = $lastState | ConvertFrom-Json
    if ($state.avatar -eq 'true' -and ($state.hair -eq 'true' -or $state.hair -eq 'not-applicable')) { break }
    if ($state.avatar -eq 'error' -or $state.hair -eq 'error') { throw "Headroom load failed: $lastState" }
    if ((Get-Date) -ge $deadline) { throw "Timed out waiting for avatar and hair readiness: $lastState" }
    Start-Sleep -Milliseconds 200
  } while ($true)

  Send-Cdp 'Runtime.evaluate' @{
    expression = 'new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))'
    awaitPromise = $true
    returnByValue = $true
  } | Out-Null
  $capture = Send-Cdp 'Page.captureScreenshot' @{ format = 'png'; captureBeyondViewport = $false }
  $fullOutputPath = [System.IO.Path]::GetFullPath($OutputPath)
  [System.IO.Directory]::CreateDirectory([System.IO.Path]::GetDirectoryName($fullOutputPath)) | Out-Null
  [System.IO.File]::WriteAllBytes($fullOutputPath, [Convert]::FromBase64String($capture.data))
  Write-Output "Captured loaded Headroom scene ($lastState) to $fullOutputPath"
}
finally {
  if ($socket) { $socket.Dispose() }
  try {
    Get-CimInstance Win32_Process -Filter "Name='$processName'" |
      Where-Object { $_.CommandLine -like "*$userData*" } |
      ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
  } catch { }
  if (Test-Path $userData) { Remove-Item -LiteralPath $userData -Recurse -Force -ErrorAction SilentlyContinue }
}
