param(
    [string]$Distro = 'Ubuntu',
    [string]$WslProjectPath = '',
    [int]$Port = 8766
)

$ErrorActionPreference = 'Stop'
if ([string]::IsNullOrWhiteSpace($WslProjectPath)) {
    $repoPath = Split-Path -Parent $PSScriptRoot
    if ($repoPath -match '^\\\\wsl(?:\.localhost|\$)?\\[^\\]+(?<WslPath>\\.*)$') {
        $WslProjectPath = $Matches.WslPath.Replace('\', '/')
    }
    else {
        throw 'Could not infer the WSL project path from this checkout. Pass -WslProjectPath /path/in/wsl explicitly.'
    }
}

$usbipdCommand = Get-Command usbipd.exe -ErrorAction SilentlyContinue
if (-not $usbipdCommand) { $usbipdCommand = Get-Command usbipd -ErrorAction SilentlyContinue }
if (-not $usbipdCommand -and (Test-Path 'C:\Program Files\usbipd-win\usbipd.exe')) {
    $usbipdPath = 'C:\Program Files\usbipd-win\usbipd.exe'
}
elseif ($usbipdCommand) {
    $usbipdPath = $usbipdCommand.Source
}
else {
    throw 'usbipd-win was not found. Install/enable usbipd-win, then rerun this helper.'
}

$usbipdRows = @(& $usbipdPath list)
if ($LASTEXITCODE -ne 0) { throw 'Could not query usbipd device status.' }
$devices = @()
foreach ($row in $usbipdRows) {
    $line = $row.ToString().Trim()
    if ($line -match '^(?<BusId>\d+-\d+)\s+(?<VidPid>[0-9a-fA-F]{4}:[0-9a-fA-F]{4})\s+(?<Name>.+?)\s+(?<State>Not shared|Shared|Attached)$') {
        $devices += [pscustomobject]@{
            BusId = $Matches.BusId
            Name = $Matches.Name.Trim()
            State = $Matches.State
            Line = $line
        }
    }
}

$cameraRows = @($devices | Where-Object { $_.Name -match '(camera|webcam|uvc|video|brio)' })
$selected = $cameraRows | Where-Object { $_.Name -match 'Logitech.*BRIO|BRIO.*Logitech' } | Select-Object -First 1
if (-not $selected) {
    $selected = $cameraRows |
        Where-Object { $_.Name -match '(integrated|built.?in|internal|laptop|wide vision|front camera)' } |
        Select-Object -First 1
}
if (-not $selected) { $selected = $cameraRows | Select-Object -First 1 }

if (-not $selected) {
    if (($usbipdRows -join "`n") -match '(?s)Persisted:.*Logitech BRIO') {
        Write-Warning 'Logitech BRIO is listed as persisted but is not currently connected to Windows. Check its USB cable/port.'
    }
    throw 'No connected Logitech BRIO or recognizable USB camera was found by usbipd. A built-in camera that usbipd does not list cannot be passed into WSL.'
}

Write-Host "Selected camera: $($selected.Name) ($($selected.BusId), $($selected.State))."
if ($selected.Name -notmatch 'Logitech.*BRIO|BRIO.*Logitech') {
    if (($usbipdRows -join "`n") -match '(?s)Persisted:.*Logitech BRIO') {
        Write-Warning 'The saved Logitech BRIO is not connected right now; using the available camera instead.'
    }
    else {
        Write-Host 'No connected Logitech BRIO was found; using the available camera fallback.'
    }
}

if ($selected.State -eq 'Not shared') {
    Write-Host "Sharing $($selected.BusId) with WSL..."
    & $usbipdPath bind --busid $selected.BusId
    if ($LASTEXITCODE -ne 0) {
        throw 'Sharing requires Administrator rights. Reopen PowerShell as Administrator and rerun this helper.'
    }
}
if ($selected.State -ne 'Attached') {
    Write-Host "Attaching $($selected.Name) to WSL..."
    & $usbipdPath attach --wsl --busid $selected.BusId
    if ($LASTEXITCODE -ne 0) { throw 'usbipd could not attach the selected camera to WSL.' }
}
else {
    Write-Host 'The selected camera is already attached to WSL.'
}

Start-Sleep -Milliseconds 1200
$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = New-Object Security.Principal.WindowsPrincipal($identity)
$isAdministrator = $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

Write-Host "`nWaiting for WSL camera, microphone, and laptop-audio bridges..."
$bashProjectPath = $WslProjectPath.Replace("'", "'\''")
$probeCommand = "cd '$bashProjectPath' && if [ -x .venv/bin/python ]; then .venv/bin/python scripts/check_capture_devices.py --strict; else python3 scripts/check_capture_devices.py --strict; fi"
$probeReady = $false
$probeOutput = @()
$probeAttempts = 8
for ($attempt = 1; $attempt -le $probeAttempts; $attempt++) {
    $probeOutput = @(& wsl.exe -d $Distro -- bash -lc $probeCommand 2>&1)
    $probeExitCode = $LASTEXITCODE
    if ($probeExitCode -eq 0) {
        $probeReady = $true
        break
    }
    if ($attempt -lt $probeAttempts) {
        Write-Host "WSL hardware bridge not ready yet ($attempt/$probeAttempts); retrying in 2 seconds..."
        Start-Sleep -Seconds 2
    }
}
$probeOutput | ForEach-Object { Write-Host $_ }
if (-not $probeReady) {
    throw 'WSL hardware preflight failed after retries. Headroom was not started. Check that the camera is attached with usbipd, the microphone source is available, and WSLg PulseAudio is connected.'
}

# The iPad loads Headroom from the laptop's trusted Wi-Fi address. WSL NAT
# requires a narrow Windows port proxy; never open this unauthenticated app on
# a Public network or beyond the local subnet.
$wifiProfile = Get-NetConnectionProfile -ErrorAction SilentlyContinue |
    Where-Object { $_.InterfaceAlias -match 'Wi-?Fi|Wireless' -and $_.IPv4Connectivity -ne 'Disconnected' } |
    Select-Object -First 1

if (-not $wifiProfile) {
    Write-Warning 'No active Wi-Fi profile found; skipped iPad port forwarding.'
}
elseif ($wifiProfile.NetworkCategory -ne 'Private') {
    Write-Warning "Wi-Fi is marked '$($wifiProfile.NetworkCategory)'. Because Headroom is unauthenticated, forwarding is limited to a trusted Private network; skipped."
}
elseif (-not $isAdministrator) {
    Write-Warning 'iPad forwarding needs Administrator rights. Rerun this helper as Administrator to configure the private-subnet port proxy.'
}
else {
    $wifiAddress = Get-NetIPAddress -InterfaceAlias $wifiProfile.InterfaceAlias `
        -AddressFamily IPv4 -AddressState Preferred -ErrorAction SilentlyContinue |
        Where-Object { $_.IPAddress -notmatch '^(127\.|169\.254\.)' } |
        Select-Object -First 1
    $wslAddressOutput = (& wsl.exe -d $Distro -- hostname -I 2>$null) -join ' '
    $wslAddress = ($wslAddressOutput -split '\s+' |
        Where-Object { $_ -match '^(\d{1,3}\.){3}\d{1,3}$' -and $_ -notmatch '^(127\.|169\.254\.)' } |
        Select-Object -First 1)

    if (-not $wifiAddress -or -not $wslAddress) {
        Write-Warning 'Could not determine the Wi-Fi or WSL IPv4 address; skipped iPad port forwarding.'
    }
    else {
        $listenAddress = $wifiAddress.IPAddress
        $ruleName = 'Headroom Pepper Ghost LAN 8766'
        $ipHelper = Get-Service -Name iphlpsvc -ErrorAction SilentlyContinue
        if ($ipHelper -and $ipHelper.Status -ne 'Running') { Start-Service -Name iphlpsvc }
        $null = & netsh.exe interface portproxy delete v4tov4 `
            "listenaddress=$listenAddress" "listenport=$Port" 2>$null
        $null = & netsh.exe interface portproxy add v4tov4 `
            "listenaddress=$listenAddress" "listenport=$Port" `
            "connectaddress=$wslAddress" "connectport=$Port"
        if ($LASTEXITCODE -ne 0) {
            throw 'Could not configure the WSL port proxy. Confirm Administrator rights and that the IP Helper service is running.'
        }

        Get-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue |
            Remove-NetFirewallRule -ErrorAction SilentlyContinue
        New-NetFirewallRule -DisplayName $ruleName -Direction Inbound -Action Allow `
            -Protocol TCP -LocalAddress $listenAddress -LocalPort $Port `
            -RemoteAddress LocalSubnet -Profile Private | Out-Null
        Write-Host "`niPad display URL: http://${listenAddress}:$Port/"
        Write-Host "Port proxy targets WSL $wslAddress; rerun this helper after WSL or Wi-Fi addresses change."
    }
}

Write-Host "`nStarting Headroom with ./run.sh in WSL. Keep this PowerShell window open; press Ctrl+C here to stop the backend."
$runCommand = "cd '$bashProjectPath' && exec ./run.sh"
& wsl.exe -d $Distro -- bash -lc $runCommand
$runExitCode = $LASTEXITCODE
if ($runExitCode -ne 0 -and $runExitCode -ne 130) {
    throw "Headroom backend exited with status $runExitCode. Check logs/headroom.log for details."
}
