# One-command Windows launcher for the Headroom WSL demo.
# Run from Windows PowerShell; elevate when usbipd binding or LAN forwarding needs it.
[CmdletBinding()]
param(
    [string]$Distro = 'Ubuntu',
    [string]$WslProjectPath = '',
    [int]$Port = 8766
)

$ErrorActionPreference = 'Stop'
$launcher = Join-Path $PSScriptRoot 'scripts\start-headroom.ps1'

if (-not (Test-Path -LiteralPath $launcher -PathType Leaf)) {
    throw "Headroom launcher not found: $launcher"
}

& $launcher @PSBoundParameters
