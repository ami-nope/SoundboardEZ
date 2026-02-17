param(
  [string]$PythonExe = "python",
  [string]$NSISExe = "makensis"
)

$ErrorActionPreference = "Stop"

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptRoot
Set-Location $repoRoot

$versionFile = Join-Path $repoRoot "version.py"
if (-not (Test-Path -LiteralPath $versionFile)) {
  throw "version.py not found."
}

$versionText = Get-Content -Path $versionFile -Raw
$match = [regex]::Match($versionText, 'APP_VERSION\s*=\s*["'']([^"'']+)["'']')
if (-not $match.Success) {
  throw "APP_VERSION was not found in version.py."
}
$version = $match.Groups[1].Value

Write-Host "Building SoundboardEZ version $version"

& $PythonExe -m PyInstaller --clean --noconfirm main.spec
if ($LASTEXITCODE -ne 0) {
  throw "PyInstaller build failed."
}

& $NSISExe "/DAPPVERSION=$version" "installer.nsi"
if ($LASTEXITCODE -ne 0) {
  throw "NSIS build failed."
}

$installerPath = Join-Path $repoRoot "SoundboardEZ-Setup.exe"
if (-not (Test-Path -LiteralPath $installerPath)) {
  throw "Installer output was not found at $installerPath"
}

$hash = Get-FileHash -Path $installerPath -Algorithm SHA256
$checksumPath = Join-Path $repoRoot "SoundboardEZ-Setup.exe.sha256"
"$($hash.Hash.ToLowerInvariant())  SoundboardEZ-Setup.exe" | Set-Content -Path $checksumPath -Encoding ascii

Write-Host "Build complete."
Write-Host "Installer: $installerPath"
Write-Host "Checksum:  $checksumPath"
