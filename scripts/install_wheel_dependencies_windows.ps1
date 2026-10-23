# Builds oneTBB for the Windows wheels; neither the runner nor PyPI has a TBB that Embree can use.
# cibuildwheel runs it once per build, see [tool.cibuildwheel.windows] in pyproject.toml.
$ErrorActionPreference = "Stop"

$TbbVersion = "v2023.1.0"
$Prefix = if ($env:TBB_PREFIX) { $env:TBB_PREFIX } else { "C:\onetbb-install" }
$Src = Join-Path $env:TEMP "onetbb"

if (Test-Path $Src) { Remove-Item -Recurse -Force $Src }
git clone --depth 1 --branch $TbbVersion https://github.com/uxlfoundation/oneTBB.git $Src
if ($LASTEXITCODE) { exit $LASTEXITCODE }
cmake -S $Src -B "$Src\build" -DCMAKE_INSTALL_PREFIX="$Prefix" -DTBB_TEST=OFF -DTBB_EXAMPLES=OFF
if ($LASTEXITCODE) { exit $LASTEXITCODE }
cmake --build "$Src\build" --config Release --parallel
if ($LASTEXITCODE) { exit $LASTEXITCODE }
cmake --install "$Src\build" --config Release
if ($LASTEXITCODE) { exit $LASTEXITCODE }
Remove-Item -Recurse -Force $Src
