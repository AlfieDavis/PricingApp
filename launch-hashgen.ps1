# PowerShell launcher for generate_hash.py.
# It locates the virtual environment's Python interpreter if available and runs
# the generator script. To run: right-click and choose Run with PowerShell or
# execute from a PowerShell prompt.

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvPy = Join-Path $here "venv\Scripts\python.exe"
if (Test-Path $venvPy) {
    $py = $venvPy
} else {
    $py = "python"
}
& $py (Join-Path $here "generate_hash.py")