<#
Windows equivalent of run_batch.sh -- launches Trials simultaneous copies of
main.py at a single --loc, first with fast_fading_enabled = False (NFF),
then flips it to True and launches the FF batch. Mirrors run_batch.sh's
NFF-then-FF ordering (main.py only reads env_params.py once near startup,
so the flag must be set correctly BEFORE each phase's processes launch --
not mid-phase).

Not a SLURM job -- this runs locally, each trial in its own visible
PowerShell window. For cluster use, use run_batch.sh/array_task.sh instead.

Usage:
    powershell -ExecutionPolicy Bypass -File run_batch.ps1
    powershell -ExecutionPolicy Bypass -File run_batch.ps1 -Loc 9 -Trials 5
#>

param(
    [string]$Env = "SIG",
    [string]$Algo = "ippo",
    [int]$NAgent = 16,
    [double]$Loc = 9.0,
    [int]$Trials = 5
)

Set-Location -Path $PSScriptRoot

$paramsFile = Join-Path $PSScriptRoot "Configuration\env_params.py"

function Set-FastFading($value) {
    $content = Get-Content -Path $paramsFile -Raw
    if ($content -notmatch '(?m)^\s*self\.fast_fading_enabled\s*=\s*\S+') {
        Write-Error "Could not find 'self.fast_fading_enabled' in $paramsFile -- nothing was changed."
        exit 1
    }
    $updated = $content -replace '(?m)^(\s*self\.fast_fading_enabled\s*=\s*)\S+', "`${1}$value"
    Set-Content -Path $paramsFile -Value $updated -NoNewline
    Write-Host "Set fast_fading_enabled = $value in $paramsFile"
}

function Run-Batch($tag, $locStr) {
    Write-Host "`n=== Starting $tag batch: $Trials trial(s) at loc=$locStr ==="
    $procs = @()
    for ($t = 0; $t -lt $Trials; $t++) {
        $proc = Start-Process powershell.exe -PassThru -ArgumentList @(
            "-NoExit",
            "-Command",
            "cd '$PSScriptRoot'; `$host.UI.RawUI.WindowTitle = '$tag loc=$locStr trial=$t'; python -u main.py --env $Env --loc $locStr --algo $Algo --n_agent $NAgent"
        ) -WindowStyle Normal
        Write-Host "Launched $tag trial $t (pid $($proc.Id))"
        $procs += $proc
        # 8s between launches -- the CSV filename timestamp is taken when each
        # process's own init_csv_logging() call runs (well after this launch,
        # after python/torch startup), not at launch time. A short stagger here
        # can still let two processes' actual startup land in the same second
        # (startup jitter varies run to run) and collide/interleave into one
        # CSV. 8s comfortably absorbs that jitter.
        Start-Sleep -Seconds 8
    }

    Write-Host "Waiting for all $Trials $tag processes to start (not finish)..."
    foreach ($proc in $procs) {
        $appeared = $false
        for ($i = 0; $i -lt 15; $i++) {
            if (Get-CimInstance Win32_Process -Filter "ParentProcessId = $($proc.Id) AND Name = 'python.exe'" -ErrorAction SilentlyContinue) {
                $appeared = $true
                break
            }
            Start-Sleep -Seconds 2
        }
        if (-not $appeared) {
            Write-Warning "python.exe never appeared under window PID $($proc.Id) -- check that window for an error."
        }
    }
    # Small buffer so each process is past the env_params.py import line
    # before the next phase overwrites the file.
    Start-Sleep -Seconds 5
    Write-Host "$tag batch started."
}

$locStr = $Loc.ToString("0.0")

Set-FastFading "False"
Run-Batch "NFF" $locStr

Set-FastFading "True"
Run-Batch "FF" $locStr

Write-Host "`nBoth batches launched: $Trials NFF + $Trials FF at loc=$locStr."
Write-Host "Check Results\IPPO\*.csv afterward to confirm $Trials distinct output files per tag, not fewer (one being overwritten)."
