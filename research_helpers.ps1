# Dot-source from the UMSM project directory: . .\research_helpers.ps1
# Loading this file only defines helpers; it never starts an experiment.

function Invoke-Stage {
    # Input: job, phase, overrides. Output: run.py output; throws on failure.
    param([hashtable]$Job, [string]$Phase, [string[]]$Extra=@())
    if (-not $Job -or -not $Job.Args) { throw 'Create a research job first.' }
    if (-not (Test-Path -LiteralPath 'run.py')) { throw 'Run from the UMSM project directory.' }
    $jobArgs = $Job.Args
    python run.py @jobArgs --phase $Phase @Extra
    if ($LASTEXITCODE -ne 0) { throw "Research phase failed: $Phase" }
}

function New-ResearchJob {
    # Input: experiment labels and overrides. Output: a job hashtable; no execution.
    param([string]$Name, [string]$Fold, [string]$Matcher, [string[]]$Extra=@(),
          [string]$Config='configs/staged.yaml')
    if ($Fold -in @('real_external', 'websites')) {
        foreach ($role in @('fit', 'validation', 'test')) {
            foreach ($field in @('name', 'sets')) {
                $key = "research.roles.$role.$field="
                if (-not ($Extra | Where-Object { $_.StartsWith($key) -and $_.Length -gt $key.Length })) {
                    throw "Missing $key override. Define the dataset role array again in this terminal."
                }
            }
        }
    }
    $artifact = "output/calibration/research/${Name}_${Matcher}_${Fold}.json"
    $jobArgs = @('-c',$Config,
        '-s',"research.name=$Name", '-s',"research.fold=$Fold",
        '-s',"research.calibration=$artifact")
    if ($Matcher -eq 'quantile') {
        $jobArgs += @('--matcher','sbert_quantile','-s','research.matcher=quantile',
            '-s','modules.matcher.params.normalization=fixed',
            '-s','modules.matcher.params.name_weight=0',
            '-s','modules.matcher.params.value_weight=1')
    } else { $jobArgs += @('--matcher',$Matcher) }
    return @{ Args=($jobArgs+$Extra); Artifact=$artifact;
        Folder="output/results/research/$Name/$Fold/$Matcher"; Selection=$null }
}

function Initialize-ResearchJob {
    # Input: job. Output: profile/calibration/validation; sets the frozen selection path.
    param([hashtable]$Job)
    if (-not $Job -or -not $Job.Args) { throw 'Create a research job first.' }
    $Job.Selection = $null
    Invoke-Stage $Job 'profile'
    if (-not (Test-Path -LiteralPath $Job.Artifact)) { Invoke-Stage $Job 'fit' }
    Invoke-Stage $Job 'qcheck'
    Invoke-Stage $Job 'validate-prior'
    $choice = Get-ChildItem ($Job.Folder+'/validate-prior/*.html') -ErrorAction Stop |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($null -eq $choice) { throw 'No prior validation report was generated.' }
    $Job.Selection = $choice.FullName
    Write-Host 'Frozen validation report:' $Job.Selection
}

function Invoke-SelectedStage {
    # Input: initialized job, phase, overrides. Output: evaluation with frozen selection.
    param([hashtable]$Job, [string]$Phase, [string[]]$Extra=@())
    if (-not $Job -or -not $Job.Selection) { throw 'Initialize the research job first.' }
    Invoke-Stage $Job $Phase (@('--selection',$Job.Selection)+$Extra)
}
