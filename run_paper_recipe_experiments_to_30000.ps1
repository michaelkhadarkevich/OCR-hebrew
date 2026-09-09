param([string]$OutputRoot = ".\output")

$ErrorActionPreference = "Stop"

$python = ".\.venv\Scripts\python.exe"
$trainer = ".\train_words_3000.py"
$targetSteps = 30000
$common = @(
    "--steps", "$targetSteps",
    "--batch-size", "8",
    "--width", "1024",
    "--height", "64",
    "--eval-every", "100",
    "--checkpoint-every", "0",
    "--chunk-epochs", "1",
    "--optimizer", "sam",
    "--sam-rho", "0.05",
    "--lr", "0.001",
    "--min-lr", "0.0000001",
    "--warm-up-steps", "1000",
    "--weight-decay", "0.5",
    "--ema-decay", "0.9999",
    "--mask-ratio", "0.4",
    "--max-span-length", "8",
    "--paper-augmentation",
    "--paper-augmentation-probability", "0.5",
    "--test-dir", ".\data\HarmonitManualTest",
    "--final-test-dir", ".\data\HarmonitManualFinalTest",
    "--strip-whitespace"
)

function Initialize-Continuation {
    param(
        [string]$SourceDirectory,
        [string]$OutputDirectory
    )

    if (Test-Path -LiteralPath $OutputDirectory) {
        return
    }
    $sourceSummaryPath = Join-Path $SourceDirectory "run\summary.json"
    $sourceSummary = Get-Content -LiteralPath $sourceSummaryPath -Raw | ConvertFrom-Json
    if ($sourceSummary.training_recipe_version -ne "global-cosine-bound-ema-v1") {
        throw "Source run uses the legacy EMA/schedule implementation. First train corrected runs under a fresh -OutputRoot."
    }
    if ($sourceSummary.status -ne "complete" -or [int]$sourceSummary.completed_steps -ne 20000) {
        throw "The source run is not a complete 20,000-step run: $SourceDirectory"
    }
    Write-Host "Copying the completed 20,000-step run to $OutputDirectory"
    Copy-Item -LiteralPath $SourceDirectory -Destination $OutputDirectory -Recurse
}

function Continue-Experiment {
    param(
        [string]$Name,
        [string]$SourceDirectory,
        [string]$OutputDirectory,
        [string[]]$TrainingDirectories
    )

    Initialize-Continuation $SourceDirectory $OutputDirectory
    Write-Host "Starting or resuming $Name toward $targetSteps steps"
    while ($true) {
        $summaryPath = Join-Path $OutputDirectory "run\summary.json"
        if (Test-Path -LiteralPath $summaryPath) {
            $existingSummary = Get-Content -LiteralPath $summaryPath -Raw | ConvertFrom-Json
            if ($existingSummary.training_recipe_version -ne "global-cosine-bound-ema-v1") {
                throw "Legacy EMA/schedule results exist in $OutputDirectory. Use corrected runs under a fresh -OutputRoot."
            }
            if (
                [int]$existingSummary.target_steps -eq $targetSteps -and
                [int]$existingSummary.completed_steps -ge $targetSteps
            ) {
                Write-Host "$Name is already complete at $($existingSummary.completed_steps) steps"
                break
            }
        }

        $resumeCheckpoint = Join-Path $OutputDirectory "run\last_model.pth"
        if (-not (Test-Path -LiteralPath $resumeCheckpoint)) {
            throw "Missing continuation checkpoint: $resumeCheckpoint"
        }
        $arguments = @($common) + @(
            "--out-dir", $OutputDirectory,
            "--resume-checkpoint", $resumeCheckpoint,
            "--train-dirs"
        ) + $TrainingDirectories

        & $python $trainer @arguments
        if ($LASTEXITCODE -ne 0) {
            throw "$Name failed with exit code $LASTEXITCODE"
        }

        $summary = Get-Content -LiteralPath $summaryPath -Raw | ConvertFrom-Json
        Write-Host "$Name stage complete: $($summary.completed_steps)/$($summary.target_steps) steps"
    }
}

$wordDirectories = @(
    ".\data\arielOnlyWord",
    ".\data\RockOnlyWord",
    ".\data\agadaOnlyWord",
    ".\data\HarmonitManualTrainOnlyWord"
)
$lineDirectories = @(
    ".\data\arielManual",
    ".\data\RockManual",
    ".\data\AgadaManual",
    ".\data\HarmonitManualTrain"
)

Continue-Experiment `
    "words only" `
    (Join-Path $OutputRoot "paper_recipe_words_only_20000") `
    (Join-Path $OutputRoot "paper_recipe_words_only_30000") `
    $wordDirectories
Continue-Experiment `
    "lines only" `
    (Join-Path $OutputRoot "paper_recipe_lines_only_20000") `
    (Join-Path $OutputRoot "paper_recipe_lines_only_30000") `
    $lineDirectories
Continue-Experiment `
    "lines and words" `
    (Join-Path $OutputRoot "paper_recipe_lines_plus_words_20000") `
    (Join-Path $OutputRoot "paper_recipe_lines_plus_words_30000") `
    ($lineDirectories + $wordDirectories)

& $python ".\plot_training_curves.py" `
    --run "Words only=$OutputRoot\paper_recipe_words_only_30000\run" `
    --run "Lines only=$OutputRoot\paper_recipe_lines_only_30000\run" `
    --run "Lines + words=$OutputRoot\paper_recipe_lines_plus_words_30000\run" `
    --output-dir (Join-Path $OutputRoot "paper_recipe_three_experiments_30000\graphs")
if ($LASTEXITCODE -ne 0) {
    throw "Graph generation failed with exit code $LASTEXITCODE"
}

Write-Host "All 30,000-step continuations and graphs completed."
