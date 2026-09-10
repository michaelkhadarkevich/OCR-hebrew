param([string]$OutputRoot = ".\output")

$ErrorActionPreference = "Stop"

$python = ".\.venv\Scripts\python.exe"
$trainer = ".\train_words_3000.py"
$targetSteps = 20000
$common = @(
    "--steps", "$targetSteps",
    "--batch-size", "16",
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
    "--test-dir", ".\data\valid",
    "--final-test-dir", ".\data\test",
    "--strip-whitespace"
)

function Run-Experiment {
    param(
        [string]$Name,
        [string]$OutputDirectory,
        [string[]]$TrainingDirectories
    )

    Write-Host "Starting or resuming $Name"
    while ($true) {
        $summaryPath = Join-Path $OutputDirectory "run\summary.json"
        if (Test-Path -LiteralPath $summaryPath) {
            $existingSummary = Get-Content -LiteralPath $summaryPath -Raw | ConvertFrom-Json
            if ($existingSummary.training_recipe_version -ne "global-cosine-bound-ema-v1") {
                throw "Legacy EMA/schedule results exist in $OutputDirectory. Use -OutputRoot with a fresh directory; these results cannot be repaired by resuming."
            }
            if (
                [int]$existingSummary.target_steps -eq $targetSteps -and
                [int]$existingSummary.completed_steps -ge $targetSteps
            ) {
                Write-Host "$Name is already complete at $($existingSummary.completed_steps) steps"
                break
            }
        }

        $arguments = @($common) + @("--out-dir", $OutputDirectory, "--train-dirs") + $TrainingDirectories
        $resumeCheckpoint = Join-Path $OutputDirectory "run\last_model.pth"
        if (Test-Path -LiteralPath $resumeCheckpoint) {
            $arguments += @("--resume-checkpoint", $resumeCheckpoint)
        }

        & $python $trainer @arguments
        if ($LASTEXITCODE -ne 0) {
            throw "$Name failed with exit code $LASTEXITCODE"
        }

        $summary = Get-Content -LiteralPath $summaryPath -Raw | ConvertFrom-Json
        Write-Host "$Name stage complete: $($summary.completed_steps)/$($summary.target_steps) steps"
        if ([int]$summary.completed_steps -ge [int]$summary.target_steps) {
            break
        }
    }
}

$wordDirectories = @(
    ".\data\train\arielOnlyWord",
    ".\data\train\RockOnlyWord",
    ".\data\train\agadaOnlyWord",
    ".\data\train\HarmonitManualTrainOnlyWord"
)
$lineDirectories = @(
    ".\data\train\arielManual",
    ".\data\train\RockManual",
    ".\data\train\AgadaManual",
    ".\data\train\HarmonitManualTrain"
)

$wordsOutput = Join-Path $OutputRoot "paper_recipe_words_only_20000"
$linesOutput = Join-Path $OutputRoot "paper_recipe_lines_only_20000"
$mixedOutput = Join-Path $OutputRoot "paper_recipe_lines_plus_words_20000"
Run-Experiment "words only" $wordsOutput $wordDirectories
Run-Experiment "lines only" $linesOutput $lineDirectories
Run-Experiment "lines and words" $mixedOutput ($lineDirectories + $wordDirectories)

& $python ".\plot_training_curves.py" `
    --run "Words only=$wordsOutput\run" `
    --run "Lines only=$linesOutput\run" `
    --run "Lines + words=$mixedOutput\run" `
    --output-dir (Join-Path $OutputRoot "paper_recipe_three_experiments\graphs")
if ($LASTEXITCODE -ne 0) {
    throw "Graph generation failed with exit code $LASTEXITCODE"
}

Write-Host "All paper-recipe experiments and graphs completed."
