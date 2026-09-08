$ErrorActionPreference = "Stop"

$python = ".\.venv\Scripts\python.exe"
$trainer = ".\train_words_3000.py"
$targetSteps = 20000
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

Run-Experiment "words only" ".\output\paper_recipe_words_only_20000" $wordDirectories
Run-Experiment "lines only" ".\output\paper_recipe_lines_only_20000" $lineDirectories
Run-Experiment "lines and words" ".\output\paper_recipe_lines_plus_words_20000" ($lineDirectories + $wordDirectories)

& $python ".\plot_training_curves.py" `
    --run "Words only=output\paper_recipe_words_only_20000\run" `
    --run "Lines only=output\paper_recipe_lines_only_20000\run" `
    --run "Lines + words=output\paper_recipe_lines_plus_words_20000\run" `
    --output-dir ".\output\paper_recipe_three_experiments\graphs"
if ($LASTEXITCODE -ne 0) {
    throw "Graph generation failed with exit code $LASTEXITCODE"
}

Write-Host "All paper-recipe experiments and graphs completed."
