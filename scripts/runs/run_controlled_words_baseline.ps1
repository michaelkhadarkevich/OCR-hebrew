$ErrorActionPreference = "Stop"

$python = ".\.venv\Scripts\python.exe"
$trainer = ".\train_words_3000.py"
$outputDirectory = ".\output\controlled_baseline_words_only_20000"
$targetSteps = 20000
$trainingDirectories = @(
    ".\data\train\arielOnlyWord",
    ".\data\train\RockOnlyWord",
    ".\data\train\agadaOnlyWord",
    ".\data\train\HarmonitManualTrainOnlyWord"
)
$common = @(
    "--steps", "$targetSteps",
    "--batch-size", "8",
    "--width", "1024",
    "--height", "64",
    "--eval-every", "100",
    "--checkpoint-every", "0",
    "--chunk-epochs", "1",
    "--optimizer", "adamw",
    "--lr", "0.0005",
    "--weight-decay", "0.0001",
    "--warm-up-steps", "0",
    "--ema-decay", "0",
    "--mask-ratio", "0",
    "--rotation-degrees", "2.0",
    "--rotation-probability", "0.5",
    "--test-dir", ".\data\valid\HarmonitManualTest",
    "--final-test-dir", ".\data\test\HarmonitManualFinalTest",
    "--strip-whitespace"
)

Write-Host "Starting or resuming the controlled words-only baseline"
while ($true) {
    $summaryPath = Join-Path $outputDirectory "run\summary.json"
    if (Test-Path -LiteralPath $summaryPath) {
        $summary = Get-Content -LiteralPath $summaryPath -Raw | ConvertFrom-Json
        if (
            [int]$summary.target_steps -eq $targetSteps -and
            [int]$summary.completed_steps -ge $targetSteps
        ) {
            Write-Host "Controlled baseline is already complete"
            break
        }
    }

    $arguments = @($common) + @(
        "--out-dir", $outputDirectory,
        "--train-dirs"
    ) + $trainingDirectories
    $resumeCheckpoint = Join-Path $outputDirectory "run\last_model.pth"
    if (Test-Path -LiteralPath $resumeCheckpoint) {
        $arguments += @("--resume-checkpoint", $resumeCheckpoint)
    }

    & $python $trainer @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Controlled baseline failed with exit code $LASTEXITCODE"
    }
    $summary = Get-Content -LiteralPath $summaryPath -Raw | ConvertFrom-Json
    Write-Host "Controlled baseline stage complete: $($summary.completed_steps)/$($summary.target_steps)"
}

& $python ".\plot_training_curves.py" `
    --run "Controlled baseline=output\controlled_baseline_words_only_20000\run" `
    --run "Paper recipe=output\paper_recipe_words_only_20000\run" `
    --output-dir ".\output\controlled_words_baseline_comparison\graphs"
if ($LASTEXITCODE -ne 0) {
    throw "Graph generation failed with exit code $LASTEXITCODE"
}

Write-Host "Controlled words-only baseline and comparison graphs completed."
