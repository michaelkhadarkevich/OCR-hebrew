param(
    [string]$DataDirectory = (Join-Path $PSScriptRoot 'data'),
    [string]$OutputPath = (Join-Path $PSScriptRoot 'hebrew_letter_statistics.xlsx')
)

$ErrorActionPreference = 'Stop'

function Read-Utf8Label {
    param([string]$Path)
    $strictUtf8 = New-Object System.Text.UTF8Encoding($false, $true)
    try {
        return [System.IO.File]::ReadAllText($Path, $strictUtf8).Trim()
    }
    catch {
        return [System.IO.File]::ReadAllText($Path, [System.Text.Encoding]::Default).Trim()
    }
}

function Get-Median {
    param([int[]]$Values)
    if (-not $Values -or $Values.Count -eq 0) { return 0 }
    $ordered = @($Values | Sort-Object)
    $middle = [int][math]::Floor($ordered.Count / 2)
    if ($ordered.Count % 2) { return $ordered[$middle] }
    return ($ordered[$middle - 1] + $ordered[$middle]) / 2.0
}

function Add-WorksheetData {
    param(
        $Workbook,
        [string]$Name,
        [object[][]]$Rows,
        [bool]$RightToLeft = $true
    )
    $sheet = $Workbook.Worksheets.Add()
    $sheet.Name = $Name
    if ($Rows.Count -eq 0) { return $sheet }

    $columns = ($Rows | ForEach-Object { $_.Count } | Measure-Object -Maximum).Maximum
    Write-Host "Writing worksheet '$Name' ($($Rows.Count) rows)..."
    for ($row = 0; $row -lt $Rows.Count; $row++) {
        for ($column = 0; $column -lt $Rows[$row].Count; $column++) {
            $value = $Rows[$row][$column]
            if ($null -ne $value) {
                $sheet.Cells.Item($row + 1, $column + 1).Value2 = [System.Convert]::ToString($value, [System.Globalization.CultureInfo]::InvariantCulture)
            }
        }
    }
    $range = $sheet.Range($sheet.Cells(1, 1), $sheet.Cells($Rows.Count, $columns))
    $header = $sheet.Range($sheet.Cells(1, 1), $sheet.Cells(1, $columns))
    $header.Font.Bold = $true
    $header.Interior.Color = 0xD9EAD3
    $header.AutoFilter() | Out-Null
    $range.VerticalAlignment = -4160
    $range.Borders.LineStyle = 1
    $sheet.Columns.AutoFit() | Out-Null
    for ($column = 1; $column -le $columns; $column++) {
        if ($sheet.Columns.Item($column).ColumnWidth -gt 45) {
            $sheet.Columns.Item($column).ColumnWidth = 45
        }
    }
    $sheet.DisplayRightToLeft = $RightToLeft
    return $sheet
}

$datasetNames = @('arielOnlyWord', 'agadaOnlyWord', 'RockOnlyWord')
$hebrewLetters = @(0x05D0,0x05D1,0x05D2,0x05D3,0x05D4,0x05D5,0x05D6,0x05D7,0x05D8,0x05D9,0x05DB,0x05DA,0x05DC,0x05DE,0x05DD,0x05E0,0x05DF,0x05E1,0x05E2,0x05E4,0x05E3,0x05E6,0x05E5,0x05E7,0x05E8,0x05E9,0x05EA) | ForEach-Object { [char]$_ }
$finalToBase = @{
    ([char]0x05DA) = [char]0x05DB
    ([char]0x05DD) = [char]0x05DE
    ([char]0x05DF) = [char]0x05E0
    ([char]0x05E3) = [char]0x05E4
    ([char]0x05E5) = [char]0x05E6
}

$records = New-Object System.Collections.Generic.List[object]
$qualityRows = New-Object System.Collections.Generic.List[object]

foreach ($dataset in $datasetNames) {
    $root = Join-Path $DataDirectory $dataset
    if (-not (Test-Path -LiteralPath $root)) { throw "Dataset directory not found: $root" }
    $files = @(Get-ChildItem -LiteralPath $root -Recurse -File -Filter '*.txt')
    foreach ($file in $files) {
        $text = Read-Utf8Label -Path $file.FullName
        $chars = $text.ToCharArray()
        $letters = @($chars | Where-Object { $_ -match '[\u05D0-\u05EA]' })
        $marks = @($chars | Where-Object { $_ -match '[\u0591-\u05C7]' })
        $digits = @($chars | Where-Object { [char]::IsDigit($_) })
        $other = @($chars | Where-Object { ($_ -notmatch '[\u0591-\u05EA]') -and -not [char]::IsDigit($_) -and -not [char]::IsWhiteSpace($_) })
        $imagePath = [System.IO.Path]::ChangeExtension($file.FullName, '.png')
        $relativePath = $file.FullName.Substring($DataDirectory.Length).TrimStart('\')
        $records.Add([pscustomobject]@{
            Dataset = $dataset
            Label = $text
            HebrewLetters = $letters.Count
            NiqqudAndMarks = $marks.Count
            Digits = $digits.Count
            OtherSymbols = $other.Count
            IsEmpty = [string]::IsNullOrWhiteSpace($text)
            HasImage = Test-Path -LiteralPath $imagePath
            RelativePath = $relativePath
        })
        if ([string]::IsNullOrWhiteSpace($text) -or -not (Test-Path -LiteralPath $imagePath) -or $letters.Count -eq 0) {
            $qualityRows.Add([pscustomobject]@{
                Dataset = $dataset
                Label = $text
                Issue = @(
                    if ([string]::IsNullOrWhiteSpace($text)) { 'Empty label' }
                    if (-not (Test-Path -LiteralPath $imagePath)) { 'Missing PNG pair' }
                    if ($letters.Count -eq 0 -and -not [string]::IsNullOrWhiteSpace($text)) { 'No Hebrew letters' }
                ) -join '; '
                RelativePath = $relativePath
            })
        }
    }
}

$summaryRows = New-Object System.Collections.Generic.List[object[]]
$summaryRows.Add(@('Dataset','Label files','Non-empty labels','Unique labels','Hebrew letters','Avg. letters/label','Median letters/label','Min','Max','Niqqud/marks','Digits','Other symbols','Missing PNG pairs'))
foreach ($dataset in $datasetNames + 'ALL') {
    $subset = if ($dataset -eq 'ALL') { @($records.ToArray()) } else { @($records | Where-Object Dataset -eq $dataset) }
    $lengths = @($subset | ForEach-Object HebrewLetters)
    $summaryRows.Add(@(
        $dataset,
        $subset.Count,
        @($subset | Where-Object { -not $_.IsEmpty }).Count,
        @($subset.Label | Where-Object { $_ } | Sort-Object -Unique).Count,
        ($subset.HebrewLetters | Measure-Object -Sum).Sum,
        [math]::Round(($subset.HebrewLetters | Measure-Object -Average).Average, 3),
        (Get-Median -Values $lengths),
        ($lengths | Measure-Object -Minimum).Minimum,
        ($lengths | Measure-Object -Maximum).Maximum,
        ($subset.NiqqudAndMarks | Measure-Object -Sum).Sum,
        ($subset.Digits | Measure-Object -Sum).Sum,
        ($subset.OtherSymbols | Measure-Object -Sum).Sum,
        @($subset | Where-Object { -not $_.HasImage }).Count
    ))
}

$frequencyRows = New-Object System.Collections.Generic.List[object[]]
$frequencyRows.Add(@('Letter','Final-form base','Overall count','Overall %') + @($datasetNames | ForEach-Object { "$_ count" }) + @($datasetNames | ForEach-Object { "$_ %" }))
$allLetterCount = ($records.HebrewLetters | Measure-Object -Sum).Sum
foreach ($letter in $hebrewLetters) {
    $counts = @()
    $percentages = @()
    foreach ($dataset in $datasetNames) {
        $labels = ($records | Where-Object Dataset -eq $dataset).Label -join ''
        $count = ([regex]::Matches($labels, [regex]::Escape($letter))).Count
        $datasetTotal = (($records | Where-Object Dataset -eq $dataset).HebrewLetters | Measure-Object -Sum).Sum
        $counts += $count
        $percentages += if ($datasetTotal) { [math]::Round(100 * $count / $datasetTotal, 4) } else { 0 }
    }
    $overallCount = ($counts | Measure-Object -Sum).Sum
    $frequencyRows.Add(@($letter, $(if ($finalToBase.ContainsKey($letter)) {$finalToBase[$letter]} else {$letter}), $overallCount, [math]::Round(100 * $overallCount / $allLetterCount, 4)) + $counts + $percentages)
}

$lengthRows = New-Object System.Collections.Generic.List[object[]]
$lengthRows.Add(@('Hebrew letter length') + @($datasetNames | ForEach-Object { "$_ labels" }) + @('All labels'))
$maxLength = ($records.HebrewLetters | Measure-Object -Maximum).Maximum
for ($length = 0; $length -le $maxLength; $length++) {
    $counts = @($datasetNames | ForEach-Object { $name = $_; @($records | Where-Object { $_.Dataset -eq $name -and $_.HebrewLetters -eq $length }).Count })
    $lengthRows.Add(@($length) + $counts + @(@($records | Where-Object HebrewLetters -eq $length).Count))
}

$wordRows = New-Object System.Collections.Generic.List[object[]]
$wordRows.Add(@('Rank','Dataset','Label','Occurrences','Hebrew letter length'))
foreach ($dataset in $datasetNames + 'ALL') {
    $subset = if ($dataset -eq 'ALL') { @($records.ToArray()) } else { @($records | Where-Object Dataset -eq $dataset) }
    $rank = 0
    $subset | Where-Object { $_.Label } | Group-Object Label | Sort-Object -Property @{Expression='Count';Descending=$true}, @{Expression='Name';Descending=$false} | Select-Object -First 100 | ForEach-Object {
        $rank++
        $letterLength = @($_.Name.ToCharArray() | Where-Object { $_ -match '[\u05D0-\u05EA]' }).Count
        $wordRows.Add(@($rank, $dataset, $_.Name, $_.Count, $letterLength))
    }
}

$issueRows = New-Object System.Collections.Generic.List[object[]]
$issueRows.Add(@('Dataset','Issue','Affected labels'))
foreach ($dataset in $datasetNames + 'ALL') {
    $subset = if ($dataset -eq 'ALL') { @($qualityRows.ToArray()) } else { @($qualityRows | Where-Object Dataset -eq $dataset) }
    foreach ($group in @($subset | Group-Object Issue | Sort-Object Count -Descending)) {
        $issueRows.Add(@($dataset,$group.Name,$group.Count))
    }
}

$excel = $null
$workbook = $null
try {
    $excel = New-Object -ComObject Excel.Application
    $excel.Visible = $false
    $excel.DisplayAlerts = $false
    $workbook = $excel.Workbooks.Add()
    while ($workbook.Worksheets.Count -gt 1) { $workbook.Worksheets.Item($workbook.Worksheets.Count).Delete() }
    $blankSheet = $workbook.Worksheets.Item(1)

    $summarySheet = Add-WorksheetData -Workbook $workbook -Name 'Summary' -Rows $summaryRows
    $blankSheet.Delete()
    $frequencySheet = Add-WorksheetData -Workbook $workbook -Name 'Letter Frequency' -Rows $frequencyRows
    $lengthSheet = Add-WorksheetData -Workbook $workbook -Name 'Word Lengths' -Rows $lengthRows
    $wordSheet = Add-WorksheetData -Workbook $workbook -Name 'Top Labels' -Rows $wordRows
    $qualitySheet = Add-WorksheetData -Workbook $workbook -Name 'Quality Checks' -Rows $issueRows

    $frequencySheet.Columns.Item(4).NumberFormat = '0.0000'
    for ($column = 8; $column -le 10; $column++) { $frequencySheet.Columns.Item($column).NumberFormat = '0.0000' }
    $summarySheet.Columns.Item(6).NumberFormat = '0.000'
    $summarySheet.Columns.Item(7).NumberFormat = '0.0'

    $chartObject = $frequencySheet.ChartObjects().Add(650, 20, 600, 360)
    $chart = $chartObject.Chart
    $chart.SetSourceData($frequencySheet.Range('A1', "C$($frequencyRows.Count)"))
    $chart.ChartType = 51
    $chart.HasTitle = $true
    $chart.ChartTitle.Text = 'Overall Hebrew Letter Counts'

    $absoluteOutput = [System.IO.Path]::GetFullPath($OutputPath)
    $workbook.SaveAs($absoluteOutput, 51)
    $detailsCsv = [System.IO.Path]::ChangeExtension($absoluteOutput, '.label_details.csv')
    $records | Export-Csv -LiteralPath $detailsCsv -NoTypeInformation -Encoding UTF8
    $qualityCsv = [System.IO.Path]::ChangeExtension($absoluteOutput, '.quality_details.csv')
    $qualityRows | Export-Csv -LiteralPath $qualityCsv -NoTypeInformation -Encoding UTF8
    Write-Output "Created: $absoluteOutput"
    Write-Output "Created: $detailsCsv"
    Write-Output "Created: $qualityCsv"
    Write-Output "Labels analyzed: $($records.Count)"
    Write-Output "Hebrew letters counted: $allLetterCount"
    Write-Output "Quality issues: $($qualityRows.Count)"
}
finally {
    if ($workbook) { $workbook.Close($false) }
    if ($excel) { $excel.Quit() }
    if ($qualitySheet) { [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($qualitySheet) }
    if ($wordSheet) { [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($wordSheet) }
    if ($lengthSheet) { [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($lengthSheet) }
    if ($frequencySheet) { [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($frequencySheet) }
    if ($summarySheet) { [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($summarySheet) }
    if ($workbook) { [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($workbook) }
    if ($excel) { [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($excel) }
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}
