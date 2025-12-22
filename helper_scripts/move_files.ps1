<#
.SYNOPSIS
    Recursively moves all .wav files from subdirectories to the script's directory.

.DESCRIPTION
    Purpose: This script automatically moves all .wav files from the script's current directory
    and all its subdirectories to the script's root location. No user interaction required.
    
    Logic Flow:
    1. Determine script's location (used as both source and destination)
    2. Recursively search for all .wav files in subdirectories
    3. Move each file to script location, handling name conflicts
    4. Provide detailed progress feedback and summary
    
    Edge Cases Handled:
    - File name conflicts: appends timestamp to prevent overwrites
    - Files already in destination: skips to avoid self-move
    - Empty results: informs user if no files found
    - Errors: logs but continues processing remaining files
    
.EXAMPLE
    .\MoveFilesByType.ps1
    
    Result: All .wav files from subdirectories moved to script's directory immediately
#>

# Get the script's directory - this is both source and destination
$scriptLocation = Split-Path -Parent $MyInvocation.MyCommand.Path

# Default settings
$extensions = @("wav")
$sourceDirectory = $scriptLocation

Write-Host "================================" -ForegroundColor Cyan
Write-Host "AUTO FILE MOVER" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host "Script location: $scriptLocation" -ForegroundColor Cyan
Write-Host "File types: $($extensions -join ', ')" -ForegroundColor Green
Write-Host "Source directory: $sourceDirectory" -ForegroundColor Green
Write-Host ""
Write-Host "Searching for files..." -ForegroundColor Cyan
Write-Host ""

# Initialize counters for summary
$movedCount = 0
$errorCount = 0
$skippedCount = 0

# Recursively find all files with specified extensions
# -Recurse handles all subdirectories automatically
foreach ($extension in $extensions) {
    Write-Host "Processing *.$extension files..." -ForegroundColor Cyan
    
    # Get all files with current extension recursively
    $files = Get-ChildItem -Path $sourceDirectory -Filter "*.$extension" -File -Recurse -ErrorAction SilentlyContinue
    
    foreach ($file in $files) {
        try {
            # Skip if file is already in the script location (avoid moving to itself)
            if ($file.DirectoryName -eq $scriptLocation) {
                Write-Host "  Skipped: $($file.Name) (already in destination)" -ForegroundColor Gray
                $skippedCount++
                continue
            }
            
            # Construct destination path
            $destinationPath = Join-Path -Path $scriptLocation -ChildPath $file.Name
            
            # Handle name conflicts: append timestamp if file exists
            if (Test-Path -Path $destinationPath) {
                $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
                $nameWithoutExt = [System.IO.Path]::GetFileNameWithoutExtension($file.Name)
                $ext = [System.IO.Path]::GetExtension($file.Name)
                $newName = "${nameWithoutExt}_${timestamp}${ext}"
                $destinationPath = Join-Path -Path $scriptLocation -ChildPath $newName
                
                Write-Host "  Conflict resolved: $($file.Name) -> $newName" -ForegroundColor Yellow
            }
            
            # Move the file (not copy)
            Move-Item -Path $file.FullName -Destination $destinationPath -Force
            Write-Host "  Moved: $($file.FullName)" -ForegroundColor Green
            $movedCount++
            
        } catch {
            # Error handling: log but continue processing other files
            Write-Host "  ERROR moving $($file.FullName): $($_.Exception.Message)" -ForegroundColor Red
            $errorCount++
        }
    }
}

# Summary report
Write-Host ""
Write-Host "================================" -ForegroundColor Cyan
Write-Host "OPERATION COMPLETE" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host "Files moved successfully: $movedCount" -ForegroundColor Green
Write-Host "Files skipped (already at destination): $skippedCount" -ForegroundColor Gray
Write-Host "Errors encountered: $errorCount" -ForegroundColor $(if ($errorCount -gt 0) { "Red" } else { "Green" })
Write-Host "Destination: $scriptLocation" -ForegroundColor Cyan
Write-Host ""

# Pause to allow user to read results
Write-Host "Press any key to exit..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")