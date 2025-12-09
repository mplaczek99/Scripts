<#
Debloat / tune-up script for IDEAPAD (Windows 11)

What it does by default:
- Creates a restore point (if enabled)
- Disables Lenovo telemetry/utility services
- Disables Microsoft Edge auto-update services & tasks
- Removes Web Experience (widgets/feed) app, which uses Edge WebView2
- Optional section (off by default): disable Hyper-V and WSL features

Run this script as Administrator.
#>

# ---------------------------
# 0. Safety: Require admin
# ---------------------------
$principal = New-Object Security.Principal.WindowsPrincipal `
    ([Security.Principal.WindowsIdentity]::GetCurrent())

if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Error "You must run this script as Administrator. Right-click PowerShell and choose 'Run as administrator'."
    exit 1
}

Write-Host "Running as Administrator ✔" -ForegroundColor Green

# ---------------------------
# 1. Try to create a restore point
# ---------------------------
Write-Host "Creating a system restore point (if System Protection is enabled)..." -ForegroundColor Cyan
try {
    Checkpoint-Computer -Description "Pre-Debloat Script Restore Point" -RestorePointType "MODIFY_SETTINGS"
    Write-Host "Restore point created." -ForegroundColor Green
} catch {
    Write-Warning "Could not create restore point. System Protection may be disabled. Continuing anyway."
}

# ---------------------------
# 2. Disable Lenovo telemetry / helper services
# ---------------------------
Write-Host "`nDisabling Lenovo background / telemetry services (safe)..." -ForegroundColor Cyan

$lenovoServices = @(
    "ImControllerService",    # System Interface Foundation Service
    "LITSSVC"                 # Lenovo Notebook ITS Service
    # Add more here if you see them in 'services.msc'
)

foreach ($svcName in $lenovoServices) {
    $svc = Get-Service -Name $svcName -ErrorAction SilentlyContinue
    if ($null -ne $svc) {
        Write-Host " - Disabling service $svcName (current status: $($svc.Status))"
        try {
            if ($svc.Status -ne 'Stopped') {
                Stop-Service -Name $svcName -Force -ErrorAction SilentlyContinue
            }
            Set-Service -Name $svcName -StartupType Disabled
        } catch {
            Write-Warning "   Failed to modify service $svcName : $($_.Exception.Message)"
        }
    } else {
        Write-Host " - Service $svcName not found, skipping."
    }
}

# ---------------------------
# 3. Disable Microsoft Edge auto-update services & tasks
# ---------------------------
Write-Host "`nDisabling Microsoft Edge auto-update services..." -ForegroundColor Cyan

$edgeServices = @("edgeupdate", "edgeupdatem")

foreach ($svcName in $edgeServices) {
    Write-Host " - Handling service $svcName"
    try {
        sc.exe stop $svcName  | Out-Null
        sc.exe config $svcName start= disabled | Out-Null
    } catch {
        Write-Warning "   Could not modify $svcName : $($_.Exception.Message)"
    }
}

Write-Host "`nDisabling Microsoft Edge scheduled tasks (if present)..." -ForegroundColor Cyan

$edgeTaskPaths = @(
    "\Microsoft\Edge\Update\*",
    "\Microsoft\Edge\*"
)

foreach ($taskPattern in $edgeTaskPaths) {
    try {
        $tasks = Get-ScheduledTask -TaskPath $taskPattern -ErrorAction SilentlyContinue
        if ($tasks) {
            foreach ($t in $tasks) {
                Write-Host " - Disabling task: $($t.TaskName)"
                Disable-ScheduledTask -TaskName $t.TaskName -TaskPath $t.TaskPath -ErrorAction SilentlyContinue
            }
        } else {
            Write-Host " - No tasks matching $taskPattern"
        }
    } catch {
        Write-Warning "   Could not enumerate tasks for $taskPattern : $($_.Exception.Message)"
    }
}

# ---------------------------
# 4. Remove WebExperience pack (widgets / feeds) to cut WebView2 bloat
# ---------------------------
Write-Host "`nRemoving Web Experience pack (Widgets / News & Interests)..." -ForegroundColor Cyan
Write-Host "This cuts WebView2 background usage. If you want it back, you can reinstall from Microsoft Store." -ForegroundColor Yellow

try {
    $webExp = Get-AppxPackage -AllUsers *WebExperience* -ErrorAction SilentlyContinue
    if ($webExp) {
        foreach ($pkg in $webExp) {
            Write-Host " - Removing $($pkg.Name) for all users"
            Remove-AppxPackage -Package $pkg.PackageFullName -AllUsers -ErrorAction SilentlyContinue
        }
    } else {
        Write-Host " - No WebExperience package found, skipping."
    }
} catch {
    Write-Warning "   Could not remove WebExperience: $($_.Exception.Message)"
}

# ---------------------------
# 5. OPTIONAL: Disable virtualization features (Hyper-V, WSL)
# ---------------------------
# Set this to $true if you want to completely turn off Hyper-V & WSL (frees RAM, handles, interrupts)
$DisableVirtualizationFeatures = $false

if ($DisableVirtualizationFeatures) {
    Write-Host "`nDisabling Hyper-V and WSL (this requires a reboot to fully apply)..." -ForegroundColor Cyan

    $featuresToDisable = @(
        "Microsoft-Hyper-V-All",
        "Microsoft-Windows-Subsystem-Linux"
    )

    foreach ($feat in $featuresToDisable) {
        Write-Host " - Disabling feature: $feat"
        try {
            # DISM command
            & dism.exe /online /disable-feature /featurename:$feat /norestart | Out-Null
        } catch {
            Write-Warning "   Failed to disable $feat : $($_.Exception.Message)"
        }
    }
} else {
    Write-Host "`n[INFO] Skipping Hyper-V / WSL disable step (set `$DisableVirtualizationFeatures = `$true to turn them off)." -ForegroundColor DarkYellow
}

# ---------------------------
# 6. Light Xbox / media background cleanup (safe, optional)
# ---------------------------
Write-Host "`nDisabling some non-essential Xbox / media services (they are usually stopped anyway)..." -ForegroundColor Cyan

$optionalServices = @(
    "WMPNetworkSvc",     # Windows Media Player Network Sharing Service
    "XblAuthManager",    # Xbox Live Auth Manager
    "XblGameSave",       # Xbox Live Game Save
    "XboxGipSvc",        # Xbox Accessory Management Service
    "XboxNetApiSvc"      # Xbox Live Networking Service
)

foreach ($svcName in $optionalServices) {
    $svc = Get-Service -Name $svcName -ErrorAction SilentlyContinue
    if ($svc) {
        Write-Host " - Disabling service $svcName (current status: $($svc.Status))"
        try {
            if ($svc.Status -ne 'Stopped') {
                Stop-Service -Name $svcName -Force -ErrorAction SilentlyContinue
            }
            Set-Service -Name $svcName -StartupType Disabled
        } catch {
            Write-Warning "   Failed to modify $svcName : $($_.Exception.Message)"
        }
    }
}

# ---------------------------
# 7. Summary
# ---------------------------
Write-Host "`nAll steps completed." -ForegroundColor Green
Write-Host "Recommended: Reboot your system so all service and feature changes fully apply." -ForegroundColor Yellow
