# Define the IPs and tab names
$firstPart = "192.168.0."
$ipNames = @{
                "52" = "RICOH" 
                "54" = "Controller" 
                "55:8000" = "Canon C710A" 
                "57" = "Canon 6755i" 
                "58" = "CFO" 
                "59" = "QA" 
                "60" = "Sales"
                "61" = "RD"
                "62" = "Customer Service"
                "65" = "Supervisor"
                "66" = "Shipping"
                "67" = "QA 2"
                "69" = "Front Desk 1"
                "93" = "Front Desk 2"
	        "200" = "Store Office"
            }

# Open Microsoft Edge with multiple tabs
foreach ($ip in $ipNames.Keys) {
    $ipAddress = $firstPart + $ip
    $tabName = $ipNames[$ip]
    Write-Host "Opening Microsoft Edge at IP: $ipAddress - Tab Name: $tabName"
    $url = "http://$ipAddress"

    # Open a new tab in the Microsoft Edge instance
    Start-Process msedge -ArgumentList "-new-tab", $url

# Does not work as intended
#   # Set the tab name using PowerShell commands
#   $shell = New-Object -ComObject Shell.Application
#   $windows = $shell.Windows()
#   $edgeProcess = $windows | Where-Object { $_.FullName -like "*msedge.exe" } | Select-Object -First 1
#   $tab = $edgeProcess.Document.IHTMLDocument3_getElementsByTagName('title') | Select-Object -First 1
#   $tab.innerText = $tabName
}