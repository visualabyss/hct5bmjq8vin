param(
  [string]$Dir = "E:\AI\DeepFaceLab\scripts",
  [string]$Remote = "origin",
  [string]$Branch = "main",
  [int]$Quiet = 5,            # debounce seconds
  [int]$MinPush = 20,         # don't push more often than this
  [int]$Poll = 60,            # periodic check even if no FS event
  [string]$Log = "E:\AI\DeepFaceLab\scripts\.pushwatch.log",
  [string]$Pid = "E:\AI\DeepFaceLab\scripts\.pushwatch.pid"
)

function Log($m){$t=(Get-Date).ToString("s");"$t  $m" | Tee-Object -FilePath $Log -Append}

Set-Location $Dir
git config push.default current      | Out-Null   # never pulls
git config pull.rebase false         | Out-Null   # belt & suspenders

# single-instance guard
if (Test-Path $Pid) {
  $old = Get-Content $Pid -ErrorAction SilentlyContinue
  if ($old -and (Get-Process -Id ([int]$old) -ErrorAction SilentlyContinue)) { throw "Watcher already running (PID $old)" }
}
$PID | Out-File $Pid -Encoding ascii

$fsw = New-Object IO.FileSystemWatcher $Dir, "*.*"
$fsw.IncludeSubdirectories = $true
$fsw.NotifyFilter = [IO.NotifyFilters]'FileName,LastWrite,Size,DirectoryName'
$fsw.EnableRaisingEvents = $true

$q = [System.Collections.Concurrent.ConcurrentQueue[object]]::new()
$act = { param($s,$e) $q.Enqueue($e) }
$subs = @(
  Register-ObjectEvent $fsw Changed -Action $act,
  Register-ObjectEvent $fsw Created -Action $act,
  Register-ObjectEvent $fsw Deleted -Action $act,
  Register-ObjectEvent $fsw Renamed -Action $act
)

$lastPush = Get-Date "2000-01-01"
$lastPoll = Get-Date "2000-01-01"
Log "Push-only watcher started. Repo: $(git remote get-url $Remote)"

try {
  while ($true) {
    Start-Sleep -Seconds $Quiet
    $need = $false
    if (-not $q.IsEmpty) { while ($q.TryDequeue([ref]$null)){}; $need = $true; Log "FS changes detected" }
    if (((Get-Date)-$lastPoll).TotalSeconds -ge $Poll) { $lastPoll=Get-Date; $need=$true; Log "Polling..." }

    if ($need) {
      # Stage all changes (whitelist in .gitignore controls what gets tracked)
      git add -A 2>$null
      $st = git status --porcelain
      if ($st) { Log "Committing..."; git commit -m ("auto: "+(Get-Date -Format s)) | Out-Null } else { Log "No changes" }

      if (((Get-Date)-$lastPush).TotalSeconds -ge $MinPush) {
        try {
          Log "Pushing..."
          git push $Remote HEAD:$Branch | Out-Null
          $lastPush = Get-Date
          Log "Push OK"
        } catch {
          # No fetch or pull. If remote rejected (you are sole pusher), force-with-lease only updates remote.
          Log "Non-fast-forward; force-with-lease"
          git push --force-with-lease $Remote HEAD:$Branch | Out-Null
          $lastPush = Get-Date
          Log "Force-with-lease OK"
        }
      } else {
        Log "Skip push (debounced)"
      }
    }
  }
}
finally {
  foreach ($s in $subs){ Unregister-Event -SourceIdentifier $s.Name -ErrorAction SilentlyContinue }
  $fsw.EnableRaisingEvents = $false; $fsw.Dispose()
  Remove-Item $Pid -ErrorAction SilentlyContinue
  Log "Watcher stopped"
}
