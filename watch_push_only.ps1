param(
  [string]$Dir = "E:\AI\DeepFaceLab\scripts",
  [string]$Remote = "origin",
  [string]$Branch = "main",
  [int]$Quiet = 5,              # debounce seconds
  [int]$MinPush = 20,           # min seconds between pushes
  [int]$Poll = 60,              # periodic poll
  [string]$LogPath = "E:\AI\DeepFaceLab\scripts\.pushwatch.log",
  [string]$PidFilePath = "E:\AI\DeepFaceLab\scripts\.pushwatch.pid"
)

function Log($m){$t=(Get-Date).ToString("s");"$t  $m" | Tee-Object -FilePath $LogPath -Append}

# ---- preflight ----
if (-not (Test-Path $Dir)) { throw "Dir not found: $Dir" }
Set-Location $Dir
if (-not (Test-Path ".git")) { throw "This folder is not a git repo. Run git init & add remote first." }
$remoteUrl = git remote get-url $Remote 2>$null
if (-not $remoteUrl) { throw "Remote '$Remote' not found. Add with: git remote add origin https://github.com/<you>/<repo>.git" }

git config push.default current  | Out-Null   # push only; never pull
git config pull.rebase false     | Out-Null

# single-instance guard
if (Test-Path $PidFilePath) {
  $old = Get-Content $PidFilePath -ErrorAction SilentlyContinue
  if ($old -and (Get-Process -Id ([int]$old) -ErrorAction SilentlyContinue)) { throw "Watcher already running (PID $old)" }
}
$PID | Out-File $PidFilePath -Encoding ascii

# file system watcher
$fsw = New-Object IO.FileSystemWatcher $Dir, "*.*"
$fsw.IncludeSubdirectories = $true
$fsw.NotifyFilter = [IO.NotifyFilters]'FileName,LastWrite,Size,DirectoryName'
$fsw.EnableRaisingEvents = $true

# simple event queue
$queue = New-Object 'System.Collections.Concurrent.ConcurrentQueue[object]'
$onEvent = { param($s,$e) $queue.Enqueue($e) }
$subs = @(
  Register-ObjectEvent $fsw Changed -Action $onEvent,
  Register-ObjectEvent $fsw Created -Action $onEvent,
  Register-ObjectEvent $fsw Deleted -Action $onEvent,
  Register-ObjectEvent $fsw Renamed -Action $onEvent
)

$lastPush = Get-Date "2000-01-01"
$lastPoll = Get-Date "2000-01-01"
Log "Push-only watcher started. Repo: $remoteUrl"

try {
  while ($true) {
    Start-Sleep -Seconds $Quiet
    $need = $false

    if (-not $queue.IsEmpty) { while ($queue.TryDequeue([ref]$null)) { } ; $need = $true ; Log "FS changes detected" }
    if (((Get-Date)-$lastPoll).TotalSeconds -ge $Poll) { $lastPoll = Get-Date ; $need = $true ; Log "Polling..." }

    if ($need) {
      # Stage everything; .gitignore whitelist controls what actually gets tracked
      git add -A 2>$null
      $st = git status --porcelain
      if ($st) {
        Log "Committing..."
        git commit -m ("auto: " + (Get-Date -Format s)) | Out-Null
      } else {
        Log "No changes"
      }

      if (((Get-Date)-$lastPush).TotalSeconds -ge $MinPush) {
        try {
          Log "Pushing..."
          git push $Remote HEAD:$Branch | Out-Null
          $lastPush = Get-Date
          Log "Push OK"
        } catch {
          # Still push-only: update remote if it diverged; never pull or checkout locally
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
  foreach ($s in $subs) { Unregister-Event -SourceIdentifier $s.Name -ErrorAction SilentlyContinue }
  $fsw.EnableRaisingEvents = $false; $fsw.Dispose()
  Remove-Item $PidFilePath -ErrorAction SilentlyContinue
  Log "Watcher stopped"
}
