@echo off
rem Start FDA automatically when you log in to Windows (undo with autostart_off.bat).
set "TARGET=%~dp0start_fda.bat"
powershell -NoProfile -Command "$s=(New-Object -ComObject WScript.Shell).CreateShortcut([Environment]::GetFolderPath('Startup')+'\FDA.lnk'); $s.TargetPath='%TARGET%'; $s.WorkingDirectory='%~dp0'; $s.WindowStyle=7; $s.Save()"
echo FDA will start at login.
pause
