; SoundboardEZ NSIS installer (offline, per-user)
; Requires NSIS with nsisunz plugin (bundled with standard NSIS)

!define APPNAME "SoundboardEZ"
!define COMPANY "SoundboardEZ"
!define APPVERSION "1.0.0"
!define EXE_NAME "SoundboardEZ.exe"
!define INSTALLER_NAME "SoundboardEZ-Setup.exe"
!define APPDIR "$LOCALAPPDATA\\${APPNAME}"
!define STARTMENU_FOLDER "${APPNAME}"

OutFile "${INSTALLER_NAME}"
RequestExecutionLevel user
InstallDir "${APPDIR}"
ShowInstDetails show
ShowUnInstDetails show

Var StartMenuFolder

Page directory
Page instfiles
UninstPage uninstConfirm
UninstPage instfiles

Section "Install"
  SetOutPath "$INSTDIR"

  ; Main app
  File "dist\\SoundboardEZ.exe"

  ; Optional sounds (if present)
  IfFileExists "sounds\\*.*" 0 +3
    CreateDirectory "$INSTDIR\\sounds"
    File /r "sounds\\*.*"

  ; Assets
  IfFileExists "assets\\app.ico" 0 +2
    File "/oname=$INSTDIR\\app.ico" "assets\\app.ico"

  IfFileExists "assets\\VBCABLE_Driver_Pack43.zip" 0 +2
    File "/oname=$INSTDIR\\VBCABLE_Driver_Pack43.zip" "assets\\VBCABLE_Driver_Pack43.zip"

  ; README
  IfFileExists "README_INSTALL.txt" 0 +2
    File "/oname=$INSTDIR\\README_INSTALL.txt" "README_INSTALL.txt"

  ; Extract VB-Cable pack (if bundled)
  IfFileExists "$INSTDIR\\VBCABLE_Driver_Pack43.zip" 0 +8
    CreateDirectory "$INSTDIR\\vb-cable"
    StrCpy $0 "$\"$SYSDIR\\WindowsPowerShell\\v1.0\\powershell.exe$\" -NoProfile -ExecutionPolicy Bypass -Command $\"Expand-Archive -LiteralPath '$INSTDIR\\VBCABLE_Driver_Pack43.zip' -DestinationPath '$INSTDIR\\vb-cable' -Force$\""
    ExecWait $0
    IfFileExists "$INSTDIR\\vb-cable\\VBCABLE_Setup_x64.exe" 0 +2
      ExecWait '"$INSTDIR\\vb-cable\\VBCABLE_Setup_x64.exe"'
    IfFileExists "$INSTDIR\\vb-cable\\VBCABLE_Setup.exe" 0 +2
      ExecWait '"$INSTDIR\\vb-cable\\VBCABLE_Setup.exe"'

  ; Shortcuts
  StrCpy $StartMenuFolder "${STARTMENU_FOLDER}"
  CreateDirectory "$SMPROGRAMS\\$StartMenuFolder"
  CreateShortCut "$SMPROGRAMS\\$StartMenuFolder\\${APPNAME}.lnk" "$INSTDIR\\${EXE_NAME}" "" "$INSTDIR\\app.ico"
  CreateShortCut "$DESKTOP\\${APPNAME}.lnk" "$INSTDIR\\${EXE_NAME}" "" "$INSTDIR\\app.ico"

SectionEnd

Section "Uninstall"
  ; Remove shortcuts
  Delete "$SMPROGRAMS\\${STARTMENU_FOLDER}\\${APPNAME}.lnk"
  RMDir "$SMPROGRAMS\\${STARTMENU_FOLDER}"
  Delete "$DESKTOP\\${APPNAME}.lnk"

  ; Remove installed files (leave vb-cable drivers intact)
  Delete "$INSTDIR\\${EXE_NAME}"
  Delete "$INSTDIR\\app.ico"
  Delete "$INSTDIR\\README_INSTALL.txt"
  Delete "$INSTDIR\\VBCABLE_Driver_Pack43.zip"
  RMDir /r "$INSTDIR\\sounds"
  RMDir /r "$INSTDIR\\vb-cable"
  RMDir "$INSTDIR"
SectionEnd

; Uninstaller
Section -Post
  WriteUninstaller "$INSTDIR\\Uninstall.exe"
SectionEnd
